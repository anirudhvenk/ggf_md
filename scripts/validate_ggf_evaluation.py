"""Focused CPU checks of the GGF evaluator using only synthetic fixtures.

Run with the pinned interpreter and a fresh --output path below the isolated
GGF validation directory. No dataset, trained checkpoint, GPU, or production
output is used. The analytic checks validate the specified numerical adapter;
they do not establish sampling quality for an unconditional learned field.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPOSITORY / "data/ggf/implementation/baseline/evaluate.py"
H = 0.0010004971259352385
SIGMA_MAX = 20.0
PEPTIDES = ["KS", "AT", "LW", "KQ", "NY", "IM", "TD", "HT", "NF", "RL", "ET", "AC", "RV", "HP", "AP"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_config():
    return {
        "data": {"root": "UNUSED_SYNTHETIC_DATA_ONLY"},
        "model": {"hidden_dim": 8, "layers": 1, "score_hidden_dim": 8,
                  "score_layers": 1, "embedding_dim": 2, "heads": 2, "head_dim": 4},
        "training": {"seed": 91},
        "ggf": {"sigma_min": H, "sigma_max": SIGMA_MAX},
        "evaluation": {"methods": ["ggf"], "peptides": ["AT"], "seed": 137,
                       "num_iid_samples": 2, "diffusion_steps": 3,
                       "simulation_steps": 2, "save_every": 1, "replicas": 2,
                       "eval_time": 1e-5, "two_for_one_time": .02,
                       "dt": .0005, "gamma": 1., "temperature": 310.,
                       "chunk_frames": 2, "inference_batch": 2,
                       "bins": 8, "plot_bins": 8, "kde_bw": .2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh validation JSON; temporary synthetic outputs stay beside it")
    parser.add_argument("--baseline-evaluate", type=Path, default=DEFAULT_BASELINE,
                        help="Read-only legacy evaluate.py for AST preservation check")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"--output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Set these before importing scientific libraries. No CUDA availability or
    # device query is necessary, even on a machine running other experiments.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    cache = Path(os.environ.get("MPLCONFIGDIR", output.parent / "evaluation-mpl"))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    sys.path.insert(0, str(REPOSITORY))

    import numpy as np
    import pandas as pd
    import torch
    import evaluate
    from ggf_score_matching import ggf_identity
    from model import DipeptideModel

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    start = time.monotonic()
    report = {
        "status": "running", "scope": "CPU synthetic GGF evaluation validation",
        "quality_claim": "None: the unconditional adapter is checked numerically only.",
        "interpreter": sys.executable, "torch": torch.__version__,
        "threads": torch.get_num_threads(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "repository": str(REPOSITORY), "baseline_evaluate": str(args.baseline_evaluate.resolve()),
        "source_sha256": {name: sha256(REPOSITORY / name) for name in
                          ("evaluate.py", "model.py", "ggf_score_matching.py", "scripts/validate_ggf_evaluation.py")},
        "checks": {}}

    def check(name, condition, **evidence):
        report["checks"][name] = {"passed": bool(condition), **evidence}
        if not condition:
            raise AssertionError(name)

    def expect_rejection(name, operation):
        try:
            operation()
        except ValueError as error:
            check(name, True, message=str(error))
        else:
            check(name, False)

    def close(name, actual, expected, *, rtol=2e-5, atol=2e-5, **evidence):
        actual = torch.as_tensor(actual).double()
        expected = torch.as_tensor(expected).double()
        passed = actual.shape == expected.shape and torch.allclose(actual, expected, rtol=rtol, atol=atol)
        check(name, passed, max_absolute_error=float((actual - expected).abs().max()),
              rtol=rtol, atol=atol, **evidence)

    def features(n, dtype=torch.long):
        one = torch.tensor([[residue, atom, element, index] for index, residue in enumerate((0, 16))
                            for atom, element in enumerate((1, 0, 0, 0, 2))], dtype=dtype)
        return one.unsqueeze(0).expand(n, -1, -1)

    def prescribed_noise(seed, n, steps):
        rng = torch.Generator(device="cpu").manual_seed(seed)
        prior = torch.randn((n, 30), generator=rng)
        increments = [torch.randn((n, 30), generator=rng) for _ in range(steps)]
        return prior, increments, rng.get_state()

    def quadratic_path(n, steps, seed):
        """Compare against the scalar affine recurrence, evaluated in float64.

        For E=||x-mu||^2/(2h^2), the specified adapter gives the time-independent
        contraction 1-2log(R)dt. Independently drawn innovations retain the VE
        time-dependent noise scale. This avoids reproducing evaluator tensor
        drift arithmetic in the reference.
        """
        prior, innovations, rng_state = prescribed_noise(seed, n, steps)
        feature = features(n)
        center = torch.linspace(-.3, .3, 30).repeat(n, 1)
        calls = []

        def quadratic(x, provided_features, t, scalar_time):
            calls.append((x.detach().clone(), t.detach().clone(), scalar_time))
            if not torch.equal(provided_features, feature):
                raise AssertionError("IID sampler changed peptide features")
            return -(x - center) / H**2

        generator = torch.Generator(device="cpu").manual_seed(seed)
        actual = evaluate.reverse_ve_diffusion(quadratic, feature, n, steps, generator,
                                               1., H, SIGMA_MAX)
        log_ratio = math.log(SIGMA_MAX / H)
        dt = 1 / (steps - 1)
        contraction = 1 - 2 * log_ratio * dt
        expected = prior.double() * SIGMA_MAX
        expected_inputs = []
        grid = np.linspace(0., 1., steps, dtype=np.float32)[::-1]
        for scalar_time, innovation in zip(grid, innovations):
            expected_inputs.append(expected.clone())
            sigma = H * math.exp(float(scalar_time) * log_ratio)
            expected = (contraction * expected + (1 - contraction) * center.double()
                        + sigma * math.sqrt(2 * log_ratio * dt) * innovation.double())
        tag = f"quadratic_n{n}_steps{steps}"
        close(tag + "_trajectory", torch.stack([call[0] for call in calls]),
              torch.stack(expected_inputs), rtol=8e-5, atol=2e-3)
        close(tag + "_endpoint", actual, expected, rtol=8e-5, atol=2e-3)
        check(tag + "_grid", len(calls) == steps and calls[0][2] == 1. and calls[-1][2] == 0.
              and all(float(t[0, 0]) == float(s) and bool(torch.all(t == s))
                      for (_, t, _), s in zip(calls, grid)), calls=len(calls),
              first_time=calls[0][2], last_time=calls[-1][2])
        check(tag + "_rng_consumption", torch.equal(generator.get_state(), rng_state),
              gaussian_arrays_consumed=steps + 1, includes_endpoint_noise=True)
        close(tag + "_prior", calls[0][0], prior * SIGMA_MAX, rtol=0., atol=0.,
              gaussian_standard_deviation=SIGMA_MAX)
        if steps == 1000:
            noise_scale = SIGMA_MAX * math.sqrt(2 * log_ratio * dt)
            recovered = (calls[1][0].double() - noise_scale * innovations[0].double()
                         - (1 - contraction) * center.double())
            close("first_step_adapted_quadratic_contraction", recovered,
                  contraction * calls[0][0].double(), rtol=1e-5, atol=1e-5)
            raw_factor = 1 - 2 * log_ratio * (SIGMA_MAX / H)**2 * dt
            check("first_step_stability_scale", 0.98 < contraction < .981 and raw_factor < -7e6,
                  adapted_factor=contraction, unadapted_factor=raw_factor,
                  interpretation="Local analytic contraction only; no learned sampling quality claim")

    try:
        baseline_tree = ast.parse(args.baseline_evaluate.read_text())
        current_tree = ast.parse((REPOSITORY / "evaluate.py").read_text())
        function = lambda tree: next(node for node in tree.body
                                     if isinstance(node, ast.FunctionDef) and node.name == "reverse_diffusion")
        check("legacy_vp_sampler_ast_unchanged",
              ast.dump(function(current_tree), include_attributes=False)
              == ast.dump(function(baseline_tree), include_attributes=False),
              baseline_sha256=sha256(args.baseline_evaluate))
        check("ggf_method_registered", "ggf" in evaluate.SUPPORTED_METHODS)
        quadratic_path(2, 3, 2001)
        quadratic_path(8, 3, 2002)
        quadratic_path(2, 1000, 2003)

        # A zero field has no deterministic motion. A VP x-drift, an omitted
        # t=0 noise draw, or a final denoising pass changes this known trajectory.
        n, steps, seed = 3, 3, 321
        prior, innovations, rng_state = prescribed_noise(seed, n, steps)
        seen = []

        def zero(x, feature, t, scalar_time):
            seen.append(x.detach().clone())
            return torch.zeros_like(x)

        rng = torch.Generator(device="cpu").manual_seed(seed)
        norm = 3.25
        actual = evaluate.reverse_ve_diffusion(zero, features(n), n, steps, rng, norm, H, SIGMA_MAX)
        log_ratio = math.log(SIGMA_MAX / H)
        reference = prior.double() * SIGMA_MAX
        reference_inputs = []
        for scalar_time, innovation in zip(np.linspace(1., 0., steps), innovations):
            reference_inputs.append(reference.clone())
            reference += H * math.exp(float(scalar_time) * log_ratio) * math.sqrt(
                2 * log_ratio / (steps - 1)) * innovation.double()
        close("zero_field_has_no_vp_drift", torch.stack(seen), torch.stack(reference_inputs),
              rtol=2e-6, atol=2e-5)
        close("zero_field_endpoint_noise_and_output_normalization", actual, reference / norm,
              rtol=2e-6, atol=2e-5, norm_factor=norm, denoise=False)
        check("zero_field_rng_consumption", torch.equal(rng.get_state(), rng_state))
        same_rng = torch.Generator(device="cpu").manual_seed(seed)
        unnormalized = evaluate.reverse_ve_diffusion(lambda x, *_: torch.zeros_like(x), features(n),
                                                    n, steps, same_rng, 1., H, SIGMA_MAX)
        close("normalization_applied_once_after_integration", actual, unnormalized / norm,
              rtol=0., atol=0.)

        config = fixture_config()
        torch.manual_seed(119)
        model = DipeptideModel("ggf", config).double().eval().requires_grad_(False)
        check("ggf_single_conservative_expert", len(model.experts) == 1 and model.experts[0].potential
              and model.experts[0].fixed_time == 0.)
        x = torch.randn((2, 30), dtype=torch.float64)
        feature = features(2)
        score = evaluate.make_score(model)
        raw = score(x, feature, torch.zeros((2, 1), dtype=x.dtype), 0.)
        for caller_time in (-.2, .02, .35, .8, 1., 1.2):
            value = score(x, feature, torch.full((2, 1), caller_time, dtype=x.dtype), caller_time)
            close(f"make_score_time_invariant_{caller_time}", value, raw, rtol=0., atol=0.)
        centered = (x.reshape(-1, 10, 3) - x.reshape(-1, 10, 3).mean(1, keepdim=True)).reshape_as(x)
        centered = centered.detach().requires_grad_(True)
        energy = model.experts[0]._differentiable_forward(centered, feature, torch.zeros((2, 1), dtype=x.dtype)).sum()
        direct_force = -torch.autograd.grad(energy, centered)[0]
        close("make_score_equals_negative_energy_gradient", raw, direct_force, rtol=1e-12, atol=1e-12)
        other_feature = feature.clone()
        other_feature[..., 0] = (other_feature[..., 0] + 1) % 20
        changed = score(x, other_feature, torch.zeros((2, 1), dtype=x.dtype), 0.)
        check("peptide_features_retained", float((changed - raw).abs().max()) > 1e-12,
              max_force_change=float((changed - raw).abs().max()))

        # Langevin must consume the same raw F. Its physical scaling is distinct
        # from the IID VE adapter, which was checked above against its recurrence.
        scalar_time = .8
        t = torch.full((2, 1), scalar_time, dtype=x.dtype)
        norm, kbt, force_scale, step_dt = 2.3, 1.7, .2, .03
        raw_simulation = score(x * norm, feature, t, scalar_time)
        next_x, next_v = evaluate.langevin_step(x, torch.zeros_like(x), torch.zeros_like(x),
                                              score, feature, t, scalar_time, norm,
                                              torch.ones(30, dtype=x.dtype), kbt,
                                              torch.tensor(.7, dtype=x.dtype), force_scale, step_dt)
        expected_v = force_scale * kbt * norm * raw_simulation
        close("simulation_uses_raw_ggf_force", next_v, expected_v, rtol=1e-12, atol=1e-12)
        close("simulation_position_update_uses_raw_force", next_x, x + step_dt * expected_v,
              rtol=1e-12, atol=1e-12)

        with tempfile.TemporaryDirectory(prefix="ggf-evaluation-fixtures-", dir=output.parent) as temporary:
            temporary = Path(temporary)
            iid_dir, sim_dir = temporary / "iid", temporary / "simulation"
            fake_model = SimpleNamespace(method="ggf", experts=[SimpleNamespace(
                _inference_cache=False, _factored_attention=False, potential=True)])
            synthetic = np.random.default_rng(57).normal(0., .2, (8, 30)).astype("float32")
            split = {"AT": {"x": synthetic, "features": feature[0].numpy()}}

            def fake_setup(config, split, model, norm, device, compiled):
                rows = len(config["evaluation"]["peptides"]) * config["evaluation"]["replicas"]
                positions = torch.tensor(synthetic[:rows].copy())
                velocity = torch.zeros_like(positions)
                generator = torch.Generator(device="cpu").manual_seed(137)
                return positions, velocity, generator, lambda x, v, noise: (x + .001 * noise, v)

            identity = ggf_identity(config)
            changed_config = deepcopy(config)
            changed_config["ggf"]["sigma_min"] *= 1.01
            check("changed_h_changes_identity", ggf_identity(changed_config) != identity)
            check("ggf_identity_records_unconditional_adapter", identity["noise_conditioned"] is False
                  and identity["bandwidth_h"] == H and identity["fixed_time"] == 0.
                  and identity["iid_sampler"] == "ve_euler_maruyama"
                  and identity["iid_score_adapter"] == "h2_over_sigma2", identity=identity)
            with patch.object(evaluate, "make_score", return_value=lambda x, *_: torch.zeros_like(x)), \
                    patch.object(evaluate, "simulation_setup", side_effect=fake_setup):
                evaluate.generate_iid(config, split, fake_model, 1., "synthetic-checkpoint", iid_dir, device)
                evaluate.simulate(config, split, fake_model, 1., "synthetic-checkpoint", sim_dir, device)
                iid_metadata = json.loads((iid_dir / "iid_metadata.json").read_text())
                sim_metadata = json.loads((sim_dir / "simulation_metadata.json").read_text())
                check("iid_metadata_contains_ggf_identity", iid_metadata.get("ggf") == identity)
                check("simulation_metadata_contains_ggf_identity", sim_metadata.get("ggf") == identity)
                before = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob("*.npy")}
                expect_rejection("iid_changed_h_rejected_by_metadata", lambda: evaluate.generate_iid(
                    changed_config, split, fake_model, 1., "synthetic-checkpoint", iid_dir, device))
                expect_rejection("simulation_changed_h_rejected_by_metadata", lambda: evaluate.simulate(
                    changed_config, split, fake_model, 1., "synthetic-checkpoint", sim_dir, device))
                (iid_dir / "iid_metadata.json").unlink()
                (sim_dir / "simulation_metadata.json").unlink()
                expect_rejection("iid_changed_h_rejected_by_committed_state", lambda: evaluate.generate_iid(
                    changed_config, split, fake_model, 1., "synthetic-checkpoint", iid_dir, device))
                expect_rejection("simulation_changed_h_rejected_by_committed_state", lambda: evaluate.simulate(
                    changed_config, split, fake_model, 1., "synthetic-checkpoint", sim_dir, device))
                after = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob("*.npy")}
                check("cache_rejections_preserve_sample_arrays", before == after)

            # The deliberately removed sidecars can have been recreated by a
            # rejected resume. Restore the original fixture identity before
            # testing its independent metrics reader.
            evaluate.write_json(iid_metadata, iid_dir / "iid_metadata.json")
            evaluate.write_json(sim_metadata, sim_dir / "simulation_metadata.json")

            expect_rejection("metrics_changed_h_rejected", lambda: evaluate.evaluate_outputs(
                changed_config, split, "ggf", sim_dir, iid_dir))
            state_path = sim_dir / "simulation_state.pt"
            state = torch.load(state_path, weights_only=False)
            state["alive"] = np.zeros(config["evaluation"]["replicas"], dtype=bool)
            torch.save(state, state_path)
            # Evaluation remains meaningful with an empty generated population:
            # coverage stays explicit and all associated metrics are JSON null.
            with patch.object(evaluate, "plot_results") as plot:
                frame = evaluate.evaluate_outputs(config, split, "ggf", sim_dir, iid_dir)
            row = json.loads((sim_dir / "metrics.json").read_text())[0]
            check("zero_survivors_have_explicit_counts", row["simulation_alive_paths"] == 0
                  and row["simulation_total_paths"] == 2 and row["simulation_valid_samples"] == 0)
            simulation_metric_names = ["simulation_" + key for key in
                                       ("jsd", "js_distance", "pmf", "code_jsd", "code_pmf", "code_mjs")]
            check("zero_survivor_simulation_metrics_are_json_null",
                  all(row[key] is None for key in simulation_metric_names))
            metadata = json.loads((sim_dir / "metrics_metadata.json").read_text())
            check("metrics_metadata_contains_ggf_identity", metadata.get("ggf") == identity)
            check("empty_simulation_passed_to_plot", plot.call_count == 1
                  and len(plot.call_args.args[2]) == 0)

            plot_text = []
            original_close = evaluate.plt.close

            def inspect_and_close(fig):
                if hasattr(fig, "axes"):
                    for axis in fig.axes:
                        plot_text.extend(text.get_text() for text in axis.texts)
                        plot_text.append(axis.get_title())
                original_close(fig)

            reference_angles = evaluate.angles(synthetic)
            with patch.object(evaluate.plt, "close", side_effect=inspect_and_close):
                evaluate.plot_results(reference_angles, reference_angles, np.empty((0, 2)),
                                      "AT", sim_dir, config["evaluation"])
            check("empty_population_plots_annotated", any(
                phrase in " ".join(plot_text).lower() for phrase in
                ("no valid", "no surviving", "no samples", "no finite samples", "0 valid", "zero")),
                plot_text=plot_text)
            check("empty_population_plot_files_created", all(
                (sim_dir / f"AT_pmf_{dimension}.png").is_file() for dimension in ("1d", "2d")))

            summary_dir = temporary / "summary"
            summary_dir.mkdir()
            rows = []
            for peptide in PEPTIDES:
                current = dict(row, peptide=peptide, iid_valid_samples=2)
                for name in [key for key in row if key.startswith(("iid_", "simulation_"))
                             and not key.endswith(("samples", "paths"))]:
                    current[name] = .1 if name.startswith("iid_") else (.2 if peptide != "AT" else None)
                if peptide != "AT":
                    current.update(simulation_alive_paths=2, simulation_valid_samples=2)
                rows.append(current)
            settings = dict(config["evaluation"], peptides=PEPTIDES)
            evaluate.write_summary(summary_dir, [pd.DataFrame(rows)], settings, ggf=identity)
            summary = json.loads((summary_dir / "summary.json").read_text())
            check("summary_metadata_contains_ggf_identity", summary.get("ggf") == identity)
            comparison = summary["comparison"]
            check("ggf_has_no_invented_paper_reference", bool(comparison) and all(
                item["method"] == "ggf" and item["paper_mean"] is None
                and item["paper_std"] is None and item["gap"] is None for item in comparison))
            coverage = summary["methods"]["ggf"]["simulation_jsd"]
            check("incomplete_panel_retains_14_of_15_coverage", coverage["evaluated_peptides"] == 14
                  and coverage["expected_peptides"] == 15, metric=coverage)
            check("incomplete_panel_does_not_claim_full_15_mean", coverage["mean"] is None)
            check("full_iid_panel_has_15_peptide_mean", summary["methods"]["ggf"]["iid_jsd"]["evaluated_peptides"] == 15
                  and math.isclose(summary["methods"]["ggf"]["iid_jsd"]["mean"], .1))

        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        report["seconds"] = time.monotonic() - start
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "output": str(output)}, indent=2))
    if report["status"] != "passed":
        print(report["error"]["traceback"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
