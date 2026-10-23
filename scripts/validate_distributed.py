#!/usr/bin/env python3
"""Exercise production batch partitioning and resume on two visible CUDA GPUs."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from model import DipeptideModel


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def tensor_errors(left, right):
    values = [(left[key].float() - right[key].float()).abs().max().item() for key in left]
    return max(values, default=0.)


def probe(config_path, output):
    config = train.load_config(config_path)
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    report = {"world_size": dist.get_world_size(), "global_batch": 1024,
              "local_batch": 512, "microbatch": 128, "scenarios": []}
    for method, stage in (("diffusion", 0), ("fokker_planck", 0),
                          ("mixture", 0), ("mixture", 1), ("mixture", 2)):
        torch.manual_seed(23)
        actual = DipeptideModel(method, config).to(device)
        if method == "mixture":
            actual.freeze_for_stage(stage)
        reference = copy.deepcopy(actual)
        x = torch.randn(1024, 30, device=device)
        features = torch.zeros(1024, 10, 4, dtype=torch.long, device=device)
        lo, hi = ((.6, 1.), (.099, .6), (1e-5, .099))[stage] if method == "mixture" else (1e-5, 1.)
        t = torch.rand(1024, 1, device=device) * (hi - lo) + lo
        noise = torch.randn_like(x)
        vectors = config["training"]["sigma"] * torch.randn(2, 1024, 30, device=device)
        initial = {"model": digest(torch.cat([p.flatten() for p in actual.parameters()])),
                   "x": digest(x), "t": digest(t), "noise": digest(noise), "vectors": digest(vectors),
                   "cuda_rng": digest(torch.cuda.get_rng_state(device))}
        initial_by_rank = [None, None]
        dist.all_gather_object(initial_by_rank, initial)
        assert initial_by_rank[0] == initial_by_rank[1], "Ranks drew different complete batches/RNG"
        full_loss = torch.zeros(3, device=device)
        local_loss = torch.zeros(3, device=device)
        for model, begin, end, acc in ((reference, 0, 1024, full_loss),
                                       (actual, rank * 512, (rank + 1) * 512, local_loss)):
            for start in range(begin, end, 128):
                stop = start + 128
                terms = train.losses(model, x[start:stop], features[start:stop], t[start:stop],
                                     config, stage, noise[start:stop], vectors[:, start:stop]) / 8
                terms.sum().backward()
                acc += terms.detach()
        train._sum_gradients(actual)
        dist.all_reduce(local_loss)
        torch.testing.assert_close(local_loss, full_loss, rtol=2e-5, atol=2e-6)
        gradient_max = 0.
        clipped = 0
        for left, right in zip(actual.parameters(), reference.parameters()):
            if not left.requires_grad:
                assert left.grad is None and right.grad is None
                continue
            expected = right.grad if right.grad is not None else torch.zeros_like(right)
            torch.testing.assert_close(left.grad, expected, rtol=3e-4, atol=2e-5)
            gradient_max = max(gradient_max, (left.grad - expected).abs().max().item())
            clipped += int((expected.abs() > config["training"]["clip"]).sum().item())
        emas = [{key: value.detach().clone() for key, value in model.state_dict().items()}
                for model in (actual, reference)]
        for model, ema in zip((actual, reference), emas):
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=config["training"]["learning_rate"],
                                         weight_decay=config["training"]["weight_decay"], foreach=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4,
                                         eta_min=config["training"]["min_learning_rate"])
            train._optimizer_step(model, optimizer, scheduler, ema, config["training"])
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=2e-4, atol=3e-6)
            torch.testing.assert_close(emas[0][key], emas[1][key], rtol=2e-4, atol=3e-6)
        scenario = {"method": method, "stage": stage, "initial_draws_identical_across_ranks": True,
                    "initial_draw_hashes": initial, "gradient_max_abs": gradient_max,
                    "parameter_max_abs": tensor_errors(actual.state_dict(), reference.state_dict()),
                    "ema_max_abs": tensor_errors(emas[0], emas[1]), "clipped_gradient_elements": clipped,
                    "loss_max_abs": (local_loss - full_loss).abs().max().item()}
        # Two complete validation batches exercise nonempty contributions from
        # both ranks and the stochastic inputs skipped by the other rank.
        validation = train.Peptides({"AA": {"x": np.tile(x.detach().cpu().numpy(), (2, 1)),
                                            "features": features[0].cpu().numpy()}})
        np_rng = np.random.default_rng(45)
        before = train._rng_state(np_rng, device)
        distributed_validation = train.validate(actual, validation, config, stage, 1., device, np_rng, 128)
        after = train._rng_state(np_rng, device)
        validation_rngs = [None, None]
        dist.all_gather_object(validation_rngs, digest(after["cuda"]))
        assert validation_rngs[0] == validation_rngs[1], "Validation desynchronized rank RNGs"
        train._restore_rng(before, np_rng, device)
        with patch.object(train.dist, "is_initialized", return_value=False):
            single_validation = train.validate(actual, validation, config, stage, 1., device, np_rng, 128)
        single_after = train._rng_state(np_rng, device)
        np.testing.assert_allclose(distributed_validation, single_validation, rtol=2e-5, atol=3e-6)
        assert torch.equal(after["cuda"], single_after["cuda"]), "Single/distributed validation RNG mismatch"
        assert after["numpy"] == single_after["numpy"]
        scenario["validation"] = {"global_batches": 2, "batches_owned_per_rank": 1,
                                  "loss_max_abs": float(np.max(np.abs(distributed_validation - single_validation))),
                                  "rng_bitwise_equal_single_vs_distributed": True,
                                  "rng_bitwise_equal_across_ranks": True}
        report["scenarios"].append(scenario)
    assert any(row["clipped_gradient_elements"] for row in report["scenarios"]), "Clipping was not exercised"
    if rank == 0:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


def prepare(base):
    config = train.load_config(ROOT / "configs/dipeptides.yaml")
    config["model"].update(hidden_dim=8, score_hidden_dim=8, layers=1, score_layers=1,
                           embedding_dim=2, heads=2, head_dim=4)
    config["training"].update(epochs=2, mixture_epochs=[1, 1, 1], clip=.001)
    rng = np.random.default_rng(34)
    arrays = {split: (.1 * rng.normal(size=(count, 10, 3))).astype("float32")
              for split, count in (("train", 2048), ("val", 2048), ("test", 64))}
    for x in arrays.values():
        x -= x.mean(axis=1, keepdims=True)
    norm = float(1 / np.asarray(arrays["train"], dtype=np.float64).std())
    features = np.asarray([[0, atom, element, residue] for residue in range(2)
                           for atom, element in enumerate((1, 0, 0, 0, 2))], dtype=np.int64)
    configs = {}
    for name in ("single", "distributed", "resumed"):
        data_root = base / name
        manifest = {"version": 1, "units": "nm", "norm_factor": norm, "splits": {}}
        for split, x in arrays.items():
            path = data_root / "processed" / split
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "AA.npy", x.reshape(len(x), 30))
            np.save(path / "AA.features.npy", features)
            manifest["splits"][split] = {"AA": {"samples": len(x), "features": features.tolist()}}
        (data_root / "processed/manifest.json").write_text(json.dumps(manifest))
        current = copy.deepcopy(config)
        current["data"]["root"] = str(data_root)
        config_path = base / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(current))
        configs[name] = config_path
    return configs


def compare_checkpoints(left_path, right_path, *, exact=False):
    left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in (left_path, right_path)]
    for key in ("epoch", "global_step", "stage", "step_in_epoch", "completed"):
        assert left[key] == right[key], (key, left[key], right[key])
    maxima = {}
    for name in ("model", "ema"):
        maxima[name] = tensor_errors(left[name], right[name])
        for key in left[name]:
            torch.testing.assert_close(left[name][key], right[name][key], rtol=0 if exact else 1e-3,
                                       atol=0 if exact else 2e-5)
    opt_error = 0.
    optimizer_exact = True
    for key, value in left["optimizer"]["state"].items():
        for field, tensor in value.items():
            other = right["optimizer"]["state"][key][field]
            # CUDA embedding backward can change the last few bits of moments
            # even between repeated uninterrupted runs. Keep model/EMA/RNG
            # exact across resume while checking moments tightly and recording
            # whether their stronger bitwise comparison also holds.
            torch.testing.assert_close(tensor, other, rtol=1e-5 if exact else 3e-3,
                                       atol=1e-9 if exact else 2e-6)
            opt_error = max(opt_error, (tensor - other).abs().max().item())
            optimizer_exact &= torch.equal(tensor, other)
    assert left["scheduler"] == right["scheduler"]
    for key in ("torch", "cuda"):
        assert torch.equal(left["rng"][key], right["rng"][key]), f"{key} RNG mismatch"
    assert left["rng"]["numpy"] == right["rng"]["numpy"]
    assert left["rng"]["python"] == right["rng"]["python"]
    assert left["completed"] and left["step_in_epoch"] == 0
    return {"parameter_max_abs": maxima["model"], "ema_max_abs": maxima["ema"],
            "optimizer_tensor_max_abs": opt_error, "epoch": left["epoch"],
            "global_step": left["global_step"], "rng_bitwise_equal": True,
            "scheduler_equal": True, "completed": True, "model_and_ema_exact_required": exact,
            "optimizer_bitwise_equal": optimizer_exact}


def orchestrate(base, output):
    base.mkdir(parents=True, exist_ok=True)
    configs = prepare(base)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="2,3", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    python = sys.executable
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "train_py_sha256": hashlib.sha256((ROOT / "train.py").read_bytes()).hexdigest(),
              "gpus": [2, 3], "synthetic_data_root": str(base), "effective_batch": 1024,
              "reference_microbatch": 128, "training": {}, "logs": []}

    def run(args, log_name, distributed=False):
        command = [python]
        if distributed:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += args
        log = base / log_name
        report["logs"].append(str(log))
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command exited {result.returncode}; see {log}:\n{log.read_text()[-6000:]}")

    try:
        probe_output = base / "gradient-probe.json"
        run([str(Path(__file__).resolve()), "--worker", "--config", str(configs["single"]),
             "--output", str(probe_output)], "gradient-probe.log", distributed=True)
        report["fixed_noise_probe"] = json.loads(probe_output.read_text())
        for method in ("diffusion", "fokker_planck", "mixture"):
            common = ["train.py", "--method", method, "--microbatch", "128"]
            for scenario in ("single", "distributed", "resumed"):
                args = common + ["--config", str(configs[scenario])]
                if scenario != "single":
                    args += ["--distributed"]
                if scenario == "resumed":
                    args += ["--max-steps", "3"]
                run(args, f"{method}-{scenario}.log", distributed=scenario != "single")
            latest = base / "resumed/runs" / method / "seed1/latest.pt"
            interrupted = torch.load(latest, map_location="cpu", weights_only=False)
            assert interrupted["epoch"] == 1 and interrupted["global_step"] == 3 and interrupted["step_in_epoch"] == 1
            run(common + ["--config", str(configs["resumed"]), "--distributed", "--resume", str(latest)],
                f"{method}-resume-complete.log", distributed=True)
            final = lambda scenario: base / scenario / "runs" / method / "seed1/final.pt"
            report["training"][method] = {
                "single_vs_distributed": compare_checkpoints(final("single"), final("distributed")),
                "uninterrupted_vs_resumed_distributed": compare_checkpoints(final("distributed"), final("resumed"), exact=True),
                "interrupted_after_epoch_boundary": {"epoch": interrupted["epoch"],
                                                     "step_in_epoch": interrupted["step_in_epoch"],
                                                     "global_step": interrupted["global_step"]}}
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--base", type=Path, default=ROOT / "data/validation/distributed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/distributed-validation.json")
    args = parser.parse_args()
    if args.worker:
        probe(args.config, args.output)
        return 0
    return orchestrate(args.base, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
