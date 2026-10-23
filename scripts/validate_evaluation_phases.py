"""CPU end-to-end checks for evaluation phases, writer locks, and resume.

All generated fixtures and evaluation outputs stay below data/validation.
Run with the experiment's pinned interpreter from the repository root.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))


def interrupted_worker(kind, arguments):
    import evaluate

    original_save = evaluate.atomic_save

    def save_and_stop(value, path):
        original_save(value, path)
        if ((kind == "iid" and str(path).endswith("_iid_state.pt"))
                or (kind == "simulation" and Path(path).name == "simulation_state.pt")):
            print(f"INTENTIONAL_STOP after {path}", flush=True)
            raise SystemExit(82)

    evaluate.atomic_save = save_and_stop
    sys.argv = [str(REPOSITORY / "evaluate.py"), *arguments]
    evaluate.main()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import yaml
    from model import DipeptideModel

    root = REPOSITORY / "data/validation/evaluation-phases" / f"run-{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    report_path = REPOSITORY / "data/results/server-evaluation-checks.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "CPU synthetic end-to-end validation; no production artifacts",
              "evaluate_sha256": digest(REPOSITORY / "evaluate.py"),
              "interpreter": sys.executable, "torch": torch.__version__, "checks": {}, "commands": []}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    torch.set_num_threads(1)

    def check(name, condition, **evidence):
        report["checks"][name] = {"passed": bool(condition), **evidence}
        if not condition:
            raise AssertionError(name)

    def arguments(method, output, phase=None):
        result = ["--config", str(root / "config.yaml"), "--method", method,
                  "--device", "cpu", "--output", str(output)]
        if phase:
            result.append(phase)
        return result

    def run(name, args, expected=0, interrupt=None):
        log = root / f"{name}.log"
        command = ([sys.executable, str(Path(__file__).resolve()), "--interrupt-after", interrupt, "--"]
                   if interrupt else [sys.executable, str(REPOSITORY / "evaluate.py")]) + args
        begin = time.monotonic()
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=REPOSITORY, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=120)
        report["commands"].append({"name": name, "arguments": args, "returncode": result.returncode,
                                   "expected_returncode": expected, "seconds": time.monotonic() - begin,
                                   "log": str(log.relative_to(REPOSITORY))})
        if result.returncode != expected:
            raise AssertionError(f"{name}: returncode {result.returncode}, expected {expected}; {log.read_text()[-6000:]}")
        return log

    def array_hashes(output):
        return {str(path.relative_to(output)): digest(path) for path in output.rglob("*.npy")}

    def verify_completed(output):
        iid = torch.load(output / "diffusion/AT_iid_state.pt", weights_only=False)
        simulations = {method: torch.load(output / method / "simulation_state.pt", weights_only=False)
                       for method in ("diffusion", "two_for_one")}
        check(f"{output.name}_completed_counts", iid["completed"] == 8
              and all(state["completed"] == 100 for state in simulations.values()),
              iid=iid["completed"], simulation={m: s["completed"] for m, s in simulations.items()})
        check(f"{output.name}_shared_iid", not (output / "two_for_one/AT_iid.npy").exists()
              and np.load(output / "diffusion/AT_iid.npy").shape == (8, 30))
        summary = json.loads((output / "summary.json").read_text())
        check(f"{output.name}_both_summary_methods", set(summary["methods"]) == {"diffusion", "two_for_one"})
        rows = {method: json.loads((output / method / "metrics.json").read_text())[0]
                for method in simulations}
        check(f"{output.name}_matching_iid_metrics", all(rows["diffusion"][key] == rows["two_for_one"][key]
              for key in rows["diffusion"] if key.startswith("iid_")))
        check(f"{output.name}_finite_arrays", all(np.isfinite(np.load(path)).all() for path in output.rglob("*.npy")))
        check(f"{output.name}_different_simulation_times",
              simulations["diffusion"]["identity"]["method"] == "diffusion"
              and simulations["two_for_one"]["identity"]["method"] == "two_for_one"
              and digest(output / "diffusion/AT_simulation.npy") != digest(output / "two_for_one/AT_simulation.npy"))

    try:
        fixture = root / "fixture"
        (fixture / "processed/test").mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((REPOSITORY / "configs/dipeptides.yaml").read_text())
        config["data"]["root"] = str(fixture)
        config["model"].update(hidden_dim=8, layers=1, score_hidden_dim=8, score_layers=1,
                                embedding_dim=2, heads=1, head_dim=4)
        config["evaluation"].update(methods=["diffusion", "two_for_one"], peptides=["AT"],
                                     num_iid_samples=8, diffusion_steps=4, simulation_steps=100,
                                     save_every=10, replicas=2, chunk_frames=2, inference_batch=4,
                                     bins=8, plot_bins=8)
        (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        features = [[residue, atom, element, index] for index, residue in enumerate((0, 16))
                    for atom, element in enumerate((1, 0, 0, 0, 2))]
        rng = np.random.default_rng(723)
        coordinates = rng.normal(0, .1, size=(32, 10, 3)).astype("float32")
        coordinates -= coordinates.mean(1, keepdims=True)
        np.save(fixture / "processed/test/AT.npy", coordinates.reshape(32, 30))
        manifest = {"version": 1, "units": "nm", "norm_factor": 1., "splits": {
            "test": {"AT": {"samples": 32, "features": features}}}}
        (fixture / "processed/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        torch.manual_seed(193)
        model = DipeptideModel("diffusion", config)
        checkpoint = fixture / f"runs/diffusion/seed{config['training']['seed']}/final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": "diffusion", "config": config, "norm_factor": 1.,
                    "model": model.state_dict(), "ema": model.state_dict()}, checkpoint)

        concurrent = root / "concurrent"
        # Holding the actual shared lock makes contention observable and deterministic.
        (concurrent / "diffusion").mkdir(parents=True, exist_ok=True)
        lock = (concurrent / "diffusion/.iid.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX)
        processes = []
        try:
            for method in ("diffusion", "two_for_one"):
                log = root / f"concurrent_{method}.log"
                handle = log.open("w")
                args = arguments(method, concurrent)
                process = subprocess.Popen([sys.executable, str(REPOSITORY / "evaluate.py"), *args],
                                           cwd=REPOSITORY, env=env, stdout=handle, stderr=subprocess.STDOUT)
                processes.append((method, process, handle, log, args))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if all("Waiting for another evaluation process:" in p[3].read_text() for p in processes):
                    break
                if any(p[1].poll() is not None for p in processes):
                    raise AssertionError("Concurrent evaluator exited before acquiring shared lock")
                time.sleep(.1)
            check("both_default_processes_waited_for_shared_iid_lock", all(
                "Waiting for another evaluation process:" in p[3].read_text() for p in processes),
                logs=[str(p[3].relative_to(REPOSITORY)) for p in processes])
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        for method, process, handle, log, args in processes:
            result = process.wait(timeout=120)
            handle.close()
            report["commands"].append({"name": f"concurrent_{method}", "arguments": args,
                                       "returncode": result, "expected_returncode": 0,
                                       "log": str(log.relative_to(REPOSITORY))})
            if result != 0:
                raise AssertionError(f"concurrent {method}: {log.read_text()[-6000:]}")
        check("shared_iid_generated_once", sum(p[3].read_text().count("IID AT: 8/8") for p in processes) == 1)
        verify_completed(concurrent)

        phased = root / "phased"
        # Simulation may run before the shared IID dependency becomes available.
        run("simulation_without_iid", arguments("two_for_one", phased, "--simulation-only"))
        check("simulation_only_does_not_generate_iid", not list(phased.rglob("*_iid.npy")))
        check("simulation_only_does_not_generate_metrics", not list(phased.rglob("metrics.json")))
        run("iid_only", arguments("diffusion", phased, "--iid-only"))
        check("iid_only_does_not_generate_simulation", not (phased / "diffusion/AT_simulation.npy").exists())
        check("iid_only_does_not_generate_metrics", not list(phased.rglob("metrics.json")))
        run("diffusion_simulation_only", arguments("diffusion", phased, "--simulation-only"))
        before_metrics = array_hashes(phased)
        run("metrics_only", arguments("all", phased, "--metrics-only"))
        check("metrics_only_preserves_sample_arrays", before_metrics == array_hashes(phased), hashes=before_metrics)
        verify_completed(phased)
        check("phased_equals_concurrent_arrays", array_hashes(phased) == array_hashes(concurrent))

        before_resume = array_hashes(phased)
        run("complete_resume", arguments("all", phased))
        check("completed_resume_preserves_all_arrays", before_resume == array_hashes(phased), hashes=before_resume)

        # A crash after the final state save can precede completion-marker
        # publication. Resuming must repair that marker without advancing RNG.
        marker = phased / "diffusion/simulation_complete.json"
        expected_marker = json.loads(marker.read_text())
        state_digest = digest(phased / "diffusion/simulation_state.pt")
        marker.unlink()
        run("repair_final_simulation_marker", arguments("diffusion", phased, "--simulation-only"))
        check("completed_resume_repairs_missing_marker", marker.exists()
              and json.loads(marker.read_text()) == expected_marker)
        check("marker_repair_preserves_samples_and_state", before_resume == array_hashes(phased)
              and state_digest == digest(phased / "diffusion/simulation_state.pt"))

        invalid_final = root / "invalid-final"
        invalid_directory = invalid_final / "diffusion"
        invalid_directory.mkdir(parents=True)
        for filename in ("simulation_state.pt", "simulation_metadata.json"):
            shutil.copy2(phased / "diffusion" / filename, invalid_directory / filename)
        run("reject_final_state_without_trajectory", arguments("diffusion", invalid_final, "--simulation-only"),
            expected=1)
        check("marker_repair_rejects_missing_trajectory", not (invalid_directory / "simulation_complete.json").exists())

        resumed = root / "interrupted"
        run("interrupted_iid", arguments("diffusion", resumed, "--iid-only"), expected=82, interrupt="iid")
        iid_state = torch.load(resumed / "diffusion/AT_iid_state.pt", weights_only=False)
        check("iid_interrupt_committed_partial_batch", iid_state["completed"] == 4)
        committed_iid = np.load(resumed / "diffusion/AT_iid.npy")[:4].copy()
        run("resume_partial_iid", arguments("diffusion", resumed, "--iid-only"))
        check("iid_resume_preserves_committed_prefix", np.array_equal(
            committed_iid, np.load(resumed / "diffusion/AT_iid.npy")[:4]))
        check("iid_resume_matches_uninterrupted", digest(resumed / "diffusion/AT_iid.npy")
              == digest(phased / "diffusion/AT_iid.npy"))

        run("interrupted_simulation", arguments("diffusion", resumed, "--simulation-only"),
            expected=82, interrupt="simulation")
        sim_state = torch.load(resumed / "diffusion/simulation_state.pt", weights_only=False)
        check("simulation_interrupt_committed_partial_chunk", sim_state["completed"] == 20)
        committed_sim = np.load(resumed / "diffusion/AT_simulation.npy")[:, :2].copy()
        run("resume_partial_simulation", arguments("diffusion", resumed, "--simulation-only"))
        check("simulation_resume_preserves_committed_prefix", np.array_equal(
            committed_sim, np.load(resumed / "diffusion/AT_simulation.npy")[:, :2]))
        check("simulation_resume_matches_uninterrupted", digest(resumed / "diffusion/AT_simulation.npy")
              == digest(phased / "diffusion/AT_simulation.npy"))
        check("evaluation_source_unchanged_during_checks", report["evaluate_sha256"]
              == digest(REPOSITORY / "evaluate.py"))
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["passed_checks"] = sum(c["passed"] for c in report["checks"].values())
        report["total_checks"] = len(report["checks"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "checks": report["total_checks"],
                          "report": str(report_path), "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interrupt-after":
        interrupted_worker(sys.argv[2], sys.argv[4:])
    else:
        main()
