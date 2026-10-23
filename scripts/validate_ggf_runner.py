#!/usr/bin/env python3
"""CPU-only lifecycle tests for the isolated GGF controller.

All processes are short dummy workers, all files are temporary, and GPU inventory
is injected. This never invokes nvidia-smi, training, evaluation, or CUDA.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


MODULE_PATH = Path(__file__).with_name("run_ggf_experiment.py")
spec = importlib.util.spec_from_file_location("ggf_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

DUMMY = r'''
import json, os, pathlib, sys, time
stage, directory, delay = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3])
record = {"stage": stage, "start": time.monotonic(), "pid": os.getpid(),
          "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "threads": [os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")],
          "tmp": os.environ.get("TMPDIR"), "cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR")}
(directory / (stage + "_started.json")).write_text(json.dumps(record))
time.sleep(delay)
if (directory / ("fail_" + stage)).exists():
    sys.stderr.write("intentional dummy failure\n")
    raise SystemExit(7)
if stage == "train":
    (directory / "final.pt").write_bytes(b"dummy checkpoint: not a torch checkpoint")
record["end"] = time.monotonic()
(directory / (stage + "_done.json")).write_text(json.dumps(record))
'''


class DummyArtifacts:
    def __init__(self, root):
        self.root = root

    def has_outputs(self):
        return self.checkpoint().exists() or bool(list(self.root.glob("*_done.json")))

    def checkpoint(self):
        return self.root / "final.pt"

    def audit(self, stage):
        result = runner.read_json(self.root / (stage + "_done.json"))
        if stage == "train":
            result["checkpoint_sha256"] = runner.digest(self.checkpoint())
        return result

    def complete(self):
        return {"status": "complete", "stages": {stage: self.audit(stage) for stage in runner.task_graph()}}


def inventory(count=2, occupied=()):
    return [{"index": str(i), "uuid": f"GPU-dummy-{i}", "total_mib": 24000,
             "used_mib": 100 if i not in occupied else 5000,
             "compute_pids": [] if i not in occupied else [999999], "idle": i not in occupied}
            for i in range(count)]


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ggf-runner-cpu-")
        self.root = Path(self.temporary.name)
        self.work = self.root / "data"
        self.work.mkdir()
        self.source = self.root / "workload.txt"
        self.source.write_text("immutable dummy workload")
        self.dummy = self.root / "dummy.py"
        self.dummy.write_text(DUMMY)
        self.identity = {"schema": 1, "ggf": {"objective": "dummy"},
                         "pins": {str(self.source): runner.digest(self.source)}, "execution": {}}
        self.args = argparse.Namespace(state_dir=self.root / "state", gpus=["0", "1"], retry_failed=False,
                                       poll_seconds=.02, config=self.root / "config.yaml", microbatch=1024,
                                       compile=True, factorized_attention=True, inference_cache=True, cuda_graphs=True)
        self.args.state_dir.mkdir()
        self.artifacts = DummyArtifacts(self.work)
        self.query_count = 0
        self.snapshot = inventory()
        self.delay = {"train": .2, "iid": .35, "simulation": .35, "metrics": .05}
        self.controllers = []

    def tearDown(self):
        # Cleanup only exact handles belonging to this test's dummy workers.
        processes = []
        for controller in self.controllers:
            processes.extend(controller.workers.values())
            if controller.state:
                for record in controller.state["tasks"].values():
                    for identity in runner.handles(record):
                        if runner.is_live(identity):
                            os.kill(identity["pid"], 15)
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.temporary.cleanup()

    def gpu_query(self):
        self.query_count += 1
        return self.snapshot

    def command(self, stage):
        return [sys.executable, str(self.dummy), stage, str(self.work), str(self.delay[stage])]

    def controller(self, prepare=True):
        controller = runner.Controller(self.args, {"data": {"root": str(self.work)}}, self.identity,
                                       self.artifacts, gpu_query=self.gpu_query, command_builder=self.command)
        self.controllers.append(controller)
        if prepare:
            controller.prepare()
        return controller

    def wait_for(self, condition, controller=None, timeout=12):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if controller:
                controller.tick()
            if condition():
                return
            time.sleep(.025)
        self.fail("Timed out waiting for dummy lifecycle transition")

    def finish(self, controller):
        self.wait_for(lambda: controller.state["status"] in {"complete", "failed"}, controller)
        return controller.state["status"]

    def test_dependency_graph_parallel_evaluation_and_cpu_metrics(self):
        controller = self.controller()
        self.assertEqual(self.finish(controller), "complete")
        stages = {name: self.artifacts.audit(name) for name in runner.task_graph()}
        self.assertLess(stages["train"]["end"], min(stages["iid"]["start"], stages["simulation"]["start"]))
        self.assertLess(max(stages["iid"]["start"], stages["simulation"]["start"]),
                        min(stages["iid"]["end"], stages["simulation"]["end"]))
        self.assertNotEqual(stages["iid"]["gpu"], stages["simulation"]["gpu"])
        self.assertGreater(stages["metrics"]["start"], max(stages["iid"]["end"], stages["simulation"]["end"]))
        self.assertEqual(stages["metrics"]["gpu"], "")
        self.assertGreaterEqual(self.query_count, 3)
        for stage, record in controller.state["tasks"].items():
            self.assertEqual(stages[stage]["threads"], ["4"]*3)
            self.assertEqual(Path(stages[stage]["tmp"]), self.root / "runtime-tmp")
            self.assertEqual(Path(stages[stage]["cache"]), self.root / "compiler-cache/inductor")
            self.assertTrue(Path(record["stdout"]).is_file())
            self.assertTrue(Path(record["stderr"]).is_file())
            self.assertEqual(runner.read_json(record["result_file"])["returncode"], 0)

    def test_one_gpu_serial_fallback_and_occupied_gpu_wait(self):
        self.snapshot = inventory(2, occupied=(0, 1))
        controller = self.controller()
        controller.tick()
        self.assertEqual(controller.state["status"], "waiting_for_idle_gpu")
        self.assertFalse(any(self.work.glob("*_started.json")))
        self.snapshot = inventory(2, occupied=(1,))
        self.assertEqual(self.finish(controller), "complete")
        iid, simulation = self.artifacts.audit("iid"), self.artifacts.audit("simulation")
        self.assertEqual(iid["gpu"], simulation["gpu"])
        self.assertLess(iid["end"], simulation["start"])

    def test_restart_adopts_exact_live_worker_and_child(self):
        self.delay["train"] = .8
        first = self.controller()
        first.tick()
        attempt = first.state["tasks"]["train"]["attempt"]
        handle_path = Path(first.state["tasks"]["train"]["handle_file"])
        self.wait_for(lambda: handle_path.exists() and runner.read_json(handle_path).get("child"))
        identities = runner.read_json(handle_path)
        second = self.controller()
        self.assertEqual(second.state["tasks"]["train"]["attempt"], attempt)
        self.assertTrue(second.state["tasks"]["train"]["live_now"])
        self.assertEqual(self.finish(second), "complete")
        self.assertEqual(second.state["tasks"]["train"]["result"]["worker"], identities["worker"])
        self.assertEqual(second.state["tasks"]["train"]["result"]["child"], identities["child"])
        self.assertEqual(len(list((self.args.state_dir / "stages/train").iterdir())), 1)

    def test_orphaned_live_child_is_adopted_without_duplicate(self):
        self.delay["train"] = .8
        first = self.controller()
        first.tick()
        record = first.state["tasks"]["train"]
        self.wait_for(lambda: Path(record["handle_file"]).exists()
                      and runner.read_json(record["handle_file"]).get("child"))
        first.workers["train"].terminate()
        first.workers["train"].wait(timeout=3)
        second = self.controller()
        self.assertEqual(second.state["tasks"]["train"]["status"], "running")
        self.assertEqual(second.state["tasks"]["train"]["attempt"], record["attempt"])
        second.state["tasks"]["train"]["launched_timestamp"] -= 31
        self.assertEqual(self.finish(second), "failed")
        self.assertEqual(len(list((self.args.state_dir / "stages/train").iterdir())), 1)
        self.assertIn("without a durable exit", second.state["tasks"]["train"]["error"])
        self.assertTrue(second.state["tasks"]["train"]["ambiguous_launch"])
        self.args.retry_failed = True
        with self.assertRaisesRegex(RuntimeError, "Cannot retry unregistered process ambiguity"):
            self.controller()

    def test_failed_training_requires_explicit_retry(self):
        failure = self.work / "fail_train"
        failure.touch()
        first = self.controller()
        self.assertEqual(self.finish(first), "failed")
        failed_attempt = first.state["tasks"]["train"]["attempt"]
        self.assertFalse((self.work / "iid_started.json").exists())
        self.assertEqual(first.state["tasks"]["train"]["result"]["returncode"], 7)
        failure.unlink()
        second = self.controller()
        self.assertEqual(self.finish(second), "failed")
        self.assertEqual(second.state["tasks"]["train"]["attempt"], failed_attempt)
        self.args.retry_failed = True
        third = self.controller()
        self.assertEqual(self.finish(third), "complete")
        self.assertNotEqual(third.state["tasks"]["train"]["attempt"], failed_attempt)
        self.assertEqual(len(third.state["tasks"]["train"]["attempt_history"]), 1)

    def test_terminal_identity_and_checkpoint_changes_rejected(self):
        first = self.controller()
        self.assertEqual(self.finish(first), "complete")
        self.identity = {**self.identity, "ggf": {"objective": "different"}}
        with self.assertRaisesRegex(RuntimeError, "Workload identity differs"):
            self.controller()
        self.identity = first.identity
        self.artifacts.checkpoint().write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(RuntimeError, "Previously completed artifacts changed"):
            self.controller()

    def test_live_source_mutation_fails_closed_and_worker_records_failure(self):
        self.delay["train"] = 2
        first = self.controller()
        first.tick()
        record = first.state["tasks"]["train"]
        self.wait_for(lambda: (self.work / "train_started.json").exists())
        self.source.write_text("changed while worker live")
        with self.assertRaisesRegex(RuntimeError, "Pinned workload input changed"):
            first.tick()
        self.wait_for(lambda: Path(record["result_file"]).exists())
        result = runner.read_json(record["result_file"])
        self.assertIn("Pinned workload input changed", result["error"])
        self.assertFalse((self.work / "train_done.json").exists())

    def test_unknown_launch_is_ambiguous_and_cannot_be_retried(self):
        controller = self.controller()
        controller.state["tasks"]["train"].update(status="launching", launched_timestamp=time.time()-60,
                                                    pins=self.identity["pins"], attempt="unknown")
        controller.save()
        controller.reconcile()
        self.assertTrue(controller.state["tasks"]["train"]["ambiguous_launch"])
        controller.save()
        self.args.retry_failed = True
        with self.assertRaisesRegex(RuntimeError, "Cannot retry unregistered process ambiguity"):
            self.controller()

    def test_status_is_read_only_and_reports_actual_liveness(self):
        self.delay["train"] = .8
        controller = self.controller()
        controller.tick()
        handle_path = Path(controller.state["tasks"]["train"]["handle_file"])
        self.wait_for(lambda: handle_path.exists() and runner.read_json(handle_path).get("child"))
        before = {p: (p.stat().st_mtime_ns, runner.digest(p)) for p in self.args.state_dir.rglob("*.json")}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.status(self.args.state_dir)
        report = json.loads(output.getvalue())
        self.assertTrue(report["controller_live_now"])
        self.assertTrue(report["tasks"]["train"]["live_now"])
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, runner.digest(p)) for p in before})
        self.assertEqual(self.query_count, 1)

    def test_durable_post_command_failure_blocks_success_and_primary_failure_skips_it(self):
        for primary_code, post_code in ((0, 9), (7, 0)):
            folder = self.root / f"wrapper-{primary_code}"
            folder.mkdir()
            marker = folder / "post-ran"
            spec = {"attempt": folder.name, "handle_file": str(folder / "handles.json"),
                    "result_file": str(folder / "result.json"), "stdout": str(folder / "stdout.log"),
                    "stderr": str(folder / "stderr.log"), "cwd": str(self.root), "env": {},
                    "pins": self.identity["pins"],
                    "command": [sys.executable, "-c", f"raise SystemExit({primary_code})"],
                    "post_command": [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch(); raise SystemExit({post_code})"]}
            runner.atomic_json(folder / "spec.json", spec)
            result = subprocess.run([sys.executable, str(MODULE_PATH), "--worker", str(folder / "spec.json")],
                                    capture_output=True, timeout=5)
            expected = post_code if primary_code == 0 else primary_code
            self.assertEqual(result.returncode, expected)
            record = runner.read_json(folder / "result.json")
            self.assertEqual(record["returncode"], expected)
            self.assertEqual(len(record["command_results"]), 2 if primary_code == 0 else 1)
            self.assertEqual(marker.exists(), primary_code == 0)

    def test_gpu_memory_contexts_and_uuid_selection(self):
        # Exercise the real parser against mocked subprocess output, no GPU access.
        from unittest.mock import patch
        answers = [subprocess.CompletedProcess([], 0, stdout="0, GPU-zero, 24000, 80\n1, GPU-one, 24000, 10\n2, GPU-two, 24000, 3000\n"),
                   subprocess.CompletedProcess([], 0, stdout="GPU-one, 12345\n")]
        with patch.object(runner.subprocess, "run", side_effect=answers) as mocked:
            snapshot = runner.query_gpus()
        self.assertEqual([gpu["idle"] for gpu in snapshot], [True, False, False])
        self.assertEqual(runner.choose_gpu(["1", "GPU-two", "0"], snapshot, set())["uuid"], "GPU-zero")
        self.assertIsNone(runner.choose_gpu(["0"], snapshot, {"GPU-zero"}))
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(all("--query-" in call.args[0][1] for call in mocked.call_args_list))

    def test_dedicated_repository_data_namespace_and_alias_guards(self):
        from unittest.mock import patch
        repo = self.root / "repository"
        data = repo / "data"
        data.mkdir(parents=True)
        dedicated = data / "ggf"
        dedicated.mkdir()
        baseline = dedicated / "baselines/non-fokker-planck"
        baseline.mkdir(parents=True)
        with patch.object(runner, "REPO", repo), patch.object(runner, "PROTECTED", repo), \
                patch.object(runner, "BASELINE", baseline):
            self.assertEqual(runner.validate_destination(dedicated), dedicated)
            self.assertEqual(runner.validate_destination(dedicated / "experiment"), dedicated / "experiment")
            for forbidden in (repo, data, data / "processed", data / "runs", data / "results", data / "experiment",
                              data / "downloads", data / "raw", data / "ggf-other",
                              dedicated / "../processed", baseline, baseline / "summary.json"):
                with self.assertRaises(ValueError, msg=str(forbidden)):
                    runner.validate_destination(forbidden)
            (dedicated / "escape").symlink_to(data / "processed")
            with self.assertRaises(ValueError):
                runner.validate_destination(dedicated / "escape/results")
            (dedicated / "escape").unlink()
            baseline.rmdir()
            baseline.parent.rmdir()
            dedicated.rmdir()
            dedicated.symlink_to(data / "processed")
            with self.assertRaises(ValueError):
                runner.validate_destination(dedicated / "experiment")
            dedicated.unlink()
            for name in ("runs", "results", "experiment", "processed", "downloads", "raw"):
                old_output = self.root / f"old-{name}"
                old_output.mkdir()
                (data / name).symlink_to(old_output)
                with self.assertRaises(ValueError):
                    runner.validate_destination(data / name / "nested")
                with self.assertRaises(ValueError):
                    runner.validate_destination(old_output / "nested")
                dedicated.symlink_to(old_output)
                with self.assertRaises(ValueError):
                    runner.validate_destination(dedicated / "experiment")
                dedicated.unlink()
            storage = self.root / "external-storage"
            storage.mkdir()
            dedicated.symlink_to(storage)
            self.assertEqual(runner.validate_destination(dedicated / "experiment"), storage / "experiment")
            with self.assertRaisesRegex(ValueError, "pinned baseline"):
                runner.validate_destination(dedicated / "baselines/non-fokker-planck/summary.json")

    def test_baseline_uses_repository_handoff_path_and_preserves_exact_pins(self):
        self.assertEqual(runner.BASELINE, runner.REPO / "data/ggf/baselines/non-fokker-planck")
        self.assertEqual(runner.BASELINE_PINS, {
            str(runner.BASELINE / "report_receipt.json"): "577fc4b1b5e389f36be149c74684efb4c4a646c4917b4a74d5aaa00214fab1bc",
            str(runner.BASELINE / "summary.json"): "22f17b072600c3c429008f2daf1cf33030d09db9090f0c4c1adf6c697c53acc6"})

    def test_protocol_protects_derived_outputs_and_old_physical_root(self):
        from unittest.mock import patch
        repo = self.root / "repository"
        data = repo / "data"
        dedicated = data / "ggf"
        baseline = dedicated / "baselines/non-fokker-planck"
        baseline.mkdir(parents=True)
        old_root = self.root / "old-data"
        for name in ("runs", "results"):
            (old_root / name).mkdir(parents=True)
            (data / name).symlink_to(old_root / name)
        config = {"data": {"root": str(dedicated)},
                  "training": {"methods": ["ggf"], "epochs": 120, "batch_size": 1024, "seed": 1},
                  "evaluation": {"methods": ["ggf"], "peptides": sorted(runner.PEPTIDES),
                                 "num_iid_samples": 49000, "diffusion_steps": 1000,
                                 "simulation_steps": 6000000, "replicas": 10, "save_every": 50,
                                 "dt": .0005, "temperature": 310., "gamma": 1., "eval_time": .00001}}
        with patch.object(runner, "REPO", repo), patch.object(runner, "PROTECTED", repo), \
                patch.object(runner, "BASELINE", baseline):
            runner.validate_protocol(config)
            config["data"]["root"] = str(old_root)
            self.assertEqual(runner.validate_destination(old_root), old_root)
            with self.assertRaisesRegex(ValueError, "protected experiment data"):
                runner.validate_protocol(config)
            config["data"]["root"] = str(dedicated)
            (dedicated / "runs").symlink_to(old_root / "runs")
            with self.assertRaisesRegex(ValueError, "protected experiment data"):
                runner.validate_protocol(config)
            (dedicated / "runs").unlink()
            (dedicated / "results").symlink_to(baseline)
            with self.assertRaisesRegex(ValueError, "pinned baseline"):
                runner.validate_protocol(config)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Pinned torch environment required for synthetic artifact fixture")
    def test_synthetic_artifact_audit_rejects_coercion_rng_and_final_frame_mismatch(self):
        import numpy as np
        import torch
        settings = {"peptides": ["AT"], "seed": 0, "simulation_steps": 4, "save_every": 2, "replicas": 2,
                    "eval_time": .00001, "two_for_one_time": .02, "dt": .0005, "gamma": 1.,
                    "temperature": 310., "num_iid_samples": 6, "diffusion_steps": 3}
        config = {"data": {"root": str(self.work)}, "training": {"seed": 1}, "evaluation": settings}
        artifacts = runner.Artifacts(config, {"objective": "synthetic"})
        artifacts.directory.mkdir(parents=True)
        artifacts.checkpoint().parent.mkdir(parents=True)
        artifacts.checkpoint().write_bytes(b"synthetic checkpoint digest only; never torch loaded")
        (self.work / "processed").mkdir()
        runner.atomic_json(self.work / "processed/manifest.json", {"norm_factor": 1.0})
        identity = {"checkpoint": runner.digest(artifacts.checkpoint()), "ggf": artifacts.ggf, "norm_factor": 1.,
                    "method": "ggf", **{k: settings[k] for k in runner.PROTOCOL_KEYS}}
        state = {"identity": identity, "completed": 4, "alive": np.array([True, False]),
                 "x": torch.zeros(2, 30, dtype=torch.float32), "v": torch.zeros(2, 30, dtype=torch.float32),
                 "rng": torch.zeros(16, dtype=torch.uint8)}
        state["x"][1] = float("nan")
        state["v"][1] = float("nan")
        path = artifacts.directory / "simulation_state.pt"
        torch.save(state, path)
        runner.atomic_json(artifacts.directory / "simulation_metadata.json", identity)
        marker = {"complete": True, "steps_per_path": 4, "replicas": 2,
                  "aggregate_ns_per_peptide": 4*.0005*2/1000, "alive_paths": [True, False]}
        marker_path = artifacts.directory / "simulation_complete.json"
        runner.atomic_json(marker_path, marker)
        trajectories = np.zeros((2, 2, 30), dtype=np.float32)
        trajectories[1] = np.nan
        np.save(artifacts.directory / "AT_simulation.npy", trajectories)
        self.assertEqual(artifacts.simulation(full=True)["dead_paths"], 1)
        for bad in ({**state, "alive": np.array([1, 0], dtype=np.uint8)},
                    {**state, "x": state["x"].double()}, {**state, "rng": torch.zeros(16)}):
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                artifacts.simulation(full=True)
        torch.save(state, path)
        for key, value in (("replicas", 99), ("aggregate_ns_per_peptide", 123.0)):
            runner.atomic_json(marker_path, {**marker, key: value})
            with self.assertRaisesRegex(ValueError, "completion marker"):
                artifacts.simulation(full=True)
        runner.atomic_json(marker_path, marker)
        trajectories[0, -1, 0] = -0.0
        np.save(artifacts.directory / "AT_simulation.npy", trajectories)
        with self.assertRaisesRegex(ValueError, "differs bytewise"):
            artifacts.simulation(full=True)
        iid_identity = {"checkpoint": runner.digest(artifacts.checkpoint()), "ggf": artifacts.ggf, "norm_factor": 1.,
                        "samples": 6, "steps": 3, "seed": 0}
        runner.atomic_json(artifacts.directory / "iid_metadata.json", iid_identity)
        iid_state = {"identity": iid_identity, "completed": 6, "rng": torch.zeros(16, dtype=torch.uint8)}
        torch.save(iid_state, artifacts.directory / "AT_iid_state.pt")
        np.save(artifacts.directory / "AT_iid.npy", np.zeros((6, 30), dtype=np.float32))
        self.assertEqual(artifacts.iid()["samples_per_peptide"], 6)
        torch.save({**iid_state, "rng": torch.zeros(16, dtype=torch.float32)}, artifacts.directory / "AT_iid_state.pt")
        with self.assertRaisesRegex(ValueError, "RNG must be"):
            artifacts.iid()


if __name__ == "__main__":
    unittest.main(verbosity=2)
