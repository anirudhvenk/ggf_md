#!/usr/bin/env python3
"""Run and resume the full dipeptide experiment across four GPUs.

The controller and its stage workers survive independently. Restarting the
controller adopts workers by PID and Linux process start time. Nonzero stage
exits stop that dependency chain; --retry-failed explicitly retries failures.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
if not (REPO / "train.py").is_file():
    # Keep the isolated candidate runnable from its validation directory too.
    REPO = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "train.py").is_file() and (parent / "configs/dipeptides.yaml").is_file())
TRAIN_METHODS = ("diffusion", "mixture", "fokker_planck")
METHODS = ("diffusion", "two_for_one", "mixture", "fokker_planck")
PEPTIDES = {"KS", "AT", "LW", "KQ", "NY", "IM", "TD", "HT", "NF", "RL", "ET", "AC", "RV", "HP", "AP"}
PROTOCOL_KEYS = ("peptides", "seed", "simulation_steps", "save_every", "replicas", "eval_time",
                 "two_for_one_time", "dt", "gamma", "temperature")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    """PID reuse and zombies must never count as a live stage."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_live(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def handles(record):
    result = [record.get("worker"), record.get("child")]
    if record.get("handle_file") and Path(record["handle_file"]).exists():
        handle = read_json(record["handle_file"])
        result.extend([handle.get("worker"), handle.get("child")])
    return [item for item in result if item]


def run_worker(spec_path):
    """A durable wrapper preserves an exit result for an adopting controller."""
    spec = read_json(spec_path)
    handle = {"worker": process_identity(os.getpid()), "started_at": now()}
    atomic_json(spec["handle_file"], handle)
    code, error = 127, None
    try:
        env = os.environ.copy()
        env.update(spec["env"])
        with Path(spec["log"]).open("a", buffering=1) as output:
            output.write(f"\n[{now()}] Starting {spec['command']}\n")
            child = subprocess.Popen(spec["command"], cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
            handle["child"] = process_identity(child.pid)
            atomic_json(spec["handle_file"], handle)
            code = child.wait()
            output.write(f"[{now()}] Stage exit code: {code}\n")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    atomic_json(spec["result_file"], {"returncode": code, "error": error, "finished_at": now(), **handle})
    return code if 0 <= code < 256 else 1


class Artifacts:
    def __init__(self, config):
        import numpy as np
        import torch
        self.np, self.torch, self.config = np, torch, config
        self.root = (REPO / config["data"]["root"]).resolve()
        self.results = self.root / "results"
        self.settings = config["evaluation"]
        self.hashes = {}

    def normalization(self):
        return read_json(self.root / "processed" / "manifest.json")["norm_factor"]

    def load(self, path):
        return self.torch.load(path, map_location="cpu", weights_only=False)

    def checkpoint_path(self, method, name="final.pt"):
        return self.root / "runs" / method / f"seed{self.config['training']['seed']}" / name

    def digest(self, method):
        path = self.checkpoint_path(method)
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if key not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            self.hashes[key] = digest.hexdigest()
        return self.hashes[key]

    def training(self, method):
        state = self.load(self.checkpoint_path(method))
        cfg = self.config["training"]
        expected = sum(cfg["mixture_epochs"]) if method == "mixture" else cfg["epochs"]
        assert state["method"] == method, "checkpoint method mismatch"
        assert state.get("completed") is True, "checkpoint is not marked completed"
        assert state["epoch"] == expected and state.get("step_in_epoch", 0) == 0, "incomplete training schedule"
        assert all(state["config"][key] == self.config[key] for key in ("model", "training")), "checkpoint config mismatch"
        manifest = read_json(self.root / "processed" / "manifest.json")
        frames = sum(item["samples"] for item in manifest["splits"]["train"].values())
        assert state["global_step"] == expected * (frames // cfg["batch_size"]), "optimizer step count mismatch"
        assert math.isclose(state["norm_factor"], manifest["norm_factor"], rel_tol=1e-8), "normalization mismatch"
        assert state.get("ema") and all(self.torch.isfinite(t).all().item() for t in state["ema"].values()), "nonfinite/missing EMA weights"
        return {"epochs": expected, "optimizer_steps": state["global_step"], "checkpoint_sha256": self.digest(method)}

    def iid(self, method, full=False):
        s, np = self.settings, self.np
        folder = self.results / method
        digest = self.digest(method)
        for peptide in s["peptides"]:
            state = self.load(folder / f"{peptide}_iid_state.pt")
            assert state["completed"] == s["num_iid_samples"], f"incomplete IID {peptide}"
            expected = {"checkpoint": digest, "norm_factor": self.normalization(), "samples": s["num_iid_samples"],
                        "steps": s["diffusion_steps"], "seed": s["seed"]}
            assert all(state["identity"].get(k) == v for k, v in expected.items()), f"IID identity mismatch: {peptide}"
            assert read_json(folder / "iid_metadata.json") == state["identity"], f"IID metadata mismatch: {peptide}"
            array = np.load(folder / f"{peptide}_iid.npy", mmap_mode="r")
            assert array.shape == (s["num_iid_samples"], 30) and array.dtype == np.float32, f"IID shape/dtype: {peptide}"
            if full:
                assert np.isfinite(array).all(), f"nonfinite IID samples: {peptide}"
        return {"peptides": len(s["peptides"]), "samples_per_peptide": s["num_iid_samples"]}

    def simulation(self, method, full=False):
        s, np = self.settings, self.np
        folder = self.results / method
        state = self.load(folder / "simulation_state.pt")
        trained_method = "diffusion" if method == "two_for_one" else method
        expected = {"checkpoint": self.digest(trained_method), "norm_factor": self.normalization(),
                    "method": method, **{k: s[k] for k in PROTOCOL_KEYS}}
        assert all(state["identity"].get(k) == v for k, v in expected.items()), "simulation identity mismatch"
        assert read_json(folder / "simulation_metadata.json") == state["identity"], "simulation metadata mismatch"
        assert state["completed"] == s["simulation_steps"], "incomplete simulation"
        complete = read_json(folder / "simulation_complete.json")
        assert complete["complete"] and complete["steps_per_path"] == s["simulation_steps"], "missing completion marker"
        alive = np.asarray(state["alive"], dtype=bool)
        assert alive.shape == (len(s["peptides"]) * s["replicas"],), "alive-path count mismatch"
        assert state["x"].shape == state["v"].shape == (len(alive), 30), "simulation state shape mismatch"
        for i, peptide in enumerate(s["peptides"]):
            array = np.load(folder / f"{peptide}_simulation.npy", mmap_mode="r")
            expected_shape = (s["replicas"], s["simulation_steps"] // s["save_every"], 30)
            assert array.shape == expected_shape and array.dtype == np.float32, f"trajectory shape/dtype: {peptide}"
            if full:
                for replica in np.flatnonzero(alive[i*s["replicas"]:(i+1)*s["replicas"]]):
                    assert np.isfinite(array[replica]).all(), f"nonfinite retained trajectory: {peptide}/{replica}"
        return {"steps_per_path": s["simulation_steps"], "replicas_per_peptide": s["replicas"],
                "alive_paths": int(alive.sum()), "total_paths": len(alive),
                "aggregate_ns_per_peptide": s["simulation_steps"] * s["dt"] * s["replicas"] / 1000}

    def metrics(self):
        s, np = self.settings, self.np
        with (self.results / "per_peptide.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(METHODS) * len(s["peptides"]), "combined metrics must contain all 60 method/peptide rows"
        assert {(r["method"], r["peptide"]) for r in rows} == {(m,p) for m in METHODS for p in s["peptides"]}, "metric method/peptide pairs mismatch"
        primary = ("iid_jsd", "simulation_jsd", "iid_pmf", "simulation_pmf")
        for row in rows:
            assert all(math.isfinite(float(row[key])) for key in primary), "missing/nonfinite metrics"
            assert int(row["iid_valid_samples"]) == int(row["simulation_valid_samples"]) == s["num_iid_samples"], "metric sample count mismatch"
        summary = read_json(self.results / "summary.json")
        assert summary["settings"] == s and set(summary["methods"]) == set(METHODS), "summary protocol/method mismatch"
        for method in METHODS:
            for metric in primary:
                item = summary["methods"][method][metric]
                assert item["evaluated_peptides"] == item["expected_peptides"] == len(s["peptides"]), "incomplete peptide summary"
            for peptide in s["peptides"]:
                folder = self.results / method
                for suffix in ("pmf_1d.png", "pmf_2d.png"):
                    assert (folder / f"{peptide}_{suffix}").stat().st_size > 0, "missing PMF plot"
                with np.load(folder / f"{peptide}_angles.npz") as angles:
                    for source in ("iid", "simulation"):
                        assert angles[source].shape == (s["num_iid_samples"], 2) and np.isfinite(angles[source]).all(), "invalid evaluated angles"
        with (self.results / "comparison.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == len(METHODS) * len(primary) * 2, "incomplete published-metric comparison"
        return {"method_peptide_rows": len(rows), "methods": list(METHODS), "plots": len(rows) * 2}

    def task(self, name):
        phase, method = name.split(":")
        if phase == "train":
            return self.training(method)
        if phase == "iid":
            return self.iid(method)
        if phase == "simulation":
            return self.simulation(method)
        return self.metrics()

    def complete(self):
        return {"verified_at": now(), "status": "complete", "protocol": self.settings,
                "training": {m: self.training(m) for m in TRAIN_METHODS},
                "independent_sampling": {m: self.iid(m, full=True) for m in TRAIN_METHODS},
                "simulation": {m: self.simulation(m, full=True) for m in METHODS}, "metrics": self.metrics()}


def task_graph():
    tasks = {}
    for method in TRAIN_METHODS:
        tasks[f"train:{method}"] = {"dependencies": []}
    # The second simulation can start as soon as Diffusion weights exist.
    tasks["simulation:two_for_one"] = {"dependencies": ["train:diffusion"]}
    for method in TRAIN_METHODS:
        tasks[f"iid:{method}"] = {"dependencies": [f"train:{method}"]}
        tasks[f"simulation:{method}"] = {"dependencies": [f"train:{method}"]}
    tasks["metrics:all"] = {"dependencies": [f"simulation:{m}" for m in METHODS] + [f"iid:{m}" for m in TRAIN_METHODS]}
    return tasks


def parse_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_factorized_training(value):
    methods = parse_list(value) if isinstance(value, str) else list(value)
    if len(set(methods)) != len(methods) or set(methods) - set(TRAIN_METHODS):
        raise ValueError("--factorized-training requires unique supported training method names")
    return sorted(methods)


def normalize_identity(identity):
    normalized = {"fp_factorized_attention": False, "factorized_training": [],
                  "factorized_inference": False, **identity}
    normalized["factorized_training"] = parse_factorized_training(normalized["factorized_training"])
    return normalized


def factorized_training_methods(identity):
    methods = set(identity["factorized_training"])
    if identity["fp_factorized_attention"]:
        methods.add("fokker_planck")
    return methods


def stage_started(record):
    return bool(record.get("status") in ("launching", "running", "succeeded", "failed")
                or any(record.get(key) for key in ("started_at", "launched_timestamp", "worker", "child",
                                                   "handle_file", "result_file", "command")))


def sampling_artifacts_started(artifacts, methods, include_combined=False):
    patterns = ("iid_metadata.json", "*_iid_state.pt", "*_iid.npy", "simulation_metadata.json",
                "simulation_state.pt", "simulation_complete.json", "*_simulation.npy",
                "metrics.json", "metrics.csv", "metrics_metadata.json", "*_angles.npz", "*_pmf_*.png")
    return (any(any((artifacts.results / method).glob(pattern)) for method in methods for pattern in patterns)
            or include_combined and any((artifacts.results / name).exists()
                                        for name in ("summary.json", "comparison.csv", "per_peptide.csv")))



def plan_evaluation_dependency_migration(state, graph, artifacts):
    """Validate the one pre-evaluation graph change without mutating saved state."""
    tasks = state.get("tasks", {})
    if not tasks:
        return None
    if set(tasks) != set(graph):
        raise RuntimeError("Persisted task names differ from the full experiment graph")
    changes = {}
    allowed = {f"simulation:{method}": [f"iid:{method}"] for method in TRAIN_METHODS}
    for name, definition in graph.items():
        previous = tasks[name].get("dependencies")
        current = definition["dependencies"]
        if previous == current:
            continue
        if name not in allowed or previous != allowed[name]:
            raise RuntimeError(f"Unexpected persisted dependency drift for {name}")
        changes[name] = {"previous": previous.copy(), "current": current.copy()}
    if not changes:
        return None
    if set(changes) != set(allowed):
        raise RuntimeError("Only the complete three-method evaluation dependency migration is allowed")
    for name, record in tasks.items():
        if name.split(":", 1)[0] not in ("iid", "simulation", "metrics"):
            continue
        if (record.get("status") != "pending" or stage_started(record)
                or any(is_live(handle) for handle in handles(record))):
            raise RuntimeError(f"Cannot change dependencies after evaluation stage {name} has started")
    if (sampling_artifacts_started(artifacts, METHODS, include_combined=True)
            or any(folder.exists() and any(folder.iterdir())
                   for folder in (artifacts.results / method for method in METHODS))):
        raise RuntimeError("Cannot change evaluation dependencies after evaluation artifacts exist")
    return {"at": now(), "changes": changes,
            "evaluation_stages_verified_pending_and_not_started": True,
            "evaluation_handles_verified_terminal_or_missing": True,
            "evaluation_artifacts_verified_absent": True}


def check_identity_migration(state, identity, artifacts):
    previous = normalize_identity(state["identity"])
    current = normalize_identity(identity)
    changed = {key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key)}
    if not changed:
        return
    allowed = {"fp_factorized_attention", "factorized_training", "factorized_inference"}
    if (changed - allowed or not set(previous["factorized_training"]) <= set(current["factorized_training"])
            or previous["fp_factorized_attention"] and not current["fp_factorized_attention"]
            or previous["factorized_inference"] and not current["factorized_inference"]):
        raise RuntimeError("Controller settings differ from persisted status.json; only monotonic factorization additions are allowed")
    newly_affected = factorized_training_methods(current) - factorized_training_methods(previous)
    verified_handles = {}
    for method in sorted(newly_affected):
        record = state["tasks"].get(f"train:{method}", {})
        previous_handles = handles(record)
        if any(is_live(handle) for handle in previous_handles):
            raise RuntimeError(f"Cannot enable factored training for {method} while a previous process handle is live")
        if (record.get("status") in ("launching", "running")
                and time.time() - record.get("launched_timestamp", 0) < 30):
            raise RuntimeError(f"Cannot enable factored training for {method} while process registration is pending")
        if record.get("status") == "succeeded" or artifacts.checkpoint_path(method).exists():
            raise RuntimeError(f"Cannot change attention mode after {method} training has completed")
        dependent_methods = {method, "two_for_one"} if method == "diffusion" else {method}
        dependent_tasks = {f"{phase}:{name}" for phase in ("iid", "simulation") for name in dependent_methods}
        dependent_tasks.add("metrics:all")
        if (any(stage_started(state["tasks"].get(name, {})) for name in dependent_tasks)
                or sampling_artifacts_started(artifacts, dependent_methods)):
            raise RuntimeError(f"Cannot change {method} attention mode after dependent evaluation stages have started")
        verified_handles[method] = previous_handles
    if current["factorized_inference"] and not previous["factorized_inference"]:
        if (any(stage_started(record) for name, record in state["tasks"].items()
                if name.split(":", 1)[0] in ("iid", "simulation", "metrics"))
                or sampling_artifacts_started(artifacts, METHODS, include_combined=True)):
            raise RuntimeError("Cannot enable factored inference after evaluation stages have started")
    migration = {"at": now(), "changes": {key: {"previous": previous[key], "current": current[key]}
                                               for key in sorted(changed)},
                 "newly_affected_training_methods": sorted(newly_affected),
                 "previous_training_handles": verified_handles,
                 "previous_training_handles_verified_terminal_or_missing": True}
    if "fp_factorized_attention" in changed:
        migration.update(setting="fp_factorized_attention", previous=False, current=True,
                         previous_fp_handles=verified_handles.get("fokker_planck", []),
                         previous_fp_handles_verified_terminal_or_missing=True)
    state.setdefault("identity_migrations", []).append(migration)


def validate_protocol(config):
    s, t = config["evaluation"], config["training"]
    assert len(s["peptides"]) == 15 and set(s["peptides"]) == PEPTIDES, "the full experiment requires the 15 paper peptides"
    assert set(s["methods"]) == set(METHODS), "all four evaluation methods are required"
    assert s["num_iid_samples"] == 49000 and s["diffusion_steps"] == 1000, "full independent sampling is required"
    assert s["simulation_steps"] == 6000000 and s["replicas"] == 10 and s["save_every"] == 50, "full simulation protocol is required"
    assert s["dt"] == .0005 and s["temperature"] == 310. and s["gamma"] == 1., "physical protocol mismatch"
    assert s["eval_time"] == .00001 and s["two_for_one_time"] == .02, "simulation score-time mismatch"
    assert t["epochs"] == 120 and t["mixture_epochs"] == [10,20,100] and t["batch_size"] == 1024, "full training schedule is required"
    assert set(t["methods"]) == set(TRAIN_METHODS) and t["fp_weight"] == .0005, "training method/FP coefficient mismatch"


def controller(args, config, state_dir):
    artifacts = Artifacts(config)
    status_path = state_dir / "status.json"
    identity = {"config": config, "gpus": args.gpus, "fp_gpus": args.fp_gpus, "microbatches": args.microbatches,
                "compile": args.compile, "cuda_graphs": args.cuda_graphs, "inference_cache": args.inference_cache,
                "python": sys.executable, "fp_factorized_attention": bool(getattr(args, "fp_factorized_attention", False)),
                "factorized_training": parse_factorized_training(getattr(args, "factorized_training", [])),
                "factorized_inference": bool(getattr(args, "factorized_inference", False))}
    state = read_json(status_path) if status_path.exists() else {"identity": identity, "created_at": now(), "tasks": {}}
    graph = task_graph()
    dependency_migration = plan_evaluation_dependency_migration(state, graph, artifacts)
    check_identity_migration(state, identity, artifacts)
    state["identity"] = identity
    if dependency_migration:
        for name, change in dependency_migration["changes"].items():
            state["tasks"][name]["dependencies"] = change["current"].copy()
        state.setdefault("evaluation_dependency_migrations", []).append(dependency_migration)
    for name, definition in graph.items():
        state["tasks"].setdefault(name, {"status": "pending", **definition})
    state["controller"] = process_identity(os.getpid())
    state["status"] = "running"
    workers = {}

    def save():
        state["updated_at"] = now()
        atomic_json(status_path, state)

    def audit(record, name):
        try:
            record.update(status="succeeded", artifacts=artifacts.task(name), finished_at=now())
            record.pop("error", None)
            return True
        except Exception as exc:
            record.update(status="failed", error=f"Artifact audit failed: {type(exc).__name__}: {exc}", finished_at=now())
            return False

    # Adopt live workers before considering any resumable checkpoint or new launch.
    for name, record in state["tasks"].items():
        live = any(is_live(handle) for handle in handles(record))
        if live:
            record["status"] = "running"
        elif record["status"] in ("running", "launching"):
            result = read_json(record["result_file"]) if record.get("result_file") and Path(record["result_file"]).exists() else None
            if result is not None:
                if result["returncode"] == 0:
                    audit(record, name)
                else:
                    record.update(status="failed", error=f"Stage exited {result['returncode']}: {result.get('error')}; see {record.get('log')}")
            elif time.time() - record.get("launched_timestamp", 0) < 30:
                # A worker launched just before a controller crash may still be
                # publishing its /proc identity. Do not race that registration.
                record["status"] = "launching"
            else:
                record.update(status="pending", recovery="Previous process handles are terminal or missing; resume saved progress")
        elif record["status"] == "succeeded":
            audit(record, name)
        if args.retry_failed and record["status"] in ("failed", "blocked") and not live:
            record.update(status="pending", recovery="Explicit --retry-failed request")
    save()

    def choose_gpus(name, busy):
        phase, method = name.split(":")
        if phase == "metrics":
            return []
        if phase == "train":
            required = args.fp_gpus if method == "fokker_planck" and args.fp_gpus else [args.gpus[TRAIN_METHODS.index(method)]]
            return required if not set(required) & busy else None
        preferred = args.gpus[3] if method == "two_for_one" else args.gpus[TRAIN_METHODS.index(method)]
        return next(([gpu] for gpu in [preferred] + args.gpus if gpu not in busy), None)

    def command(name, gpu_ids):
        phase, method = name.split(":")
        if phase == "train":
            cmd = [sys.executable, "-u", str(REPO / "train.py"), "--config", str(args.config), "--method", method,
                   "--device", "cuda:0", "--microbatch", str(args.microbatches[TRAIN_METHODS.index(method)])]
            if method == "fokker_planck" and args.fp_gpus:
                cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpu_ids)}"] + cmd[2:] + ["--distributed"]
            if method in factorized_training_methods(identity):
                cmd.append("--factorized-attention")
            latest = artifacts.checkpoint_path(method, "latest.pt")
            if latest.exists():
                cmd += ["--resume", str(latest)]
        else:
            cmd = [sys.executable, "-u", str(REPO / "evaluate.py"), "--config", str(args.config), "--method", method,
                   "--device", "cpu" if phase == "metrics" else "cuda:0", f"--{phase}-only"]
            if phase != "metrics":
                if getattr(args, "factorized_inference", False):
                    cmd.append("--factorized-attention")
                if args.cuda_graphs:
                    cmd.append("--cuda-graphs")
                if args.inference_cache:
                    cmd.append("--inference-cache")
        if args.compile and phase != "metrics":
            cmd.append("--compile")
        return cmd

    while True:
        for name, record in state["tasks"].items():
            if name in workers:
                workers[name].poll()  # Reap exited children so /proc never treats zombies as live.
            if record["status"] not in ("running", "launching"):
                continue
            if any(is_live(handle) for handle in handles(record)):
                record["status"] = "running"
                continue
            result = read_json(record["result_file"]) if Path(record["result_file"]).exists() else None
            if result is not None:
                if result["returncode"] == 0:
                    audit(record, name)
                else:
                    record.update(status="failed", error=f"Stage exited {result['returncode']}: {result.get('error')}; see {record['log']}", finished_at=now())
            elif record["status"] == "launching" and time.time() - record["launched_timestamp"] < 30:
                continue
            else:
                record.update(status="failed", error="Worker and child handles are terminal or missing without an exit result", finished_at=now())

        busy = {gpu for record in state["tasks"].values() if record["status"] in ("running", "launching") for gpu in record.get("assigned_gpus", [])}
        for name, record in state["tasks"].items():
            if record["status"] != "pending":
                continue
            dependencies = [state["tasks"][dep]["status"] for dep in record["dependencies"]]
            if any(status in ("failed", "blocked") for status in dependencies):
                record.update(status="blocked", error="A required preceding stage failed")
                continue
            if not all(status == "succeeded" for status in dependencies):
                continue
            # A previous successful run may already have authoritative artifacts.
            try:
                record.update(status="succeeded", artifacts=artifacts.task(name), finished_at=now())
                record.pop("error", None)
                continue
            except (FileNotFoundError, AssertionError, KeyError, ValueError, OSError, RuntimeError):
                pass
            gpu_ids = choose_gpus(name, busy)
            if gpu_ids is None:
                continue
            run_id = uuid.uuid4().hex[:12]
            stage_dir = state_dir / "stages" / name.replace(":", "_")
            stage_dir.mkdir(parents=True, exist_ok=True)
            record.update(status="launching", assigned_gpus=gpu_ids, launched_timestamp=time.time(), started_at=now(),
                          handle_file=str(stage_dir / f"{run_id}_handles.json"), result_file=str(stage_dir / f"{run_id}_result.json"),
                          log=str(stage_dir / f"{run_id}.log"), command=command(name, gpu_ids))
            record.pop("error", None)
            spec = {key: record[key] for key in ("handle_file", "result_file", "log", "command")}
            spec["env"] = {"CUDA_VISIBLE_DEVICES": ",".join(gpu_ids), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            spec_path = stage_dir / f"{run_id}_spec.json"
            atomic_json(spec_path, spec)
            save()  # Persist the launch intent before creating the independent worker.
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                                      cwd=REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            workers[name] = worker
            record.update(status="running", worker=process_identity(worker.pid))
            busy.update(gpu_ids)
            print(f"[{now()}] Started {name} on GPUs {gpu_ids}; log {record['log']}", flush=True)
            save()

        statuses = [record["status"] for record in state["tasks"].values()]
        if all(status == "succeeded" for status in statuses):
            try:
                report = artifacts.complete()
                atomic_json(state_dir / "completion_audit.json", report)
                state.update(status="complete", completed_at=now(), audit=str(state_dir / "completion_audit.json"))
                save()
                print(f"[{now()}] Full replication complete and audited: {state['audit']}", flush=True)
                return 0
            except Exception as exc:
                state.update(status="failed", error=f"Final completion audit failed: {type(exc).__name__}: {exc}")
                save()
                print(state["error"], file=sys.stderr, flush=True)
                return 1
        if not any(status in ("running", "launching") for status in statuses):
            state.update(status="failed", error="No runnable stages remain; inspect failed stages and their logs")
            save()
            print(state["error"], file=sys.stderr, flush=True)
            return 1
        save()
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/dipeptides.yaml")
    parser.add_argument("--state-dir", type=Path, default=REPO / "data/experiment")
    parser.add_argument("--gpus", default="0,1,2,3", help="Four visible physical GPU IDs, in training preference order")
    parser.add_argument("--fp-gpus", help="Optional two GPU IDs reserved for synchronous distributed FP training")
    parser.add_argument("--fp-factorized-attention", action="store_true",
                        help="Use exact factored edge attention only during FP training")
    parser.add_argument("--factorized-training", default="", help="CSV training methods using factored potential experts; direct-score experts stay original")
    parser.add_argument("--factorized-inference", action="store_true", help="Use factored attention for future independent sampling and simulations")
    parser.add_argument("--microbatches", default="1024,1024,512", help="Diffusion, Mixture, FP microbatch sizes")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--inference-cache", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--status", action="store_true", help="Read status and check recorded process identities without changing files")
    parser.add_argument("--audit-only", action="store_true", help="Verify all existing full experiment artifacts without running stages")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry terminal failed stages, resuming their saved progress")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    try:
        args.factorized_training = parse_factorized_training(args.factorized_training)
    except ValueError as exc:
        parser.error(str(exc))
    state_dir = args.state_dir.resolve()
    if args.status:
        state = read_json(state_dir / "status.json")
        state["controller_live_now"] = is_live(state.get("controller"))
        for record in state["tasks"].values():
            record["live_now"] = any(is_live(handle) for handle in handles(record))
        print(json.dumps(state, indent=2))
        return 0
    args.config = args.config.resolve()
    args.gpus = parse_list(args.gpus)
    args.fp_gpus = parse_list(args.fp_gpus) if args.fp_gpus else []
    args.microbatches = [int(value) for value in parse_list(args.microbatches)]
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        parser.error("--gpus requires four distinct GPU IDs")
    if args.fp_gpus and (len(args.fp_gpus) != 2 or len(set(args.fp_gpus)) != 2 or set(args.fp_gpus) - set(args.gpus)):
        parser.error("--fp-gpus requires two distinct IDs from --gpus")
    if len(args.microbatches) != 3 or any(value < 1 or value > 1024 for value in args.microbatches):
        parser.error("--microbatches requires three values between 1 and 1024")
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    import yaml
    config = yaml.safe_load(args.config.read_text())
    validate_protocol(config)
    if args.audit_only:
        print(json.dumps(Artifacts(config).complete(), indent=2))
        return 0
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another controller holds {state_dir / 'controller.lock'}; use --status")
        return controller(args, config, state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
