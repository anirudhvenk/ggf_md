#!/usr/bin/env python3
"""Run/resume the isolated GGF experiment without claiming occupied GPUs.

The controller owns only its lock. Detached workers own their subprocesses and
publish durable process handles/results, so restarting the controller adopts
live work. Failed stages require --retry-failed. --status never writes files.
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
PROTECTED = REPO
PEPTIDES = {"KS", "AT", "LW", "KQ", "NY", "IM", "TD", "HT", "NF", "RL", "ET", "AC", "RV", "HP", "AP"}
PROTOCOL_KEYS = ("peptides", "seed", "simulation_steps", "save_every", "replicas", "eval_time",
                 "two_for_one_time", "dt", "gamma", "temperature")
ACTIVE = {"launching", "running"}
REGISTRATION_SECONDS = 30
_LARGE_FILE_CHECKS = {}
BASELINE = REPO / "data/ggf/baselines/non-fokker-planck"
BASELINE_PINS = {str(BASELINE / "report_receipt.json"): "577fc4b1b5e389f36be149c74684efb4c4a646c4917b4a74d5aaa00214fab1bc",
                 str(BASELINE / "summary.json"): "22f17b072600c3c429008f2daf1cf33030d09db9090f0c4c1adf6c697c53acc6"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def process_identity(pid):
    """Read only a recorded PID's public lifetime fields; reject PID reuse/zombies."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
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
    path = record.get("handle_file")
    if path and Path(path).exists():
        handle = read_json(path)
        if handle.get("attempt") != record.get("attempt"):
            raise RuntimeError("Process handle belongs to a different stage attempt")
        result.extend([handle.get("worker"), handle.get("child")])
    return [item for item in result if item]


def verify_files(pins):
    for path, expected in pins.items():
        target = Path(path)
        if not target.is_file():
            raise RuntimeError(f"Pinned workload input changed: {path}")
        stat = target.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
        # Source/config bytes are hashed every check. Large immutable checkpoints
        # are hashed once per process and again whenever any file metadata changes.
        if stat.st_size > 8 << 20 and _LARGE_FILE_CHECKS.get(path) == signature:
            continue
        if digest(target) != expected:
            raise RuntimeError(f"Pinned workload input changed: {path}")
        _LARGE_FILE_CHECKS[path] = signature


def validate_destination(path):
    lexical = Path(os.path.abspath(path))
    path = Path(path).resolve()
    baseline = BASELINE.resolve()
    if path == baseline or baseline in path.parents:
        raise ValueError(f"Refusing a write destination in the pinned baseline snapshot: {path}")
    protected_data = [(REPO / "data" / name).resolve()
                      for name in ("runs", "results", "experiment", "processed", "downloads", "raw")]
    if any(path == root or root in path.parents for root in protected_data):
        raise ValueError(f"Refusing a write destination in protected experiment data: {path}")
    # Keep this namespace literal: resolving it would incorrectly authorize a
    # data/ggf alias into another protected repository directory or old NFS data.
    dedicated = REPO / "data/ggf"
    for candidate in (lexical, path):
        if ((candidate == PROTECTED or PROTECTED in candidate.parents)
                and candidate != dedicated and dedicated not in candidate.parents):
            raise ValueError(f"Refusing a write destination in the original experiment: {path}")
    return path


def task_graph():
    return {"train": [], "iid": ["train"], "simulation": ["train"],
            "metrics": ["iid", "simulation"]}


def query_gpus():
    """Minimal, fresh inventory; never read another process's command/environment."""
    def rows(query):
        output = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=15).stdout
        return list(csv.reader(output.splitlines(), skipinitialspace=True))

    inventory = rows("--query-gpu=index,uuid,memory.total,memory.used")
    apps = rows("--query-compute-apps=gpu_uuid,pid")
    occupied = {}
    for row in apps:
        if len(row) != 2:
            raise RuntimeError("Unrecognized nvidia-smi compute application inventory")
        gpu_uuid, pid = [part.strip() for part in row]
        occupied.setdefault(gpu_uuid, []).append(int(pid))
    result = []
    for row in inventory:
        if len(row) != 4:
            raise RuntimeError("Unrecognized nvidia-smi GPU inventory")
        index, gpu_uuid, total, used = [part.strip() for part in row]
        total, used = int(total), int(used)
        result.append({"index": index, "uuid": gpu_uuid, "total_mib": total, "used_mib": used,
                       "compute_pids": occupied.get(gpu_uuid, []),
                       "idle": not occupied.get(gpu_uuid) and used <= max(512, total * .01)})
    return result


def choose_gpu(preferences, snapshot, busy):
    by_name = {name: gpu for gpu in snapshot for name in (gpu["index"], gpu["uuid"])}
    for name in preferences:
        if name not in by_name:
            raise RuntimeError(f"Preferred GPU {name} absent from fresh inventory")
        gpu = by_name[name]
        if gpu["idle"] and gpu["uuid"] not in busy:
            return gpu
    return None


def run_worker(spec_path):
    """Only this detached wrapper launches its stage and records its actual exit."""
    spec = read_json(spec_path)
    handle = {"attempt": spec["attempt"], "worker": process_identity(os.getpid()), "started_at": now()}
    atomic_json(spec["handle_file"], handle)
    code, error, child = 127, None, None
    try:
        verify_files(spec["pins"])
        env = os.environ.copy()
        env.update(spec["env"])
        with Path(spec["stdout"]).open("a", buffering=1) as output, Path(spec["stderr"]).open("a", buffering=1) as errors:
            commands = [spec["command"]] + ([spec["post_command"]] if spec.get("post_command") else [])
            handle["command_results"] = []
            for index, command in enumerate(commands):
                verify_files(spec["pins"])
                output.write(f"[{now()}] Starting command {index}: {command}\n")
                child = subprocess.Popen(command, cwd=spec["cwd"], env=env,
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
                handle.update(child=process_identity(child.pid), command_index=index)
                atomic_json(spec["handle_file"], handle)
                while child.poll() is None:
                    verify_files(spec["pins"])
                    time.sleep(.5)
                code = child.returncode
                verify_files(spec["pins"])
                handle["command_results"].append({"index": index, "returncode": code, "finished_at": now()})
                atomic_json(spec["handle_file"], handle)
                output.write(f"[{now()}] Command {index} exit code: {code}\n")
                if code != 0:
                    break
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        # These are only our exact recorded child handles, never GPU inventory PIDs.
        if child is not None and child.poll() is None and is_live(handle.get("child")):
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if is_live(handle.get("child")):
                    child.kill()
                child.wait()
    atomic_json(spec["result_file"], {"attempt": spec["attempt"], "returncode": code, "error": error,
                                      "finished_at": now(), "pins": spec["pins"], **handle})
    return code if not error and 0 <= code < 256 else 1


def validate_protocol(config):
    s, t = config["evaluation"], config["training"]
    conditions = [
        (len(s["peptides"]) == 15 and set(s["peptides"]) == PEPTIDES, "15 paper peptides"),
        (s["methods"] == ["ggf"] and t["methods"] == ["ggf"], "GGF-only methods"),
        (s["num_iid_samples"] == 49000 and s["diffusion_steps"] == 1000, "49,000 IID samples/1,000 steps"),
        (s["simulation_steps"] == 6000000 and s["replicas"] == 10 and s["save_every"] == 50, "full simulation"),
        (s["dt"] == .0005 and s["temperature"] == 310. and s["gamma"] == 1., "physical protocol"),
        (s["eval_time"] == .00001, "simulation score time"),
        (t["epochs"] == 120 and t["batch_size"] == 1024 and t["seed"] == 1, "120 epochs/1,024 batch/seed 1"),
    ]
    for valid, description in conditions:
        if not valid:
            raise ValueError(f"GGF experiment requires {description}")
    root = Path(config["data"]["root"])
    validate_destination(root)
    validate_destination(root / "runs/ggf")
    validate_destination(root / "results")


class Artifacts:
    """Structural completion checks retain divergence and nullable metric values."""

    def __init__(self, config, ggf, execution=None):
        import numpy as np
        import torch
        self.np, self.torch, self.config, self.ggf = np, torch, config, ggf
        self.root = Path(config["data"]["root"]).resolve()
        self.results = self.root / "results"
        self.directory = self.results / "ggf"
        self.settings = config["evaluation"]
        self.execution = execution

    def checkpoint(self, name="final.pt"):
        return self.root / "runs/ggf" / f"seed{self.config['training']['seed']}" / name

    def load(self, path):
        return self.torch.load(path, map_location="cpu", weights_only=False)

    def norm(self):
        return read_json(self.root / "processed/manifest.json")["norm_factor"]

    def compatible_checkpoint(self, path, execution=None):
        state = self.load(path)
        if (state.get("method") != "ggf" or state.get("ggf") != self.ggf
                or any(state["config"].get(k) != self.config[k] for k in ("model", "training", "ggf"))):
            raise ValueError(f"Incompatible GGF checkpoint: {path}")
        if not math.isclose(state["norm_factor"], self.norm(), rel_tol=1e-8):
            raise ValueError(f"Checkpoint normalization differs: {path}")
        if execution is not None:
            saved = state.get("execution", {})
            if (saved.get("microbatch") != execution["microbatch"]
                    or saved.get("factorized_attention", False) != execution["factorized_attention"]):
                raise ValueError(f"Checkpoint execution settings differ: {path}")
        return state

    def resume_checkpoint(self, execution):
        latest = self.checkpoint("latest.pt")
        if latest.exists():
            self.compatible_checkpoint(latest, execution)
            return latest
        return None

    def has_outputs(self):
        return (any((self.root / "runs/ggf").rglob("*.pt"))
                or self.directory.exists() and any(self.directory.iterdir())
                or any((self.results / name).exists() for name in ("summary.json", "per_peptide.csv", "comparison.csv")))

    def training(self):
        state = self.compatible_checkpoint(self.checkpoint(), self.execution)
        t = self.config["training"]
        if not state.get("completed") or state["epoch"] != t["epochs"] or state.get("step_in_epoch", 0) != 0:
            raise ValueError("Training checkpoint does not complete the requested schedule")
        manifest = read_json(self.root / "processed/manifest.json")
        frames = sum(item["samples"] for item in manifest["splits"]["train"].values())
        if state["global_step"] != t["epochs"] * (frames // t["batch_size"]):
            raise ValueError("Optimizer step count differs from the requested schedule")
        if (not state.get("ema") or not state.get("model") or set(state["ema"]) != set(state["model"])
                or any(state["model"][k].shape != state["ema"][k].shape for k in state["model"])
                or not all(self.torch.isfinite(tensor).all().item()
                           for weights in (state["ema"], state["model"]) for tensor in weights.values())):
            raise ValueError("Missing, incompatible or nonfinite model/EMA weights")
        from model import DipeptideModel
        architecture = DipeptideModel("ggf", self.config).state_dict()
        if (set(architecture) != set(state["model"])
                or any(architecture[k].shape != state["model"][k].shape for k in architecture)):
            raise ValueError("Checkpoint weight structure differs from the GGF architecture")
        return {"epochs": state["epoch"], "optimizer_steps": state["global_step"],
                "checkpoint_sha256": digest(self.checkpoint()), "ggf": self.ggf}

    def iid(self, full=False):
        s, np = self.settings, self.np
        expected = {"checkpoint": digest(self.checkpoint()), "ggf": self.ggf, "norm_factor": self.norm(),
                    "samples": s["num_iid_samples"], "steps": s["diffusion_steps"], "seed": s["seed"]}
        metadata = read_json(self.directory / "iid_metadata.json")
        if self.execution is not None:
            expected.update(compiled=self.execution["compile"], inference_cache=self.execution["inference_cache"],
                            batch=s["inference_batch"])
            if bool(metadata.get("factorized_attention", False)) != self.execution["factorized_attention"]:
                raise ValueError("IID attention execution setting differs")
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("IID checkpoint/GGF/protocol identity differs")
        finite_counts = {}
        for peptide in s["peptides"]:
            state = self.load(self.directory / f"{peptide}_iid_state.pt")
            if state["completed"] != s["num_iid_samples"] or state["identity"] != metadata:
                raise ValueError(f"Incomplete or incompatible IID state: {peptide}")
            self.validate_rng(state.get("rng"), f"IID {peptide}")
            array = np.load(self.directory / f"{peptide}_iid.npy", mmap_mode="r")
            if array.shape != (s["num_iid_samples"], 30) or array.dtype != np.float32:
                raise ValueError(f"IID shape/dtype differs: {peptide}")
            if full:
                finite_counts[peptide] = int(np.isfinite(array).all(axis=1).sum())
        report = {"peptides": len(s["peptides"]), "samples_per_peptide": s["num_iid_samples"],
                  "checkpoint_sha256": expected["checkpoint"]}
        if full:
            report["finite_coordinate_samples"] = finite_counts
        return report

    def validate_rng(self, rng, label):
        if (not isinstance(rng, self.torch.Tensor) or rng.dtype != self.torch.uint8
                or rng.device.type != "cpu" or rng.ndim != 1 or rng.numel() == 0):
            raise ValueError(f"{label} RNG must be a nonempty native CPU uint8 vector")

    def simulation(self, full=False):
        s, np = self.settings, self.np
        state = self.load(self.directory / "simulation_state.pt")
        expected = {"checkpoint": digest(self.checkpoint()), "ggf": self.ggf, "norm_factor": self.norm(),
                    "method": "ggf", **{k: s[k] for k in PROTOCOL_KEYS}}
        if self.execution is not None:
            expected.update(compiled=self.execution["compile"], inference_cache=self.execution["inference_cache"],
                            cuda_graphs=self.execution["cuda_graphs"])
            if bool(state["identity"].get("factorized_attention", False)) != self.execution["factorized_attention"]:
                raise ValueError("Simulation attention execution setting differs")
        if (any(state["identity"].get(k) != v for k, v in expected.items())
                or read_json(self.directory / "simulation_metadata.json") != state["identity"]):
            raise ValueError("Simulation checkpoint/GGF/protocol identity differs")
        if state["completed"] != s["simulation_steps"]:
            raise ValueError("Simulation has not completed 6,000,000 steps")
        alive = state["alive"]
        total = len(s["peptides"]) * s["replicas"]
        if not isinstance(alive, np.ndarray) or alive.dtype != np.bool_ or alive.shape != (total,):
            raise ValueError("Simulation alive paths must be a native NumPy bool vector with the protocol shape")
        if any(not isinstance(state[key], self.torch.Tensor) or state[key].dtype != self.torch.float32
               or state[key].device.type != "cpu" or state[key].shape != (total, 30) for key in ("x", "v")):
            raise ValueError("Simulation state positions/velocities must be CPU float32 with the protocol shape")
        self.validate_rng(state.get("rng"), "Simulation")
        if full and any(not self.torch.isfinite(state[key][alive]).all().item() for key in ("x", "v")):
            raise ValueError("Simulation has nonfinite surviving final positions/velocities")
        marker = read_json(self.directory / "simulation_complete.json")
        if (marker.get("complete") is not True or marker["steps_per_path"] != s["simulation_steps"]
                or marker["replicas"] != s["replicas"]
                or marker["aggregate_ns_per_peptide"] != s["simulation_steps"] * s["dt"] * s["replicas"] / 1000
                or marker["alive_paths"] != alive.tolist()):
            raise ValueError("Simulation completion marker differs from state")
        counts = {}
        for i, peptide in enumerate(s["peptides"]):
            array = np.load(self.directory / f"{peptide}_simulation.npy", mmap_mode="r")
            if array.shape != (s["replicas"], s["simulation_steps"] // s["save_every"], 30) or array.dtype != np.float32:
                raise ValueError(f"Simulation array shape/dtype differs: {peptide}")
            retained = alive[i*s["replicas"]:(i+1)*s["replicas"]]
            counts[peptide] = {"alive": int(retained.sum()), "dead": int((~retained).sum())}
            if full:
                for replica in np.flatnonzero(retained):
                    if not np.isfinite(array[replica]).all():
                        raise ValueError(f"Retained simulation path contains nonfinite coordinates: {peptide}/{replica}")
                    if array[replica, -1].tobytes() != state["x"][i*s["replicas"] + replica].numpy().tobytes():
                        raise ValueError(f"Last stored frame differs bytewise from final state: {peptide}/{replica}")
        return {"steps_per_path": s["simulation_steps"], "replicas_per_peptide": s["replicas"],
                "alive_paths": int(alive.sum()), "dead_paths": int((~alive).sum()), "per_peptide": counts,
                "checkpoint_sha256": expected["checkpoint"]}

    def metrics(self):
        s = self.settings
        with (self.results / "per_peptide.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(s["peptides"]) or {(r["method"], r["peptide"]) for r in rows} != {("ggf", p) for p in s["peptides"]}:
            raise ValueError("Metrics require exactly the 15 GGF peptide rows")
        metadata = read_json(self.directory / "metrics_metadata.json")
        expected = {"ggf": self.ggf, **{k: s[k] for k in PROTOCOL_KEYS + ("num_iid_samples", "diffusion_steps", "bins")}}
        if metadata != expected:
            raise ValueError("Metrics GGF/protocol identity differs")
        summary = read_json(self.results / "summary.json")
        if summary["settings"] != s or set(summary["methods"]) != {"ggf"} or summary.get("ggf") != self.ggf:
            raise ValueError("Summary protocol/method differs")
        raw_rows = read_json(self.directory / "metrics.json")
        if len(raw_rows) != len(rows) or {row["peptide"] for row in raw_rows} != set(s["peptides"]):
            raise ValueError("Per-method metrics JSON is incomplete")
        simulation = self.load(self.directory / "simulation_state.pt")
        alive = self.np.asarray(simulation["alive"], dtype=bool).reshape(len(s["peptides"]), s["replicas"])
        by_peptide = {row["peptide"]: row for row in raw_rows}
        for i, peptide in enumerate(s["peptides"]):
            row = by_peptide[peptide]
            if (row["method"] != "ggf" or row["simulation_alive_paths"] != int(alive[i].sum())
                    or row["simulation_dead_paths"] != int((~alive[i]).sum())
                    or row["simulation_total_paths"] != s["replicas"]):
                raise ValueError(f"Reported simulation path counts differ: {peptide}")
            for source in ("iid", "simulation"):
                count = row[f"{source}_valid_samples"]
                if not 0 <= count <= s["num_iid_samples"]:
                    raise ValueError(f"Invalid reported sample count: {peptide}/{source}")
                for metric in ("jsd", "js_distance", "pmf", "code_jsd", "code_pmf", "code_mjs"):
                    value = row[f"{source}_{metric}"]
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"Nonfinite JSON metric: {peptide}/{source}/{metric}")
                    if count == 0 and (value is not None or not row.get(f"{source}_missing_reason")):
                        raise ValueError(f"Missing samples must retain null metrics/reason: {peptide}/{source}")
        for name, metric in summary["methods"]["ggf"].items():
            source = "iid" if name.startswith("iid_") else "simulation"
            coverage = sum(row.get(name) is not None and math.isfinite(row[name]) for row in raw_rows)
            complete = coverage == len(raw_rows) and all(row[f"{source}_valid_samples"] == s["num_iid_samples"] for row in raw_rows)
            if (metric["expected_peptides"] != len(raw_rows) or metric["evaluated_peptides"] != coverage
                    or (not complete and (metric["mean"] is not None or metric["std_across_peptides"] is not None))):
                raise ValueError(f"Summary coverage/null aggregate differs: {name}")
        for peptide in s["peptides"]:
            for suffix in ("pmf_1d.png", "pmf_2d.png"):
                if (self.directory / f"{peptide}_{suffix}").stat().st_size <= 0:
                    raise ValueError(f"Missing plot: {peptide}/{suffix}")
            with self.np.load(self.directory / f"{peptide}_angles.npz") as angles:
                if (angles["iid"].shape != (s["num_iid_samples"], 2)
                        or angles["simulation"].ndim != 2 or angles["simulation"].shape[1] != 2):
                    raise ValueError(f"Invalid evaluated angles shape: {peptide}")
                for source in ("iid", "simulation"):
                    if int(self.np.isfinite(angles[source]).all(axis=1).sum()) != by_peptide[peptide][f"{source}_valid_samples"]:
                        raise ValueError(f"Angle and metric sample counts differ: {peptide}/{source}")
        with (self.results / "comparison.csv").open() as stream:
            comparison = list(csv.DictReader(stream))
        if len(comparison) != 8 or any(row["method"] != "ggf" for row in comparison):
            raise ValueError("GGF comparison requires 8 paper/released-code rows")
        for filename in ("baseline_comparison.csv", "GGF_RESULTS.md"):
            if (self.results / filename).stat().st_size <= 0:
                raise ValueError(f"Missing baseline comparison report: {filename}")
        report = read_json(self.results / "baseline_comparison_metadata.json")
        if (report.get("ggf") != self.ggf or report["ggf_summary_sha256"] != digest(self.results / "summary.json")
                or report["baseline_receipt_sha256"] != BASELINE_PINS[str(BASELINE / "report_receipt.json")]
                or report["baseline_summary_sha256"] != BASELINE_PINS[str(BASELINE / "summary.json")]
                or report["comparison_rows"] != 24
                or any(report["output_sha256"][name] != digest(self.results / name)
                       for name in ("baseline_comparison.csv", "GGF_RESULTS.md"))):
            raise ValueError("Baseline comparison report provenance differs")
        # JSON nulls and all reported path/sample counts are copied, not corrected.
        return {"method_peptide_rows": len(rows), "plots": 2*len(rows), "methods": ["ggf"],
                "reported_metrics": raw_rows, "summary": summary["methods"]["ggf"], "baseline_comparison": report}

    def audit(self, stage, full=False):
        return {"train": self.training, "iid": lambda: self.iid(full),
                "simulation": lambda: self.simulation(full), "metrics": self.metrics}[stage]()

    def complete(self):
        return {"status": "complete", "verified_at": now(), "ggf": self.ggf,
                "training": self.training(), "independent_sampling": self.iid(full=True),
                "simulation": self.simulation(full=True), "metrics": self.metrics()}


def build_identity(args, config):
    sys.path.insert(0, str(REPO))
    from ggf_score_matching import ggf_identity
    import numpy as np
    import torch
    inputs = [REPO / name for name in ("train.py", "evaluate.py", "model.py", "ggf_score_matching.py",
                                      "scripts/report_ggf_results.py")]
    inputs += [Path(__file__).resolve(), args.config, Path(config["data"]["root"]) / "processed/manifest.json"]
    pins = {str(p.resolve()): digest(p) for p in inputs}
    verify_files(BASELINE_PINS)
    pins.update(BASELINE_PINS)
    return {"schema": 1, "config": config, "config_path": str(args.config), "ggf": ggf_identity(config),
            "python": sys.executable, "pins": pins,
            "runtime_versions": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__},
            "execution": {"microbatch": args.microbatch, "compile": args.compile,
                          "factorized_attention": args.factorized_attention,
                          "inference_cache": args.inference_cache, "cuda_graphs": args.cuda_graphs}}


class Controller:
    def __init__(self, args, config, identity, artifacts, gpu_query=query_gpus, command_builder=None):
        self.args, self.config, self.identity, self.artifacts = args, config, identity, artifacts
        self.state_dir, self.gpu_query, self.command_builder = args.state_dir, gpu_query, command_builder
        self.status_path = self.state_dir / "status.json"
        self.workers = {}
        self.state = None

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.status_path, self.state)

    def prepare(self):
        verify_files(self.identity["pins"])
        if self.status_path.exists():
            saved = read_json(self.status_path)
            if saved["identity"] != self.identity:
                raise RuntimeError("Workload identity differs from persisted status; no cached output may be reused")
            if {k: v["dependencies"] for k, v in saved["tasks"].items()} != task_graph():
                raise RuntimeError("Persisted dependency graph differs")
            self.state = saved
        else:
            if self.artifacts.has_outputs():
                raise RuntimeError("Unregistered GGF outputs exist without matching controller provenance")
            self.state = {"identity": self.identity, "created_at": now(), "tasks": {
                name: {"status": "pending", "dependencies": deps} for name, deps in task_graph().items()}}
        self.state.update(controller=process_identity(os.getpid()), status="running", gpu_preferences=self.args.gpus)
        self.reconcile()
        for name, record in self.state["tasks"].items():
            if record["status"] == "succeeded":
                audited = self.artifacts.audit(name)
                if audited != record["artifacts"]:
                    raise RuntimeError(f"Previously completed artifacts changed: {name}")
            if self.args.retry_failed and record["status"] in {"failed", "blocked"}:
                if record.get("ambiguous_launch"):
                    raise RuntimeError(f"Cannot retry unregistered process ambiguity: {name}")
                if record.get("integrity_failure"):
                    raise RuntimeError(f"Cannot retry artifacts produced during a workload mutation: {name}")
                if any(is_live(h) for h in handles(record)):
                    raise RuntimeError(f"Cannot retry live stage: {name}")
                record.setdefault("attempt_history", []).append({k: v for k, v in record.items() if k != "attempt_history"})
                record.update(status="pending", recovery="Explicit --retry-failed")
                record.pop("error", None)
        self.save()

    def stage_pins(self, name):
        pins = dict(self.identity["pins"])
        if name != "train":
            pins[str(self.artifacts.checkpoint())] = self.state["tasks"]["train"]["artifacts"]["checkpoint_sha256"]
        return pins

    def reconcile(self):
        for name, record in self.state["tasks"].items():
            if name in self.workers:
                self.workers[name].poll()
            live = any(is_live(handle) for handle in handles(record))
            record["live_now"] = live
            if live:
                if record["status"] not in ACTIVE:
                    raise RuntimeError(f"A recorded {record['status']} stage still has a live process: {name}")
                record["status"] = "running"
                verify_files(record["pins"])
                continue
            if record["status"] not in ACTIVE:
                continue
            result_path = record.get("result_file")
            if result_path and Path(result_path).exists():
                result = read_json(result_path)
                if result.get("attempt") != record["attempt"] or result.get("pins") != record["pins"]:
                    raise RuntimeError(f"Stage result provenance differs: {name}")
                record["result"] = result
                record["finished_at"] = result["finished_at"]
                if result["returncode"] != 0 or result.get("error"):
                    record.update(status="failed", error=f"Stage exited {result['returncode']}: {result.get('error')}; see {record['stderr']}")
                    if "Pinned workload input changed:" in (result.get("error") or ""):
                        record["integrity_failure"] = True
                    continue
                try:
                    verify_files(record["pins"])
                    record.update(status="succeeded", artifacts=self.artifacts.audit(name))
                    record.pop("error", None)
                except Exception as exc:
                    record.update(status="failed", error=f"Artifact audit: {type(exc).__name__}: {exc}")
            elif time.time() - record["launched_timestamp"] < REGISTRATION_SECONDS:
                record["status"] = "launching"
            else:
                registered = handles(record)
                # A wrapper may die between spawning a child (including a report
                # post-command) and publishing that child's handle. Even a dead
                # registered wrapper/earlier child cannot prove no unregistered
                # subprocess exists. Only a durable exit result closes this gap.
                record.update(status="failed", finished_at=now(), ambiguous_launch=True,
                              error="Recorded processes are terminal/missing without a durable exit result; process registration is ambiguous" if registered
                              else "Launch intent has no registered process/result; refusing an ambiguous duplicate")

    def command(self, name):
        if self.command_builder:
            return self.command_builder(name)
        common = [sys.executable, "-u", str(REPO / ("train.py" if name == "train" else "evaluate.py")),
                  "--config", str(self.args.config), "--method", "ggf", "--device", "cpu" if name == "metrics" else "cuda:0"]
        if name == "train":
            common += ["--microbatch", str(self.args.microbatch)]
            latest = self.artifacts.resume_checkpoint(self.identity["execution"])
            if latest:
                common += ["--resume", str(latest)]
        else:
            common += [f"--{name}-only"]
        if name != "metrics":
            for flag in ("compile", "factorized_attention"):
                if getattr(self.args, flag):
                    common.append("--" + flag.replace("_", "-"))
            if name != "train":
                for flag in ("inference_cache", "cuda_graphs"):
                    if getattr(self.args, flag):
                        common.append("--" + flag.replace("_", "-"))
        return common

    def launch(self, name, gpu):
        verify_files(self.stage_pins(name))
        record = self.state["tasks"][name]
        attempt = uuid.uuid4().hex
        folder = self.state_dir / "stages" / name / attempt
        folder.mkdir(parents=True)
        gpu_ids = [gpu["uuid"]] if gpu else []
        record.update(status="launching", attempt=attempt, assigned_gpus=gpu_ids,
                      gpu_snapshot=gpu, launched_timestamp=time.time(), started_at=now(),
                      handle_file=str(folder / "handles.json"), result_file=str(folder / "result.json"),
                      stdout=str(folder / "stdout.log"), stderr=str(folder / "stderr.log"),
                      command=self.command(name), pins=self.stage_pins(name))
        for field in ("worker", "child", "result", "error", "finished_at", "ambiguous_launch"):
            record.pop(field, None)
        # Private experiment directories keep scratch/cache out of shared paths.
        runtime = Path(self.config["data"]["root"]).resolve().parent / "runtime-tmp"
        compiler = runtime.parent / "compiler-cache"
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        compiler.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime.chmod(0o700)
        compiler.chmod(0o700)
        env = {"CUDA_VISIBLE_DEVICES": ",".join(gpu_ids), "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TORCHINDUCTOR_COMPILE_THREADS": "4",
               "TMPDIR": str(runtime), "TMP": str(runtime), "TEMP": str(runtime),
               "TORCHINDUCTOR_CACHE_DIR": str(compiler / "inductor"), "TRITON_CACHE_DIR": str(compiler / "triton")}
        spec = {key: record[key] for key in ("attempt", "handle_file", "result_file", "stdout", "stderr", "command", "pins")}
        if name == "metrics" and self.command_builder is None:
            spec["post_command"] = [sys.executable, str(REPO / "scripts/report_ggf_results.py"),
                                    "--config", str(self.args.config)]
            record["post_command"] = spec["post_command"]
        spec.update(cwd=str(REPO), env=env)
        spec_path = folder / "spec.json"
        atomic_json(spec_path, spec)
        record["spec_file"] = str(spec_path)
        self.save()  # Launch intent is durable before creating an independent process.
        worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                                  cwd=REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        self.workers[name] = worker
        record.update(status="running", worker=process_identity(worker.pid))
        self.save()
        print(f"[{now()}] Started {name} on {gpu_ids or 'CPU'}; logs {folder}", flush=True)

    def tick(self):
        verify_files(self.identity["pins"])
        self.reconcile()
        busy = {gpu for r in self.state["tasks"].values() if r["status"] in ACTIVE for gpu in r.get("assigned_gpus", [])}
        for name, record in self.state["tasks"].items():
            if record["status"] != "pending":
                continue
            dependencies = [self.state["tasks"][dep]["status"] for dep in record["dependencies"]]
            if any(status in {"failed", "blocked"} for status in dependencies):
                record.update(status="blocked", error="A required preceding stage failed")
                continue
            if not all(status == "succeeded" for status in dependencies):
                continue
            gpu = None
            if name != "metrics":
                # Query again for every launch, including IID/simulation in one tick.
                snapshot = self.gpu_query()
                self.state["last_gpu_inventory"] = {"at": now(), "gpus": snapshot}
                gpu = choose_gpu(self.args.gpus, snapshot, busy)
                if gpu is None:
                    record["waiting_for"] = "An idle preferred GPU with no compute contexts and nearly free memory"
                    continue
            record.pop("waiting_for", None)
            self.launch(name, gpu)
            if gpu:
                busy.add(gpu["uuid"])
        statuses = [r["status"] for r in self.state["tasks"].values()]
        if all(s == "succeeded" for s in statuses):
            report = self.artifacts.complete()
            report["identity"] = self.identity
            atomic_json(self.state_dir / "completion_audit.json", report)
            self.state.update(status="complete", completed_at=now(), audit=str(self.state_dir / "completion_audit.json"))
            self.save()
            return 0
        if not any(s in ACTIVE or s == "pending" for s in statuses):
            self.state.update(status="failed", error="Stages failed; inspect logs and use --retry-failed explicitly")
            self.save()
            return 1
        self.state["status"] = "running" if any(s in ACTIVE for s in statuses) else "waiting_for_idle_gpu"
        self.save()
        return None

    def run(self):
        self.prepare()
        while True:
            result = self.tick()
            if result is not None:
                return result
            time.sleep(self.args.poll_seconds)


def status(state_dir):
    state = read_json(state_dir / "status.json")
    state["controller_live_now"] = is_live(state.get("controller"))
    for record in state["tasks"].values():
        record["live_now"] = any(is_live(handle) for handle in handles(record))
        record["live_handles_now"] = [handle for handle in handles(record) if is_live(handle)]
    print(json.dumps(state, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/dipeptides_ggf.yaml")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--gpus", default="3,2,1,0", help="One or more physical index/UUID preferences; only idle GPUs are used")
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--factorized-attention", action="store_true")
    parser.add_argument("--inference-cache", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--status", action="store_true", help="Read process liveness without modifying files or querying GPUs")
    parser.add_argument("--audit-only", action="store_true", help="Read and audit completed artifacts without executing stages")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    if args.status and args.state_dir:
        status(args.state_dir.resolve())
        return 0
    args.config = args.config.resolve()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    args.state_dir = (args.state_dir or Path(config["data"]["root"]) / "experiment").resolve()
    if args.status:
        status(args.state_dir)
        return 0
    args.gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus requires one or more distinct physical index/UUID preferences")
    if not 1 <= args.microbatch <= 1024 or not 1 <= args.poll_seconds <= 60:
        parser.error("--microbatch must be 1..1024 and --poll-seconds must be 1..60")
    validate_protocol(config)
    args.state_dir = validate_destination(args.state_dir)
    identity = build_identity(args, config)
    artifacts = Artifacts(config, identity["ggf"], identity["execution"])
    if args.audit_only:
        state = read_json(args.state_dir / "status.json")
        if state["identity"] != identity or any(any(is_live(h) for h in handles(r)) for r in state["tasks"].values()):
            raise RuntimeError("Audit requires matching immutable workload and no live stages")
        print(json.dumps(artifacts.complete(), indent=2, allow_nan=False))
        return 0
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another controller holds this experiment lock; use --status")
        controller = Controller(args, config, identity, artifacts)
        try:
            return controller.run()
        except Exception as exc:
            if controller.state is not None:
                if "Pinned workload input changed:" in str(exc):
                    for record in controller.state["tasks"].values():
                        if record["status"] in ACTIVE:
                            record["integrity_failure"] = True
                controller.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                controller.save()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
