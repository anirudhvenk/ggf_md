"""Prepare the released dipeptides and train the original experiment in PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch
import yaml

LOG = logging.getLogger("train")
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLY", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
ATOMS = ["N", "CA", "CB", "C", "O"]
# GitHub's SHA-256 release digests, retrieved from the official 1.0.0 release.
ASSETS = {
    "minipeptides.z01": (1073741824, "9b6c7addbd20e1640321e65385d29b5f73943d9d86dd455f9eab99574f85598a"),
    "minipeptides.z02": (1073741824, "56dcd1f8dc59b2a9e3688d0a090243cc857c77030236071225b94492fab7620e"),
    "minipeptides.z03": (1073741824, "95534a6681fd6dba83426897a06128b7a56aeea2d30fc77248dc7e31aa5fe67c"),
    "minipeptides.z04": (1073741824, "7eddcd1e7e15e468a2c80b086d212c08db14474ce1ebe9c4411884d966c7340b"),
    "minipeptides.z05": (1073741824, "af828ef4c94bcf9dd90e0d03d6ae20d2d0e753dd4c2fedd6780ad96d18b0e0b6"),
    "minipeptides.z06": (1073741824, "dcb31c33939a17d7f1828efbd61489ee8ae740b3f7dd07965e5d321227b5b1f3"),
    "minipeptides.zip": (835667572, "c3d1c7d91c2f325a9aaa293d749b715e5e1e26a7b86e8779855d160717f821e6"),
}


def load_config(path="configs/dipeptides.yaml"):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, path, size, digest):
    if path.exists() and path.stat().st_size == size and _sha256(path) == digest:
        LOG.info("Verified %s", path.name)
        return
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0
                with open(partial, "ab" if offset else "wb") as output:
                    last_log = time.monotonic()
                    while block := response.read(8 << 20):
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_log > 30:
                            LOG.info("Downloading %s: %.1f%%", path.name, 100 * offset / size)
                            last_log = time.monotonic()
            if partial.stat().st_size != size or _sha256(partial) != digest:
                partial.unlink(missing_ok=True)
                raise ValueError(f"Release checksum/size mismatch for {path.name}")
            os.replace(partial, path)
            return
        except Exception:
            LOG.exception("Download attempt %s/5 for %s failed", attempt + 1, path.name)
            if attempt == 4:
                raise
            time.sleep(2)


def _raw_directory(root):
    candidates = list((root / "raw").rglob("train.npy")) if (root / "raw").exists() else []
    candidates = [p.parent for p in candidates if (p.parent / "val.npy").exists() and (p.parent / "test.npy").exists() and (p.parent / "pdbs").is_dir()]
    if len(candidates) > 1:
        raise ValueError(f"Multiple raw datasets found: {candidates}")
    return candidates[0] if candidates else None


def _align_and_features(sequence, values, pdb):
    import mdtraj as md
    topology = md.load_pdb(str(pdb)).topology
    kept = topology.select("name N or name CA or name CB or name C or name O")
    top = topology.subset(kept)
    if len(kept) != 10 or top.n_residues != 2:
        raise ValueError(f"{sequence}: expected 10 atoms in two residues")
    selected = np.asarray(values, dtype=np.float32).reshape(len(values), -1, 3)[:, kept]
    # Source aligns all selected atoms to the first frame, then puts atoms in canonical order.
    selected -= selected.mean(axis=1, keepdims=True)
    reference = selected[0].copy()
    for start in range(0, len(selected), 8192):
        batch = selected[start:start + 8192]
        u, _, vt = np.linalg.svd(np.einsum("bni,nj->bij", batch, reference))
        wrong = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)) < 0
        vt[wrong, -1] *= -1
        selected[start:start + len(batch)] = batch @ (u @ vt)
    atoms = list(top.atoms)
    features = np.array([[AMINO_ACIDS.index(a.residue.name), ATOMS.index(a.name), ["C", "N", "O"].index(a.element.symbol), a.residue.index] for a in atoms], dtype=np.int64)
    order = [next(i for i, atom in enumerate(atoms) if atom.residue.index == residue and atom.name == name) for residue in range(2) for name in ATOMS]
    return selected[:, order].reshape(-1, 30), features[order]


def prepare_data(config, download=True):
    """Verify/extract the official split archive and cache aligned float32 nm coordinates."""
    root = Path(config["data"]["root"])
    cache = root / "processed"
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") == 1 and all((cache / split / f"{seq}.npy").exists() for split, items in manifest["splits"].items() for seq in items):
            return manifest
    raw = _raw_directory(root)
    if raw is None:
        if not download:
            raise FileNotFoundError("Dataset absent; run python train.py --prepare-data first")
        downloads = root / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        release = config["data"].get("release", "https://github.com/noegroup/ScoreMD/releases/download/1.0.0").rstrip("/")
        for name, (size, digest) in ASSETS.items():
            _download(f"{release}/{name}", downloads / name, size, digest)
        # Info-ZIP 3.0 silently truncates large split members here; 7-Zip
        # reads the original volumes directly and validates CRC during extraction.
        seven_zip = shutil.which("7z") or str(Path(sys.executable).with_name("7z"))
        subprocess.run([seven_zip, "x", "-y", "-bsp0", str(downloads / "minipeptides.zip"), f"-o{root / 'raw'}"], check=True)
        _atomic_json(downloads / "verified.json", {"release": release, "assets": ASSETS, "crc_verified": True})
        raw = _raw_directory(root)
        if raw is None:
            raise FileNotFoundError("Extracted archive lacks train/val/test.npy and pdbs")
    manifest = {"version": 1, "units": "nm", "source": str(raw), "splits": {}, "norm_factor": None}
    moments = np.zeros(3, dtype=np.float64)  # count, sum, sum of squares; train only.
    for split in ("train", "val", "test"):
        output = cache / split
        output.mkdir(parents=True, exist_ok=True)
        values = np.load(raw / f"{split}.npy", allow_pickle=True).item()
        items = {}
        for sequence, frames in values.items():
            if "G" in sequence:
                continue
            path = output / f"{sequence}.npy"
            feature_path = output / f"{sequence}.features.npy"
            if path.exists() and feature_path.exists():
                x, features = np.load(path, mmap_mode="r"), np.load(feature_path)
            else:
                x, features = _align_and_features(sequence, frames, raw / "pdbs" / f"{sequence}.pdb")
                np.save(path.with_suffix(".tmp.npy"), x)
                os.replace(path.with_suffix(".tmp.npy"), path)
                np.save(feature_path.with_suffix(".tmp.npy"), features)
                os.replace(feature_path.with_suffix(".tmp.npy"), feature_path)
            if x.ndim != 2 or x.shape[1] != 30 or not np.isfinite(x).all():
                raise ValueError(f"Invalid processed data for {split}/{sequence}")
            if split == "train":
                for start in range(0, len(x), 8192):
                    block = np.asarray(x[start:start + 8192], dtype=np.float64)
                    moments += (block.size, block.sum(), np.square(block).sum())
            items[sequence] = {"samples": len(x), "features": features.tolist()}
            LOG.info("Prepared %s/%s: %d frames", split, sequence, len(x))
        manifest["splits"][split] = items
        del values
    sequence_sets = [set(manifest["splits"][s]) for s in ("train", "val", "test")]
    if any(sequence_sets[i] & sequence_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Peptide identity leakage across train/validation/test")
    expected = config["data"].get("expected_split_sizes", {"train": 175, "val": 75, "test": 92})
    for split, count in expected.items():
        if len(manifest["splits"][split]) != count:
            raise ValueError(f"{split}: expected {count} nonglycine peptides, found {len(manifest['splits'][split])}")
    variance = moments[2] / moments[0] - (moments[1] / moments[0]) ** 2
    manifest["norm_factor"] = float(1 / np.sqrt(variance))
    _atomic_json(manifest_path, manifest)
    LOG.info("Data ready. Training normalization factor %.8g", manifest["norm_factor"])
    return manifest


def load_split(config, split):
    """Return peptide keyed, memory-mapped aligned coordinates and one feature table."""
    root = Path(config["data"]["root"]) / "processed"
    manifest = json.loads((root / "manifest.json").read_text())
    return {sequence: {"x": np.load(root / split / f"{sequence}.npy", mmap_mode="r"), "features": np.asarray(info["features"], dtype=np.int64)} for sequence, info in manifest["splits"][split].items()}


class Peptides:
    def __init__(self, data):
        self.items = list(data.values())
        self.ends = np.cumsum([len(item["x"]) for item in self.items])
        self.starts = np.r_[0, self.ends[:-1]]
        self.length = int(self.ends[-1])

    def batch(self, indices, device):
        owners = np.searchsorted(self.ends, indices, side="right")
        x = torch.empty((len(indices), 30), pin_memory=device.type == "cuda")
        features = torch.empty((len(indices), 10, 4), dtype=torch.long, pin_memory=device.type == "cuda")
        x_np, f_np = x.numpy(), features.numpy()
        for owner in np.unique(owners):
            rows = np.flatnonzero(owners == owner)
            x_np[rows] = self.items[owner]["x"][indices[rows] - self.starts[owner]]
            f_np[rows] = self.items[owner]["features"]
        return x.to(device, non_blocking=True), features.to(device, non_blocking=True)


def random_rotations(x):
    q = torch.randn((len(x), 4), device=x.device, dtype=x.dtype)
    w, a, b, c = (q / q.norm(dim=1, keepdim=True)).unbind(dim=1)
    r = torch.stack([1 - 2*b*b - 2*c*c, 2*a*b - 2*w*c, 2*a*c + 2*w*b, 2*a*b + 2*w*c, 1 - 2*a*a - 2*c*c, 2*b*c - 2*w*a, 2*a*c - 2*w*b, 2*b*c + 2*w*a, 1 - 2*a*a - 2*b*b], dim=1).reshape(-1, 3, 3)
    xyz = x.reshape(-1, 10, 3)
    offset = xyz.mean(dim=1, keepdim=True)
    return (torch.einsum("bij,bnj->bni", r, xyz - offset) + offset).reshape_as(x)


def time_weights(t, stage, mixture, config):
    if mixture:
        lo, hi = [(0.6, 1.), (0.1, 0.6), (0., 0.1)][stage]
        mid = config["training"]["mixture_midpoints"][stage]
    else:
        lo, hi, mid = 0., 1., config["training"]["time_midpoint"]
    return torch.where(t.reshape(-1) <= mid, (hi-lo)/(mid-lo)/2, (hi-lo)/(hi-mid)/2)


def losses(model, x, features, t, config, stage=0, noise=None, vectors=None, training=True):
    """DSM and source residual FP; fixed noise/vectors support numerical reference checks."""
    cfg = config["training"]
    active = ([0], [0, 1], [1, 2])[stage] if model.method == "mixture" else None
    log_mean = -0.5 * t * 0.1 - 0.25 * t.square() * (20.0 - 0.1)
    std = (1 - torch.exp(2 * log_mean)).sqrt()
    noise = torch.randn_like(x) if noise is None else noise
    perturbed = torch.exp(log_mean) * x + std * noise
    weights = time_weights(t, stage, model.method == "mixture", config)
    scalar_fp = x.new_zeros(())
    fp_weight = cfg["fp_weight"] if model.method == "fokker_planck" else 0.
    if fp_weight:
        sigma, hs, hd = cfg["sigma"], cfg["fd_hs"], cfg["fd_hd"]
        v = sigma * torch.randn((2, *x.shape), device=x.device) if vectors is None else vectors
        score = model(perturbed, features, t, create_graph=training, active_experts=active)
        residuals = []
        beta = torch.sqrt(0.1 + t.reshape(-1) * (20.0 - 0.1)).square()
        for perturbation in v:
            xp, xm = perturbed + perturbation, perturbed - perturbation
            qc, sp = model.log_q_and_score(xp, features, t, create_graph=training, active_experts=active)
            qc = qc.reshape(-1)
            sm = model(xm, features, t, create_graph=training, active_experts=active)
            qp = model.log_q(xp, features, t + hd, active_experts=active).reshape(-1)
            qm = model.log_q(xp, features, t - hs, active_experts=active).reshape(-1)
            dlogdt = (hs*hs*qp + (hd*hd-hs*hs)*qc - hd*hd*qm) / (hs*hd*(hd+hs))
            trace = (0.5 * beta[:, None] * (perturbation / sigma) * (sp-sm) / (2*sigma)).sum(dim=1)
            # Source divergence is -beta/2 rather than -D*beta/2. Preserve its +1.
            gamma = (0.5*beta*torch.linalg.vector_norm(sp.abs()+1e-7, dim=1).square()
                     + 0.5*beta*(xp*sp).sum(dim=1) + 0.5*beta - dlogdt)
            residuals.append(trace + gamma)
        scalar_fp = ((residuals[0] * residuals[1]) * weights).mean() / x.shape[1]**2 * fp_weight
    else:
        score = model(perturbed, features, t, create_graph=training, active_experts=active)
    dsm = ((noise + std * score).abs() + 1e-7).square().mean(dim=1)
    dsm = (dsm * weights).mean()
    return torch.stack((dsm, x.new_zeros(()), scalar_fp))


def _rng_state(rng, device=None):
    cuda = torch.cuda.get_rng_state(device) if device is not None and device.type == "cuda" else None
    return {"torch": torch.get_rng_state(), "cuda": cuda, "numpy": rng.bit_generator.state, "python": random.getstate()}


def _restore_rng(state, rng, device=None):
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if isinstance(cuda, list):  # Older local checkpoints saved every device.
        index = (device.index or 0) if device is not None else 0
        cuda = cuda[index if index < len(cuda) else 0] if cuda else None
    if cuda is not None and device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda.cpu(), device)
    rng.bit_generator.state = state["numpy"]
    random.setstate(state["python"])


def _save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _optimizer_step(model, optimizer, scheduler, ema, cfg):
    old = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    # Optax supplies zero arrays for unused parameters, so AdamW still decays them.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all():
        raise FloatingPointError("Nonfinite parameter gradient; checkpoint retained for diagnosis")
    torch._foreach_clamp_min_(gradients, -cfg["clip"])
    torch._foreach_clamp_max_(gradients, cfg["clip"])
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name in old:
                changed = (value - old[name]).abs().amax() > 1e-8
                ema[name].copy_(torch.where(changed, cfg["ema_weight"] * value + (1-cfg["ema_weight"]) * ema[name], ema[name]))


def train(config, args):
    from model import DipeptideModel
    cfg = config["training"]
    enabled = cfg.get("methods", ["diffusion", "mixture", "fokker_planck"])
    if args.method not in enabled:
        raise ValueError(f"Training method {args.method!r} is disabled by training.methods; enabled methods: {enabled}")
    seed = cfg["seed"]
    device = torch.device(args.device)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Match JAX float32 matmuls, retaining float32 rather than mixed precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = prepare_data(config, download=False)
    norm = manifest["norm_factor"]
    training = Peptides(load_split(config, "train"))
    validation = Peptides(load_split(config, "val"))
    model = DipeptideModel(args.method, config).to(device)
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stages = cfg["mixture_epochs"] if args.method == "mixture" else [cfg["epochs"]]
    batch_size = cfg["batch_size"]
    microbatch = args.microbatch or min(128 if args.method == "fokker_planck" else 256, batch_size)
    if microbatch < 1 or microbatch > batch_size:
        raise ValueError("microbatch must be between 1 and the effective batch size")
    steps_per_epoch = training.length // batch_size
    if not steps_per_epoch:
        raise ValueError("Training data smaller than one effective batch")
    run = Path(config["data"]["root"]) / "runs" / args.method / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    resume = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    start_epoch = resume["epoch"] if resume else 0
    global_step = resume.get("global_step", 0) if resume else 0
    start_step = resume.get("step_in_epoch", 0) if resume else 0
    permutation = resume.get("permutation") if resume else None
    if resume:
        if resume["method"] != args.method or any(resume["config"][key] != config[key] for key in ("model", "training")):
            raise ValueError("Resume method/model/training settings differ from checkpoint")
        if not np.isclose(resume["norm_factor"], norm, rtol=1e-8, atol=0):
            raise ValueError("Resume data normalization differs from checkpoint")
        model.load_state_dict(resume["model"])
        ema = {key: value.to(device) for key, value in resume["ema"].items()}
        _restore_rng(resume["rng"], rng, device)
    if resume and start_epoch >= sum(stages) and start_step == 0:
        if not args.profile_only:
            resume["completed"] = True
            resume["config"] = config
            _save(run / "final.pt", resume)
        LOG.info("Training schedule already complete: %s", run / "final.pt")
        return
    if getattr(args, "compile", False):
        model.enable_compile()
    LOG.info("%s: %d parameters; %d frames, %d steps/epoch, %d epochs, microbatch %d", args.method, sum(p.numel() for p in model.parameters()), training.length, steps_per_epoch, sum(stages), microbatch)
    epoch = start_epoch
    stage_start = 0
    total_started, step_seconds = time.monotonic(), []
    for stage, num_epochs in enumerate(stages):
        stage_end = stage_start + num_epochs
        if epoch >= stage_end:
            stage_start = stage_end
            continue
        if args.method == "mixture":
            for index, expert in enumerate(model.experts):
                expert.requires_grad_(index == stage)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], foreach=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * training.length // batch_size, eta_min=cfg["min_learning_rate"])
        if resume and resume["stage"] == stage:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
        def checkpoint(epoch_value, step_value, current_permutation, completed=False):
            return {"model": model.state_dict(), "ema": ema, "norm_factor": norm, "method": args.method, "config": config, "epoch": epoch_value, "stage": stage, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": _rng_state(rng, device), "global_step": global_step, "step_in_epoch": step_value, "permutation": current_permutation, "completed": completed}
        while epoch < stage_end:
            model.train()
            if permutation is None:
                permutation = rng.permutation(training.length)[:steps_per_epoch * batch_size]
            losses_sum = np.zeros(3)
            epoch_started = time.monotonic()
            for step in range(start_step, steps_per_epoch):
                began = time.monotonic()
                x, features = training.batch(permutation[step*batch_size:(step+1)*batch_size], device)
                x = random_rotations(x) * norm
                lo, hi = [(0.6, 1.), (0.099, 0.6), (1e-5, 0.099)][stage] if args.method == "mixture" else (1e-5, 1.)
                t = torch.rand((batch_size, 1), device=device) * (hi-lo) + lo
                noise = torch.randn_like(x)
                vectors = cfg["sigma"] * torch.randn((2, *x.shape), device=device) if args.method == "fokker_planck" else None
                optimizer.zero_grad(set_to_none=True)
                batch_losses = torch.zeros(3, device=device)
                try:
                    for begin in range(0, batch_size, microbatch):
                        end = min(begin+microbatch, batch_size)
                        terms = losses(model, x[begin:end], features[begin:end], t[begin:end], config, stage, noise[begin:end], None if vectors is None else vectors[:, begin:end])
                        if not torch.isfinite(terms).all():
                            raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1} step {step+1}: {terms.detach()}")
                        weighted = terms * ((end-begin)/batch_size)
                        if weighted.requires_grad:
                            weighted.sum().backward()
                        batch_losses += weighted.detach()
                    _optimizer_step(model, optimizer, scheduler, ema, cfg)
                except Exception:
                    _save(run / "failure.pt", checkpoint(epoch, step, permutation))
                    raise
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic()-began
                step_seconds.append(elapsed)
                losses_sum += batch_losses.cpu().numpy()
                global_step += 1
                if step % 50 == 0 or step == steps_per_epoch-1:
                    recent = np.mean(step_seconds[-50:])
                    remaining = (sum(stages)-epoch-1)*steps_per_epoch + steps_per_epoch-step-1
                    LOG.info("epoch %d/%d stage %d step %d/%d loss=%s lr=%.4g %.3fs/step ETA %.1fh", epoch+1, sum(stages), stage+1, step+1, steps_per_epoch, np.round(batch_losses.cpu().numpy(), 6), optimizer.param_groups[0]["lr"], recent, remaining*recent/3600)
                if args.profile_only and len(step_seconds) >= 3:
                    report = {"method": args.method, "microbatch": microbatch, "batch_size": batch_size, "seconds_per_step": float(np.mean(step_seconds[1:])), "estimated_training_hours": float(np.mean(step_seconds[1:])*steps_per_epoch*sum(stages)/3600), "peak_gpu_gb": torch.cuda.max_memory_allocated(device)/1e9 if device.type == "cuda" else None}
                    _atomic_json(run / "profile.json", report)
                    LOG.info("Profile: %s", report)
                    return
                if args.max_steps and global_step >= args.max_steps:
                    _save(run / "latest.pt", checkpoint(epoch, step+1, permutation))
                    LOG.info("Stopped after requested %d optimizer steps; resumable checkpoint saved", args.max_steps)
                    return
                if (step+1) % cfg.get("checkpoint_steps", 100) == 0:
                    _save(run / "latest.pt", checkpoint(epoch, step+1, permutation))
            # Source SEMA: replace train parameters at every epoch, retain Adam moments.
            model.load_state_dict(ema)
            epoch += 1
            start_step, permutation = 0, None
            validation_terms = validate(model, validation, config, stage, norm, device, rng, microbatch)
            row = {"epoch": epoch, "stage": stage, "train": (losses_sum/(steps_per_epoch - (resume.get('step_in_epoch', 0) if resume and epoch == start_epoch+1 else 0))).tolist(), "validation": validation_terms.tolist(), "seconds": time.monotonic()-epoch_started}
            with open(run / "metrics.jsonl", "a") as stream:
                stream.write(json.dumps(row)+"\n")
            _save(run / "latest.pt", checkpoint(epoch, 0, None))
            if epoch == stage_end:
                _save(run / f"stage{stage+1}.pt", checkpoint(epoch, 0, None))
            LOG.info("Completed epoch %d: %s", epoch, row)
            resume = None
        stage_start = stage_end
    _save(run / "final.pt", checkpoint(epoch, 0, None, completed=True))
    LOG.info("Full training completed in %.2f hours: %s", (time.monotonic()-total_started)/3600, run / "final.pt")


def validate(model, dataset, config, stage, norm, device, rng, microbatch):
    model.eval()
    batch_size = config["training"]["batch_size"]
    permutation = rng.permutation(dataset.length)[:dataset.length//batch_size*batch_size]
    total = np.zeros(3)
    for indices in permutation.reshape(-1, batch_size):
        x, features = dataset.batch(indices, device)
        x = random_rotations(x)*norm
        lo, hi = [(0.6, 1.), (0.099, 0.6), (1e-5, 0.099)][stage] if model.method == "mixture" else (1e-5, 1.)
        t = torch.rand((len(x), 1), device=device)*(hi-lo)+lo
        for begin in range(0, len(x), microbatch):
            end = min(begin+microbatch, len(x))
            # Conservative models require local coordinate autograd even in evaluation.
            with torch.no_grad():
                terms = losses(model, x[begin:end], features[begin:end], t[begin:end], config, stage, training=False)
            if not torch.isfinite(terms).all():
                raise FloatingPointError("Nonfinite validation loss")
            total += terms.detach().cpu().numpy()*((end-begin)/batch_size)
    return total/(len(permutation)//batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dipeptides.yaml")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--method", choices=["diffusion", "mixture", "fokker_planck"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile unchanged expert calculations with TorchInductor")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    config = load_config(args.config)
    if args.prepare_data:
        prepare_data(config)
    elif args.method:
        train(config, args)
    else:
        parser.error("Specify --prepare-data or --method")


if __name__ == "__main__":
    main()
