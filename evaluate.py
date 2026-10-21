"""Independent generation, resumable Langevin simulation, and dipeptide metrics."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from model import DipeptideModel
from train import load_config, load_split

METHODS = ("diffusion", "two_for_one", "mixture", "fokker_planck")
# Table of complete dipeptide metrics, arXiv:2506.17139v3, section C.8.1.
PAPER = {
    "diffusion": ([.0155, .2256, .206, 6.515], [.0083, .1304, .159, 3.175]),
    "two_for_one": ([.0153, .0466, .203, .741], [.0080, .0114, .149, .319]),
    "mixture": ([.0155, .0444, .200, .658], [.0078, .0237, .127, .407]),
    "fokker_planck": ([.0200, .0254, .241, .368], [.0071, .0119, .105, .267]),
}
PAPER_METRICS = ("iid_jsd", "simulation_jsd", "iid_pmf", "simulation_pmf")
PROTOCOL_KEYS = ("peptides", "seed", "simulation_steps", "save_every", "replicas", "eval_time",
                 "two_for_one_time", "dt", "gamma", "temperature")
MASS = np.repeat([14.00672, 12.01078, 12.01078, 12.01078, 15.99943] * 2, 3)
R_GAS = .00831446261815324  # kJ mol^-1 K^-1


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def write_json(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def angles(x):
    """Source dihedral convention and the released ten-bead phi/psi indices."""
    x = np.asarray(x).reshape(-1, 10, 3)
    values = []
    for indices in ([3, 5, 6, 8], [0, 1, 3, 5]):
        p = x[:, indices]
        b = p[:, :-1] - p[:, 1:]
        b[:, 0] *= -1
        with np.errstate(invalid="ignore", divide="ignore"):
            b1 = b[:, 1]
            v = b[:, [0, 2]] - ((b[:, [0, 2]] * b1[:, None]).sum(-1)
                                   / (b1 * b1).sum(-1)[:, None])[..., None] * b1[:, None]
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
            values.append(np.arctan2((np.cross(v[:, 0], b1) * v[:, 1]).sum(-1),
                                    (v[:, 0] * v[:, 1]).sum(-1)))
    return np.stack(values, axis=-1)


def metrics(truth, samples, bins=64):
    """Report paper definitions first, then the released evaluator's definitions.

    Paper JSD uses add-one counts on the fixed angular domain. Paper PMF
    uniformly averages log-proportion errors on all bins, with an additive
    1e-6 baseline. Companion PMF uses 62 interior bins, fills only missing bins,
    and averages with normalized mixture-probability weights, as released.
    """
    truth, samples = np.asarray(truth), np.asarray(samples)
    truth = truth[np.isfinite(truth).all(axis=1)]
    samples = samples[np.isfinite(samples).all(axis=1)]
    if not len(truth) or not len(samples):
        return {key: None for key in ("jsd", "js_distance", "pmf", "code_jsd", "code_pmf", "code_mjs")}
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    p = np.histogram2d(*truth.T, bins=(edges, edges))[0].ravel()
    q = np.histogram2d(*samples.T, bins=(edges, edges))[0].ravel()
    ps, qs = (p + 1) / (p.sum() + bins**2), (q + 1) / (q.sum() + bins**2)
    divergence = float(jensenshannon(ps, qs)**2)
    pmf = np.square(np.log(p / p.sum() + 1e-6) - np.log(q / q.sum() + 1e-6)).mean()
    both = np.concatenate((truth, samples))
    code_edges = [np.linspace(both[:, i].min(), both[:, i].max(), bins + 1) for i in range(2)]
    pc = np.histogram2d(*truth.T, bins=code_edges)[0].ravel()
    qc = np.histogram2d(*samples.T, bins=code_edges)[0].ravel()
    code_jsd = float(jensenshannon(pc, qc)**2)
    inner_edges = np.linspace(-np.pi, np.pi, bins - 1)
    counts = []
    for sample in (truth, samples):
        index = np.digitize(sample, inner_edges)
        counts.append(np.bincount(index[:, 0] * bins + index[:, 1], minlength=bins**2) / len(sample))
    occupied = (counts[0] + counts[1]) > 0
    a, b = [np.where(c[occupied] > 0, c[occupied], 1e-6) for c in counts]
    mixture = (a + b) / 2
    weights = mixture / mixture.sum()
    return {"jsd": divergence, "js_distance": math.sqrt(divergence), "pmf": float(pmf),
            "code_jsd": code_jsd, "code_pmf": float((weights * (np.log(a) - np.log(b))**2).sum()),
            "code_mjs": float((weights * (a / mixture * np.log(a / mixture)
                                          + b / mixture * np.log(b / mixture)) / 2).sum())}


def make_score(model, compile_model=False):
    """Use a static expert to avoid a GPU synchronization in mixture routing."""
    functions = []
    for expert in model.experts:
        if expert.potential:
            gradient = torch.func.grad(lambda x, features, t, expert=expert:
                                       -expert._differentiable_forward(x, features, t).sum())
            def call(x, features, t, gradient=gradient):
                centered = x.reshape(-1, 10, 3)
                centered = (centered - centered.mean(dim=1, keepdim=True)).reshape_as(x)
                return gradient(centered, features, t)
        else:
            def call(x, features, t, expert=expert):
                centered = x.reshape(-1, 10, 3)
                centered = (centered - centered.mean(dim=1, keepdim=True)).reshape_as(x)
                return -expert._differentiable_forward(centered, features, t).reshape_as(x)
        functions.append(torch.compile(call, fullgraph=True, dynamic=False) if compile_model else call)

    def score(x, features, t, scalar_time):
        active = [i for i, (hi, lo) in enumerate(model.ranges) if lo <= scalar_time <= hi]
        return sum(functions[i](x, features, t) for i in active) / len(active)
    return score


def load_model(config, method, device, checkpoint=None):
    trained_method = "diffusion" if method == "two_for_one" else method
    root = Path(config["data"].get("root", "data"))
    checkpoint = Path(checkpoint) if checkpoint else root / "runs" / trained_method / f"seed{config['training']['seed']}" / "final.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("method", trained_method) != trained_method:
        raise ValueError(f"Expected {trained_method} checkpoint, got {state.get('method')} in {checkpoint}")
    model = DipeptideModel(trained_method, state.get("config", config)).to(device)
    model.load_state_dict(state.get("ema", state["model"]))
    model.eval().requires_grad_(False)
    return model, float(state["norm_factor"]), checkpoint, file_hash(checkpoint)


def cache_inference_checked(model, split, config, norm, device):
    sequence = config["evaluation"]["peptides"][0]
    x = torch.tensor(np.asarray(split[sequence]["x"][:2]), device=device) * norm
    features = torch.as_tensor(split[sequence]["features"], device=device).unsqueeze(0).expand(len(x), -1, -1)
    times = (.03, .35, .8) if model.method == "mixture" else (.02,)
    references = [model(x, features, torch.full((len(x), 1), t, device=device)).detach() for t in times]
    model.enable_inference_cache()
    for scalar_time, reference in zip(times, references):
        actual = model(x, features, torch.full((len(x), 1), scalar_time, device=device))
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)


def reverse_diffusion(score, features, n, steps, generator, norm_factor):
    """Released VP Euler-Maruyama: inclusive time grid, noise also at t=0."""
    device = features.device
    x = torch.randn(n, 30, device=device, generator=generator)
    dt = torch.tensor(1 / (steps - 1), dtype=x.dtype, device=device)
    for scalar_time in np.linspace(0, 1, steps, dtype=np.float32)[::-1]:
        t = torch.full((n, 1), float(scalar_time), device=device)
        beta = .1 + t * (20 - .1)
        force = score(x, features, t, float(scalar_time))
        diffusion = torch.sqrt(beta)
        drift = .5 * beta * x + diffusion.square() * force
        x = (x + drift * dt + diffusion * torch.sqrt(dt)
             * torch.randn(x.shape, device=device, generator=generator)).detach()
    return x / norm_factor


def generate_iid(config, split, model, norm, checkpoint_hash, directory, device, compiled=False):
    settings = config["evaluation"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint": checkpoint_hash, "norm_factor": norm, "samples": settings["num_iid_samples"],
                "steps": settings["diffusion_steps"], "seed": settings["seed"], "batch": settings["inference_batch"],
                "compiled": compiled, "inference_cache": model.experts[0]._inference_cache}
    metadata = directory / "iid_metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError(f"IID settings/checkpoint differ from {metadata}; use a fresh output directory")
    write_json(identity, metadata)
    score = make_score(model, compiled)
    for sequence in settings["peptides"]:
        count, batch = settings["num_iid_samples"], settings["inference_batch"]
        path, state_path = directory / f"{sequence}_iid.npy", directory / f"{sequence}_iid_state.pt"
        state = torch.load(state_path, weights_only=False) if state_path.exists() else None
        if state and state.get("identity") != identity:
            raise ValueError(f"IID checkpoint/settings differ from saved state: {state_path}")
        done = state["completed"] if state else 0
        if done == count:
            continue
        array = np.lib.format.open_memmap(path, mode="r+" if state else "w+", dtype="float32", shape=(count, 30))
        generator = torch.Generator(device=device).manual_seed(settings["seed"])
        if state:
            generator.set_state(state["rng"])
        feature = torch.as_tensor(split[sequence]["features"], dtype=torch.long, device=device)
        start_time = time.monotonic()
        while done < count:
            n = min(batch, count - done)
            features = feature.unsqueeze(0).expand(n, -1, -1)
            x = reverse_diffusion(score, features, n, settings["diffusion_steps"], generator, norm)
            array[done:done + n] = x.cpu().numpy()
            done += n
            array.flush()
            atomic_save({"completed": done, "rng": generator.get_state(), "identity": identity}, state_path)
            print(f"IID {sequence}: {done}/{count}, {(time.monotonic() - start_time):.1f} s this session", flush=True)


def langevin_step(x, v, noise, score, features, t, scalar_time, norm, mass, kbt, alpha, force_scale, dt):
    force = kbt * norm * score(x * norm, features, t, scalar_time)
    v_new = alpha * v + force_scale * force / mass + torch.sqrt(kbt * (1 - alpha**2) / mass) * noise
    return (x + dt * v_new).detach(), v_new.detach()


def capture_step(step, x, v, generator):
    """Capture deterministic arithmetic; keep noise outside for exact RNG resume."""
    noise = torch.zeros_like(x)
    static_x, static_v = x.clone(), v.clone()
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            a, b = step(static_x, static_v, noise)
            static_x.copy_(a)
            static_v.copy_(b)
    torch.cuda.current_stream(x.device).wait_stream(stream)
    static_x.copy_(x)
    static_v.copy_(v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(x.device), torch.cuda.graph(graph):
        a, b = step(static_x, static_v, noise)
        static_x.copy_(a)
        static_v.copy_(b)
    # Capturing executes a step; restore the initial state before replay.
    static_x.copy_(x)
    static_v.copy_(v)
    noise.normal_(generator=generator)
    expected_x, expected_v = step(x, v, noise)
    graph.replay()
    torch.cuda.synchronize(x.device)
    torch.testing.assert_close(static_x, expected_x, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(static_v, expected_v, rtol=2e-5, atol=2e-6)
    static_x.copy_(x)
    static_v.copy_(v)
    def replay():
        noise.normal_(generator=generator)
        graph.replay()
        return static_x, static_v
    return static_x, static_v, replay


def simulation_setup(config, split, model, norm, device, compiled):
    settings = config["evaluation"]
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    starting, features = [], []
    for sequence in settings["peptides"]:
        array = split[sequence]["x"]
        indices = np.random.default_rng(settings["seed"]).permutation(len(array))[:settings["replicas"]]
        if len(indices) < settings["replicas"]:
            raise ValueError(f"Not enough starting frames for {sequence}")
        starting.append(np.asarray(array[indices]).reshape(len(indices), 30))
        features.append(np.broadcast_to(split[sequence]["features"], (len(indices), 10, 4)).copy())
    x = torch.as_tensor(np.concatenate(starting), dtype=torch.float32, device=device)
    features = torch.as_tensor(np.concatenate(features), dtype=torch.long, device=device)
    mass = torch.as_tensor(MASS, dtype=torch.float32, device=device)
    kbt = R_GAS * settings["temperature"]
    v = torch.sqrt(kbt / mass) * torch.randn(x.shape, device=device, generator=generator)
    scalar_time = settings["two_for_one_time"] if model.method == "two_for_one" else settings["eval_time"]
    t = torch.full((len(x), 1), scalar_time, device=device)
    score = make_score(model, compiled)
    with torch.enable_grad():
        reference = model(x * norm, features, t, create_graph=False).detach()
    torch.testing.assert_close(score(x * norm, features, t, scalar_time), reference, rtol=2e-5, atol=2e-5)
    alpha = torch.exp(torch.tensor(-settings["gamma"] * settings["dt"], dtype=x.dtype, device=device))
    def step(x, v, noise):
        return langevin_step(x, v, noise, score, features, t, scalar_time, norm, mass, kbt,
                             alpha, (1 - alpha) / settings["gamma"], settings["dt"])
    return x, v, generator, step


def simulate(config, split, model, norm, checkpoint_hash, directory, device, compiled=False,
             cuda_graphs=False, profile_steps=0):
    settings = config["evaluation"]
    if settings["simulation_steps"] % settings["save_every"]:
        raise ValueError("simulation_steps must be divisible by save_every")
    nframes = settings["simulation_steps"] // settings["save_every"]
    identity = {"checkpoint": checkpoint_hash, "norm_factor": norm,
                **{key: settings[key] for key in PROTOCOL_KEYS}, "method": model.method,
                "compiled": compiled, "cuda_graphs": cuda_graphs, "inference_cache": model.experts[0]._inference_cache}
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "simulation_metadata.json"
    if not profile_steps and metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError(f"Simulation settings/checkpoint differ from {metadata}; use a fresh output directory")
    x, v, generator, step = simulation_setup(config, split, model, norm, device, compiled)
    state_path = directory / "simulation_state.pt"
    state = torch.load(state_path, weights_only=False) if state_path.exists() and not profile_steps else None
    if state and state.get("identity") != identity:
        raise ValueError(f"Simulation checkpoint/settings differ from saved state: {state_path}")
    completed = state["completed"] if state else 0
    alive = state["alive"] if state else np.ones(len(x), dtype=bool)
    if state:
        x.copy_(state["x"].to(device))
        v.copy_(state["v"].to(device))
        generator.set_state(state["rng"])
    if completed == settings["simulation_steps"] and not profile_steps:
        return
    if cuda_graphs:
        if device.type != "cuda":
            raise ValueError("--cuda-graphs requires a CUDA device")
        # capture_step consumes one test noise draw; restoring makes its setup RNG-neutral.
        rng = generator.get_state()
        x, v, replay = capture_step(step, x, v, generator)
        generator.set_state(rng)
    else:
        def replay():
            nonlocal x, v
            noise = torch.randn(x.shape, device=device, generator=generator)
            x, v = step(x, v, noise)
            return x, v
    if profile_steps:
        for _ in range(5):
            replay()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.monotonic()
        for _ in range(profile_steps):
            replay()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - start
        profile = {"steps": profile_steps, "seconds": seconds, "paths": len(x), "compile": compiled,
                   "cuda_graphs": cuda_graphs, "inference_cache": model.experts[0]._inference_cache,
                   "steps_per_second": profile_steps / seconds,
                   "full_simulation_hours": settings["simulation_steps"] * seconds / profile_steps / 3600}
        write_json(profile, directory / "profile.json")
        print(json.dumps(profile, indent=2), flush=True)
        return
    write_json(identity, metadata)
    arrays = [np.lib.format.open_memmap(directory / f"{sequence}_simulation.npy", mode="r+" if state else "w+",
                                      dtype="float32", shape=(settings["replicas"], nframes, 30))
              for sequence in settings["peptides"]]
    chunk = settings["chunk_frames"]
    start = time.monotonic()
    while completed < settings["simulation_steps"]:
        begin_frame = completed // settings["save_every"]
        frames = min(chunk, nframes - begin_frame)
        positions = torch.empty((len(x), frames, 30), dtype=x.dtype, device=device)
        for frame in range(frames):
            for _ in range(settings["save_every"]):
                x, v = replay()
            positions[:, frame].copy_(x)
            if frame % max(1, 1000 // settings["save_every"]) == 0:
                print(f"Simulation {model.method}: step {completed + (frame + 1) * settings['save_every']}/"
                      f"{settings['simulation_steps']}", flush=True)
        cpu_positions = positions.cpu().numpy()
        alive &= np.isfinite(cpu_positions).all(axis=(1, 2))
        alive &= torch.isfinite(v).all(dim=1).cpu().numpy()
        for i, array in enumerate(arrays):
            array[:, begin_frame:begin_frame + frames] = cpu_positions[i * settings["replicas"]:(i + 1) * settings["replicas"]]
            array.flush()
        completed += frames * settings["save_every"]
        atomic_save({"completed": completed, "x": x.cpu(), "v": v.cpu(), "alive": alive,
                     "rng": generator.get_state(), "identity": identity}, state_path)
        elapsed = time.monotonic() - start
        print(f"Saved {model.method}: {completed}/{settings['simulation_steps']}, "
              f"alive {alive.sum()}/{len(alive)}, {elapsed / 60:.1f} min this session", flush=True)
    write_json({"complete": True, "steps_per_path": completed, "replicas": settings["replicas"],
                "aggregate_ns_per_peptide": completed * settings["dt"] * settings["replicas"] / 1000,
                "alive_paths": alive.tolist()}, directory / "simulation_complete.json")


def plot_results(truth, iid, simulation, sequence, directory, settings):
    kbt = settings["temperature"] * R_GAS
    datasets = {"Reference": truth, "Independent": iid, "Simulation": simulation}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (label, sample) in zip(axes, datasets.items()):
        counts = np.histogram2d(*sample.T, bins=settings["plot_bins"], range=[[-np.pi, np.pi]] * 2)[0]
        pmf = -np.log(counts / max(1, counts.sum()) + 1e-6)
        pmf -= pmf.min()
        image = ax.imshow(pmf.T, extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower", vmin=0, vmax=12,
                          aspect="equal", cmap="magma_r")
        ax.set(xlabel=r"$\varphi$", ylabel=r"$\psi$", title=f"{sequence}: {label}")
    fig.colorbar(image, ax=list(axes), label=r"PMF / $k_BT$")
    fig.savefig(directory / f"{sequence}_pmf_2d.png", dpi=150)
    plt.close(fig)
    grid = np.linspace(-np.pi, np.pi, 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for dim, ax in enumerate(axes):
        for label, sample in datasets.items():
            # Preserve source plot_fes_angles: concatenate periodic images, then
            # plot_fes filters to [-pi, pi] before fitting the KDE.
            extended = np.concatenate([sample[:, dim], sample[:, dim] + 2 * np.pi, sample[:, dim] - 2 * np.pi])
            extended = extended[(extended >= grid.min()) & (extended <= grid.max())]
            if len(extended) < 2 or np.std(extended) < 1e-12:
                continue
            fes = -kbt * gaussian_kde(extended, bw_method=settings["kde_bw"]).logpdf(grid)
            fes -= fes.min()
            ax.plot(grid, fes, label=label)
        ax.set(xlabel=(r"$\varphi$", r"$\psi$")[dim], ylabel="PMF (kJ/mol)", title=sequence, xlim=(-np.pi, np.pi))
        ax.legend()
    fig.savefig(directory / f"{sequence}_pmf_1d.png", dpi=150)
    plt.close(fig)


def evaluate_outputs(config, split, method, directory, iid_directory):
    settings = config["evaluation"]
    state_path = directory / "simulation_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing completed simulation: {state_path}")
    state = torch.load(state_path, weights_only=False)
    if any(state["identity"].get(key) != settings[key] for key in PROTOCOL_KEYS):
        raise ValueError(f"Configured simulation protocol differs from saved state: {state_path}")
    if state["identity"]["method"] != method:
        raise ValueError(f"Saved simulation method differs from {method}: {state_path}")
    if state["completed"] != settings["simulation_steps"]:
        raise ValueError(f"Incomplete simulation at {state['completed']} steps in {directory}")
    rows = []
    for i, sequence in enumerate(settings["peptides"]):
        iid_state = torch.load(iid_directory / f"{sequence}_iid_state.pt", weights_only=False)
        expected_iid = {"samples": settings["num_iid_samples"], "steps": settings["diffusion_steps"], "seed": settings["seed"]}
        if any(iid_state["identity"].get(key) != value for key, value in expected_iid.items()):
            raise ValueError(f"Configured IID protocol differs from saved {sequence} samples")
        if iid_state["identity"]["checkpoint"] != state["identity"]["checkpoint"]:
            raise ValueError(f"Independent and simulation samples use different checkpoints for {sequence}")
        if iid_state["completed"] != settings["num_iid_samples"]:
            raise ValueError(f"Incomplete IID samples for {sequence}")
        truth = angles(split[sequence]["x"])
        iid = angles(np.load(iid_directory / f"{sequence}_iid.npy", mmap_mode="r"))
        trajectories = np.load(directory / f"{sequence}_simulation.npy", mmap_mode="r")
        alive = state["alive"][i * settings["replicas"]:(i + 1) * settings["replicas"]]
        simulation = np.asarray(trajectories[alive]).reshape(-1, 30)
        if len(simulation):
            indices = np.random.default_rng(settings["seed"]).permutation(len(simulation))[:settings["num_iid_samples"]]
            simulation = angles(simulation[indices])
        else:
            simulation = np.empty((0, 2))
        row = {"method": method, "peptide": sequence, "reference_samples": len(truth),
               "iid_valid_samples": int(np.isfinite(iid).all(axis=1).sum()),
               "simulation_valid_samples": int(np.isfinite(simulation).all(axis=1).sum()),
               "simulation_alive_paths": int(alive.sum()), "simulation_total_paths": settings["replicas"]}
        for source, sample in (("iid", iid), ("simulation", simulation)):
            row.update({f"{source}_{key}": value for key, value in metrics(truth, sample, settings["bins"]).items()})
        rows.append(row)
        np.savez(directory / f"{sequence}_angles.npz", reference=truth, iid=iid, simulation=simulation)
        plot_results(truth[np.isfinite(truth).all(1)], iid[np.isfinite(iid).all(1)],
                     simulation[np.isfinite(simulation).all(1)], sequence, directory, settings)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "metrics.csv", index=False)
    write_json(rows, directory / "metrics.json")
    write_json(metrics_identity(settings), directory / "metrics_metadata.json")
    return frame


def metrics_identity(settings):
    return {key: settings[key] for key in PROTOCOL_KEYS + ("num_iid_samples", "diffusion_steps", "bins")}


def write_summary(result_root, frames, settings):
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(result_root / "per_peptide.csv", index=False)
    summary, comparison = {}, []
    metric_names = [c for c in combined.columns if c.startswith(("iid_", "simulation_"))
                    and not c.endswith(("samples", "paths"))]
    for method, group in combined.groupby("method", sort=False):
        summary[method] = {}
        for name in metric_names:
            values = group[name].dropna()
            mean, std = (float(values.mean()), float(values.std(ddof=0))) if len(values) else (None, None)
            summary[method][name] = {"mean": mean, "std_across_peptides": std,
                                   "evaluated_peptides": len(values), "expected_peptides": len(settings["peptides"])}
        for index, name in enumerate(PAPER_METRICS):
            result = summary[method][name]
            for definition in ("paper", "released_code"):
                current = name if definition == "paper" else name.replace("_jsd", "_code_jsd").replace("_pmf", "_code_pmf")
                measured = summary[method][current]
                comparison.append({"method": method, "metric_definition": definition, "metric": name,
                                   "mean": measured["mean"], "std": measured["std_across_peptides"],
                                   "paper_mean": PAPER[method][0][index], "paper_std": PAPER[method][1][index],
                                   "gap": None if measured["mean"] is None else measured["mean"] - PAPER[method][0][index],
                                   "evaluated_peptides": measured["evaluated_peptides"]})
    report = {"source": "https://arxiv.org/pdf/2506.17139v3", "methods": summary,
              "settings": settings, "comparison": comparison,
              "notes": ["The scalar Fokker–Planck coefficient is 0.0005 from the paper; released YAML uses 0.005.",
                        "Paper and released-code metric definitions differ; both are reported on identical samples.",
                        "Standard deviations are across peptides, not independent training seeds.",
                        "Nonfinite simulation paths are excluded as in released code; counts are reported explicitly.",
                        "Two-for-One shares diffusion weights and independent samples."]}
    write_json(report, result_root / "summary.json")
    pd.DataFrame(comparison).to_csv(result_root / "comparison.csv", index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dipeptides.yaml")
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--compile", action="store_true", help="Compile static expert score functions; validates against eager")
    parser.add_argument("--inference-cache", action="store_true", help="Compose frozen adjacent edge projections; validates score equivalence")
    parser.add_argument("--cuda-graphs", action="store_true", help="Capture simulation arithmetic; validates one-step equivalence")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--iid-samples", type=int)
    parser.add_argument("--simulation-steps", type=int)
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--inference-batch", type=int)
    parser.add_argument("--output", help="Override result root")
    args = parser.parse_args()
    if args.checkpoint and args.method == "all":
        parser.error("--checkpoint requires a single --method")
    if args.metrics_only and args.profile_only:
        parser.error("--metrics-only and --profile-only are mutually exclusive")
    config = load_config(args.config)
    settings = config["evaluation"]
    configured_methods = tuple(settings.get("methods", METHODS))
    if not configured_methods or len(set(configured_methods)) != len(configured_methods) or set(configured_methods) - set(METHODS):
        parser.error("evaluation.methods must contain unique supported method names")
    overrides = {"num_iid_samples": args.iid_samples, "simulation_steps": args.simulation_steps, "replicas": args.replicas}
    smoke = any(value is not None and value != settings[key] for key, value in overrides.items())
    settings.update({key: value for key, value in overrides.items() if value is not None})
    if args.inference_batch is not None:
        settings["inference_batch"] = args.inference_batch
    if min(settings["num_iid_samples"], settings["replicas"], settings["inference_batch"]) < 1:
        parser.error("sample counts, replicas and batch size must be positive")
    if settings["diffusion_steps"] < 2 or settings["simulation_steps"] < settings["save_every"]:
        parser.error("at least two diffusion steps and one saved simulation frame are required")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(settings["seed"])
    root = Path(config["data"].get("root", "data")) / "results"
    if smoke:
        tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
        root = root / f"smoke_{tag}"
    result_root = Path(args.output) if args.output else root
    result_root.mkdir(parents=True, exist_ok=True)
    split = load_split(config, "test")
    missing = set(settings["peptides"]) - set(split)
    if missing:
        raise ValueError(f"Missing evaluation peptides from test split: {sorted(missing)}")
    methods = configured_methods if args.method == "all" else (args.method,)
    frames = []
    for method in methods:
        directory = result_root / method
        directory.mkdir(parents=True, exist_ok=True)
        iid_directory = result_root / ("diffusion" if method == "two_for_one" else method)
        if not args.metrics_only:
            model, norm, checkpoint, digest = load_model(config, method, device, args.checkpoint)
            # Keep the same weights while distinguishing Two-for-One's simulation time.
            model.method = method
            print(f"Evaluating {method}: {checkpoint} on {device}", flush=True)
            if args.inference_cache:
                cache_inference_checked(model, split, config, norm, device)
            if not args.profile_only:
                generate_iid(config, split, model, norm, digest, iid_directory, device, args.compile)
            simulate(config, split, model, norm, digest, directory, device, args.compile,
                     args.cuda_graphs, args.profile_steps if args.profile_only else 0)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not args.profile_only:
            frames.append(evaluate_outputs(config, split, method, directory, iid_directory))
    if frames:
        # Include other already completed methods when an individual method finishes.
        by_method = {frame.iloc[0]["method"]: frame for frame in frames}
        for method in configured_methods:
            existing = result_root / method / "metrics.csv"
            if method not in by_method and existing.exists():
                metadata = existing.with_name("metrics_metadata.json")
                if not metadata.exists() or json.loads(metadata.read_text()) != metrics_identity(settings):
                    raise ValueError(f"Incompatible saved metrics: {existing}. Recompute with --method all --metrics-only")
                frame = pd.read_csv(existing)
                if set(frame.peptide) != set(settings["peptides"]) or frame.peptide.duplicated().any():
                    raise ValueError(f"Saved metrics have an incompatible peptide list: {existing}")
                by_method[method] = frame
        write_summary(result_root, list(by_method.values()), settings)


if __name__ == "__main__":
    main()
