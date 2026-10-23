# Single-energy GGF experiment

Configuration: [configs/dipeptides_ggf.yaml](configs/dipeptides_ggf.yaml).

Training is stopped for handoff. See [GGF_HANDOFF.md](GGF_HANDOFF.md) for the
verified checkpoint and explicit resume instructions. Nothing restarts automatically.

## Model and loss

One potential graph transformer retains the Diffusion architecture and peptide
features: width 128, three layers, embeddings 16, eight attention heads of
dimension 64, and 1,398,497 parameters. Its field is the negative coordinate
gradient of scalar energy. The internal time input is fixed at zero, including
compiled/cached inference. There is no mixture or noise conditioning.

For each clean normalized, augmented conformation `y`, draw
`rho = exp(Uniform(log(h), log(20)))` and standard Gaussian noise `eps`.
Train on `z = y + rho * eps` against `-rho * eps / h²`, using raw mean squared
error over samples and coordinates. Each query uses its own clean parent.
There is no KDE, cross-sample target, DSM/FP term, extra loss normalization,
or target projection.

`h` is derived from `ggf.sigma_min = 0.0010004971259352385`; `sigma_max = 20.0`.
The minimum is the numerical floor selected during planning. GGF uses VE;
there is no VP drift in its training or IID sampler.

Other effective settings match Diffusion: seed 1, batch 1,024, 120 epochs,
AdamW learning rate 0.0003 with cosine decay to 0.00001, weight decay 0.0001,
coordinatewise clipping at 1,000, and the existing EMA convention/weight 0.995.
Retained Mixture/FP-only YAML fields are inactive. Production uses microbatch
1,024, float32, and TF32 disabled.

## Evaluation

IID sampling uses ScoreMD's 1,000 inclusive reverse Euler–Maruyama updates,
with final denoising disabled. The VE prior is `Normal(0, 20² I)` and
`sigma(t) = h * (20/h)^t`. Only the IID sampling score is multiplied by
`h²/sigma(t)²`. This is the agreed experimental adapter for an unconditional
field, not a claim that the network is an exact noise-conditioned VE score.

Simulation uses the raw energy field at minimum noise, where that conversion
is one. Physical-force normalization and Langevin dynamics are unchanged:
ten replicas per peptide, six million steps at 0.0005 ps, saving every 50 steps,
temperature 310 K, and friction 1. All 150 paths advance together.

Evaluation uses seed 0, the same 15 test peptides, and 49,000 IID samples per
peptide. Simulation metrics select up to 49,000 frames from surviving paths.
Both metric definitions and population standard deviations are reported.
Missing coverage is explicit; aggregates require all peptides and full valid
sample counts. Empty plots are annotated and raw trajectories are retained.

## Run and status

All GGF code, configuration, and documentation live in `/home/anirudh/ggf_md`.
The earlier separate worktree has been archived and retired.

The configuration uses `/home/anirudh/ggf_md/data/ggf`. This link exposes the
existing run directory, with checkpoints, results, experiment status, validation,
and implementation evidence available under the main folder. The underlying
files remain on the existing NFS storage; no checkpoint bytes are rewritten.

From this checkout, start or adopt the durable pipeline:

```bash
cd /home/anirudh/ggf_md
CUDA_VISIBLE_DEVICES='' \
MPLCONFIGDIR=/home/anirudh/ggf_md/data/ggf/matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/home/anirudh/.local/venvs/ggf_md/bin/python -u scripts/run_ggf_experiment.py \
  --config configs/dipeptides_ggf.yaml --gpus 3,2,1,0 --microbatch 1024 \
  --compile --factorized-attention --inference-cache --cuda-graphs
```

GPU IDs are preferences. Each launch checks current compute contexts and memory;
occupied GPUs are left alone. Training is followed by concurrent IID/simulation
on distinct idle GPUs when available, otherwise sequential execution. Metrics
and baseline comparisons follow automatically. Detached workers publish exact
process identities, logs, and results; controller restarts adopt live work.
Known failures require `--retry-failed`. Missing durable exit evidence is treated
as ambiguous and requires investigation rather than a duplicate launch.

Read status without starting work:

```bash
cd /home/anirudh/ggf_md
/home/anirudh/.local/venvs/ggf_md/bin/python scripts/run_ggf_experiment.py \
  --status --state-dir /home/anirudh/ggf_md/data/ggf/experiment
```

Under `data/ggf/`:

- `runs/ggf/seed1/`: checkpoints and training metrics.
- `results/ggf/`: raw samples/trajectories, states, angles, and plots.
- `results/GGF_RESULTS.md`: comparison with completed baseline measurements.
- `results/baseline_comparison.csv`: differences under both definitions.
- `experiment/`: status, process identities, stage logs, workload pins, and
  the eventual full completion audit.
- `validation/`: CPU checks and isolated CUDA preflight evidence, not full results.

GGF has no published paper reference row. Direct paper-reference fields remain
null; the separate comparison uses our verified completed Diffusion,
Two-for-One, and Mixture measurements.
