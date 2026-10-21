# Transferable dipeptides in PyTorch

Minimal reproduction of the dipeptide generalization experiment in
[Consistent Sampling and Simulation](https://arxiv.org/pdf/2506.17139v3).
Original implementation: [ScoreMD](https://github.com/noegroup/ScoreMD), revision
`149b1ec54d60068c421cee6bce77ecac5cb2a9ef` of this repository.

The code is three files: `model.py` (graph transformer and expert routing),
`train.py` (data preparation and training), and `evaluate.py` (sampling,
simulation, metrics, and plots). Configuration lives in `configs/`; all downloaded
data and generated artifacts live in `data/`. Models and automatic differentiation
use PyTorch only.

## Handoff to another server

**No production training has been launched.** All three training methods are
enabled. The prepared bundle is `data/ggf_md_handoff.tar.gz`; it includes the code,
environment recipe, configuration, validation reports, and the 1.9 GiB processed
dataset. It omits the original download volumes and raw files because training
and evaluation can use the processed cache directly.

Copy the bundle, replacing `USER@SERVER` with your destination:

```bash
scp /home/anirudh/ggf_md/data/ggf_md_handoff.tar.gz{,.sha256} USER@SERVER:~/
```

On that server:

```bash
cd ~
sha256sum -c ggf_md_handoff.tar.gz.sha256
tar -xzf ggf_md_handoff.tar.gz
cd ~/ggf_md
conda env create -f environment.yml
conda activate ggf_md
python train.py --prepare-data  # validates the included cache; no download needed
```

When ready, these commands launch the three independent jobs on GPUs 0, 1, and 2:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --method diffusion --device cuda:0 --compile --microbatch 1024 > data/diffusion.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --method mixture --device cuda:0 --compile --microbatch 1024 > data/mixture.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --method fokker_planck --device cuda:0 --compile --microbatch 512 > data/fokker_planck.log 2>&1 &
```

Choose available GPUs and reduce `--microbatch` if memory is limited. The effective
batch remains 1024. The Conda recipe includes a C++ compiler for compilation;
omit `--compile` to use eager execution. The first compiled step includes compilation time.
To estimate runtime before committing to a full run, use the corresponding
training command with `--profile-only` in the foreground. It measures three
optimizer steps without saving a production checkpoint.

Checkpoints are under `data/runs/METHOD/seed1/`. To resume Diffusion, for example:

```bash
python train.py --method diffusion --device cuda:0 --compile --microbatch 1024 --resume data/runs/diffusion/seed1/latest.pt
```

After all three final checkpoints exist:

```bash
python evaluate.py --method all --device cuda:0 --compile --cuda-graphs --inference-cache
```

The validated inference cache combines adjacent frozen linear projections.
It retains the same weights and mathematical function, with small float32
rounding differences; omit its flag to use the original operation ordering.
Evaluation includes Diffusion, Two-for-One, Mixture, and Fokker–Planck.

## Environment

Install [Miniforge](https://conda-forge.org/download/) if Conda is unavailable, then:

```bash
conda env create -f environment.yml
conda activate ggf_md
python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

No Docker, local CUDA toolkit, JAX, experiment tracking account, or package
installation of this repo is needed. On this account Miniforge is installed at
`~/.local/miniforge3`; activate its shell support with
`source ~/.local/miniforge3/etc/profile.d/conda.sh`.

## Data and training

```bash
python train.py --prepare-data
python train.py --method diffusion --device cuda:0 --compile --microbatch 1024
python train.py --method mixture --device cuda:1 --compile --microbatch 1024
python train.py --method fokker_planck --device cuda:2 --compile --microbatch 512
```

Data preparation downloads the seven split archive parts from the
[official release](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0), verifies
them, extracts the split archive with the environment's `7zip` utility, and prepares
the original sequence splits. Downloads total approximately 7.3 GB; allow additional
space for extraction, processed data, checkpoints, and trajectories.

The experiment excludes glycine-containing sequences, retains atoms
`N, CA, CB, C, O` for each residue, and preserves first-frame Kabsch alignment and
training-only scalar normalization. Coordinates are in nm. The split contains
175 training, 75 validation, and 92 test peptides. Evaluation uses the paper's
15 held-out peptides: AC, AP, AT, ET, HP, HT, IM, KQ, KS, LW, NF, NY, RL, RV, TD.

Use `--microbatch N` to control memory use without changing the effective batch
size of 1024. The three methods train independently and can use separate GPUs.
Checkpoints include optimizer and RNG state; resume with `--resume PATH`.
Use `--profile-only` before a long run to measure actual throughput on shared GPUs.
The configuration defaults to one full run with seed 1: 120 epochs for Diffusion
and Fokker–Planck, and 10/20/100 epochs for the high/middle/low-time Mixture experts.

## Evaluation

```bash
python evaluate.py --method all --device cuda:2
```

Two-for-One shares Diffusion weights and independent samples, and changes the
simulation diffusion time from `1e-5` to `0.02`. Independent sampling uses the
original 1,000-step reverse VP Euler–Maruyama calculation and 49,000 samples per
peptide. Each simulation uses ten replicas of six million steps at 0.5 fs,
giving 30 ns aggregate time per peptide. Every 50th frame is stored; 49,000
frames are selected for comparison. Simulation progress is resumable.

Results include per-peptide values, means and population standard deviations
across the 15 peptides, published reference values, and PMF plots for independent
generation and simulation. KS is the paper's principal example. Short evaluation
overrides are for smoke checks; they must not be reported as the full reproduction.

## Fidelity choices

The graph transformer, source time features, mixture boundaries, finite-difference
loss, optimizer settings, and numerical samplers are preserved. Attention uses
eight heads of dimension 64. The conservative transformer has 1,398,497
parameters; the mixture has 2,763,997. The source EMA update is
`ema = 0.995 * new_parameters + 0.005 * ema`, with parameters replaced by EMA at
each epoch end; this convention is preserved.

The scalar FP coefficient is **0.0005**, following the paper's Table 6. Both the
released checkpoint and training instructions instead use **0.005**. The source's
FP residual constant, dimension normalization, DSM weighting, and asymmetric time
finite differences remain unchanged. The source scalar coefficient is named
`beta`; its separate vector penalty remains zero. These choices reproduce the
agreed experiment and are recorded rather than treated as equivalent formulas.

Two metric definitions are reported on the same samples:

- **Paper interpretation (primary):** fixed 64×64 bins on `[-pi, pi]^2`, add-one
  count smoothing for natural-log JSD, and uniform average squared log-proportion
  error with baseline `1e-6` for PMF. The square root of JSD is also reported as
  JS distance because the prose uses both terms.
- **Released evaluator (comparison):** unsmoothed natural-log JSD with bin edges
  from the combined observed range, and probability-weighted PMF error using the
  original outlier-bin and missing-bin conventions.

The published values correspond to the original evaluation implementation;
comparison metrics help distinguish metric-definition changes from model error.
The physical Langevin noise uses the source/OpenMM variance `kBT*(1-a*a)/mass`,
where `a=exp(-gamma*dt)`; the squared `(1-a)` printed in paper Equation 33 is
inconsistent with that integrator. No force clipping or extra burn-in is added.
Hyperparameters are not selected against test-set results.

| Method | IID JSD | Simulation JSD | IID PMF | Simulation PMF |
|---|---:|---:|---:|---:|
| Diffusion | 0.0155 | 0.2256 | 0.206 | 6.515 |
| Two-for-One | 0.0153 | 0.0466 | 0.203 | 0.741 |
| Mixture | 0.0155 | 0.0444 | 0.200 | 0.658 |
| Fokker–Planck | 0.0200 | 0.0254 | 0.241 | 0.368 |

Reference means: paper Appendix C.8.1, PDF Table 17. Published standard deviations
are across peptides, not training seeds. Shared IID samples make Diffusion and
Two-for-One results identical here; their published difference reflects sampling.

Code retains the original MIT license and attribution. Dataset provenance and
licenses are described in the paper and official release.
