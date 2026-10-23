# GGF training handoff

**Status: stopped for handoff. No controller or training worker is running, and nothing will restart automatically.**

Training was stopped at the user's request at 2026-10-23T21:14:18.904016+00:00. The registered training process exited with SIGTERM (`-15`), and its wrapper recorded the actual exit before terminating. The original run and validation evidence are retained.

## Saved progress

- Completed epochs in the checkpoint: **2 of 120**.
- Next/current epoch on resume: **3**, with **2,000 of 8,374 updates already saved**.
- Saved optimizer updates: **18,748 of 1,004,880**.
- Checkpoint: [data/ggf/runs/ggf/seed1/latest.pt](data/ggf/runs/ggf/seed1/latest.pt).
- Checkpoint bytes: `99846489`; SHA-256: `2c6e5be1554ac253bc38d224c6c4c4ac9c0b6bd2108e8f9fff9d081d054dd393`.
- Model, EMA, Adam states, scheduler, data permutation, and Python/NumPy/Torch/CUDA RNG state are preserved. The checkpoint was validated after the processes stopped and was not rewritten during relocation.

Only committed checkpoint progress is retained. SIGTERM does not create a new checkpoint, so updates after the last regular save will be replayed on resume.

## Everything available from this repository

| Item | Path |
|---|---|
| GGF configuration | [configs/dipeptides_ggf.yaml](configs/dipeptides_ggf.yaml) |
| Model and training implementation | [model.py](model.py), [train.py](train.py) |
| Paired-singleton GGF objective | [ggf_score_matching.py](ggf_score_matching.py) |
| IID and simulation evaluation | [evaluate.py](evaluate.py) |
| Resumable pipeline | [scripts/run_ggf_experiment.py](scripts/run_ggf_experiment.py) |
| Model/loss/sampling contract | [GGF_EXPERIMENT.md](GGF_EXPERIMENT.md) |
| Machine-readable handoff | [GGF_HANDOFF.json](GGF_HANDOFF.json) |
| Dataset and normalization manifest | [data/ggf/processed/manifest.json](data/ggf/processed/manifest.json) |
| Checkpoints and training logs | [data/ggf/runs/ggf/seed1](data/ggf/runs/ggf/seed1) |
| Controller status and stage logs | [data/ggf/experiment](data/ggf/experiment) |
| CPU/CUDA validation evidence | [data/ggf/validation](data/ggf/validation) |
| Main-repository validation | [data/ggf/implementation/main-folder-migration/validation](data/ggf/implementation/main-folder-migration/validation) |
| Runtime and installed packages | [environment-at-handoff.json](data/ggf/implementation/main-folder-migration/environment-at-handoff.json), [environment.yml](environment.yml) |
| Source, stop, and checkpoint records | [data/ggf/implementation/main-folder-migration](data/ggf/implementation/main-folder-migration) |
| Completed baseline snapshot | [data/ggf/baselines/non-fokker-planck](data/ggf/baselines/non-fokker-planck) |

`data/ggf` is a link to the existing NFS run data, and its `processed` entry links to the existing dataset in this repository. Additional links expose validation, implementation records, caches, and the controller log under the same namespace. No large data files were duplicated or deleted. When transferring to another server, include the linked targets; a source-only Git checkout does not include the checkpoints or dataset. Controller process/path provenance is machine-specific and must be rebound before resuming on a different host.

The old separate worktree was archived, verified, and removed. The main repository passed 55 evaluator checks and 15 runner checks on CPU. Native trainer/checkpoint compatibility was verified without starting training.

## Resume only when explicitly requested

From the current repository on this server:

```bash
cd /home/anirudh/ggf_md
CUDA_VISIBLE_DEVICES='' \
MPLCONFIGDIR="$PWD/data/ggf/matplotlib" \
PYTHONDONTWRITEBYTECODE=1 \
/home/anirudh/.local/venvs/ggf_md/bin/python -u scripts/run_ggf_experiment.py \
  --config configs/dipeptides_ggf.yaml --gpus 3,2,1,0 --microbatch 1024 \
  --compile --factorized-attention --inference-cache --cuda-graphs
```

The handoff state is already bound to this main repository. This command resumes the compatible existing `latest.pt`; it does not start from random weights. Keep the configured total at **120 epochs**, with the same batch, execution flags, and GGF bounds. No `--retry-failed` is required for the documented handoff stop. GPU IDs are preferences; the runner waits for a genuinely idle GPU.

Read status without starting training:

```bash
cd /home/anirudh/ggf_md
/home/anirudh/.local/venvs/ggf_md/bin/python scripts/run_ggf_experiment.py \
  --status --state-dir data/ggf/experiment
```

## Remaining experiment

Finish the saved 120-epoch schedule, then generate 49,000 IID samples for each of 15 peptides, run the full 150-path/six-million-step simulation, and compute metrics and plots. The runner performs these stages automatically after an explicit resume. **GGF IID, full simulation, and trained-model metrics have not started.** The small files under `validation/` are preflight evidence, not trained GGF results.

The single energy model, fixed-time features, raw singleton loss, `h = sigma_min = 0.0010004971259352385`, VE range up to 20, and agreed IID-only `h²/sigma(t)²` adapter are unchanged. Full details are in [GGF_EXPERIMENT.md](GGF_EXPERIMENT.md).
