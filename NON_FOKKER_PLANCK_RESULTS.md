# Non-Fokker–Planck dipeptide results compared with the paper

Results snapshot: **2026-10-23, 17:48 UTC**. This report covers the completed Diffusion, Two-for-One, and Mixture experiments.

The main comparison uses the **released-code metric definitions**, consistently with the results reported in chat. Reference values were checked directly against [Table 17, Appendix C.8.1, PDF page 33](https://arxiv.org/pdf/2506.17139v3#page=33) of *Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models*. The PMF references also appear in Table 2.

All metrics are errors: **lower is better**. Values are **mean ± standard deviation across 15 peptides**. Our standard deviations use the population convention (`ddof=0`); they do not measure uncertainty across independent training runs. Differences and percentage changes use unrounded means:

- Difference = this run − paper.
- Change = 100 × (this run − paper) / paper.
- Negative differences indicate lower error in this run.

## Independent sampling (IID)

| Method | Metric | This run | Paper | Difference | Change |
|---|---|---:|---:|---:|---:|
| Diffusion | JSD | 0.0160 ± 0.0087 | 0.0155 ± 0.0083 | +0.000518 | +3.34% |
| Diffusion | PMF error | 0.203 ± 0.143 | 0.206 ± 0.159 | -0.002601 | -1.26% |
| Two-for-One | JSD | 0.0160 ± 0.0087 | 0.0153 ± 0.0080 | +0.000718 | +4.69% |
| Two-for-One | PMF error | 0.203 ± 0.143 | 0.203 ± 0.149 | +0.000399 | +0.20% |
| Mixture | JSD | 0.0160 ± 0.0077 | 0.0155 ± 0.0078 | +0.000510 | +3.29% |
| Mixture | PMF error | 0.200 ± 0.110 | 0.200 ± 0.127 | -0.000446 | -0.22% |

Two-for-One reuses the Diffusion checkpoint and IID samples, so their IID results are identical in this run. The paper reports slightly different IID values for these methods; its two reference rows are retained separately.

## Molecular simulation

| Method | Metric | This run | Paper | Difference | Change |
|---|---|---:|---:|---:|---:|
| Diffusion | JSD | 0.2911 ± 0.1357 | 0.2256 ± 0.1304 | +0.065539 | +29.05% |
| Diffusion | PMF error | 17.862 ± 8.550 | 6.515 ± 3.175 | +11.347033 | +174.17% |
| Two-for-One | JSD | 0.0462 ± 0.0125 | 0.0466 ± 0.0114 | -0.000386 | -0.83% |
| Two-for-One | PMF error | 0.725 ± 0.256 | 0.741 ± 0.319 | -0.015745 | -2.12% |
| Mixture | JSD | 0.0427 ± 0.0229 | 0.0444 ± 0.0237 | -0.001661 | -3.74% |
| Mixture | PMF error | 0.545 ± 0.338 | 0.658 ± 0.407 | -0.113170 | -17.20% |

Using these definitions, Mixture has the lowest simulation errors in this run. Its JSD and PMF errors are below the corresponding reported means. Two-for-One is close to its reported means; Diffusion has higher simulation errors. These are comparisons of observed means, not tests of statistical significance.

## Completed run coverage

- All three methods were evaluated on all 15 peptides; there are no partial-peptide averages.
- IID evaluation uses 49,000 samples per peptide. Simulation metrics use 49,000 frames selected per peptide from the completed trajectories with a fresh seed-0 permutation.
- Each simulation completed 6,000,000 steps for each of 10 replicas per peptide, at 0.5 fs per step: 3 ns per replica and 30 ns aggregate per peptide.
- All 150 trajectories survived in each method; the full raw trajectories are finite and retained.
- Training seed is 1. Diffusion completed 120 epochs; Mixture completed its full 10/20/100-epoch schedule. Two-for-One shares Diffusion weights.
- Fokker–Planck results are outside this report. Completion here does not establish completion of the full four-method replication.

## Alternative metric definitions

The evaluator also saves metrics under the label `paper`, meaning the local interpretation of the written formulas. This label is distinct from the **published reference numbers** in the comparison tables above.

The released-code JSD uses unsmoothed counts with edges from the combined observed range; its PMF error weights bins by their mixture probability. The alternative implementation uses a fixed 64 × 64 angular grid, add-one JSD smoothing, and a uniform mean squared log-proportion error with a `1e-6` PMF baseline. Both implementations use the same samples. JSD here is the divergence, not its square root (JS distance).

For transparency, our alternative-definition results are:

| Method | IID JSD | IID PMF error | Simulation JSD | Simulation PMF error |
|---|---:|---:|---:|---:|
| Diffusion | 0.0117 ± 0.0074 | 1.003 ± 0.341 | 0.2483 ± 0.1190 | 6.924 ± 2.218 |
| Two-for-One | 0.0117 ± 0.0074 | 1.003 ± 0.341 | 0.0359 ± 0.0108 | 2.137 ± 0.442 |
| Mixture | 0.0115 ± 0.0064 | 1.148 ± 0.364 | 0.0346 ± 0.0207 | 1.362 ± 0.470 |

Metric definitions substantially affect PMF magnitudes. The main tables use one released-code definition across all methods; the alternative table documents the difference. The source comparison CSV includes both definitions against the same published references.

## Source artifacts

- [Full summary](data/ggf/baselines/non-fokker-planck/summary.json)
- [Per-peptide metrics](data/ggf/baselines/non-fokker-planck/per_peptide.csv)
- [Both definitions and paper comparisons](data/ggf/baselines/non-fokker-planck/comparison.csv)
- [Report provenance and hashes](data/ggf/baselines/non-fokker-planck/report_receipt.json)
- [Independent report review](data/ggf/baselines/all-non-fokker-planck-report-review.json)

Report receipt SHA-256: `577fc4b1b5e389f36be149c74684efb4c4a646c4917b4a74d5aaa00214fab1bc`.
