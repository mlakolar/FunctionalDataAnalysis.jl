# Noise-variance estimators (issue #17)

The `SmoothedCovariance` keywords are `noise_method=:rotated|:surface_diagonal`
and `noise_interval=:full|:middle_half`; the defaults (first in each pair) were
chosen by the issue #17 benchmark on USC Discovery (48 designs × 500
replicates): the rotated diagonal over the full interval had the lowest MSE in
41 of 48 cells, and both surface estimators had much larger upward bias. Raw estimates and truncation flags
remain available. The rotated calculation follows the source-audited fdapace
0.6.0 `PC_CovE` calculation at fixed bandwidths; it does not alter the stored
covariance surface. The [source audit](@ref fdapace-rotated-noise) gives the
exact rotation, polynomial columns, bandwidths, cut rule, and departures.

## Defaults and their sources

| Setting | Value / behavior | Source |
|:--|:--|:--|
| Noise method | `:rotated` | Issue #17 benchmark; fdapace 0.6.0 `PC_CovE` (Yao, Müller and Wang 2005). `:surface_diagonal` keeps the milestone 2 estimator. |
| Noise interval | `:full` | Issue #17 benchmark (lower MSE than the middle half in 42 of 48 cells); internal `PC_CovE` formal default. fdapace 0.6.0 `SetOptions` uses `rotationCut=c(.25,.75)`, available as `:middle_half`. |
| Grid, covariance bandwidth | Required; no new automatic selection | Existing `SmoothedCovariance` contract. Explicit `CurveCV` candidates remain supported. |
| Raw-square bandwidth | Selected covariance bandwidth unless overridden | fdapace 0.6.0 `PC_CovE` uses the first covariance bandwidth for raw squares. Existing separate override is retained. |
| Kernel | Epanechnikov at the `SmoothedCovariance` entry point | Existing API; fdapace 0.6.0 low-level `RotateLwls2DV2` / `PC_CovE` formal defaults. PACE's default is Gaussian. |
| Weighting | `:obs`; optional `:subj` | Existing package convention. Observation weights reproduce the raw fdapace calculation; subject weights are an explicit weighted extension. Stored data weights are unused. |
| Binning | `nothing` | Existing API; no snapping unless requested. |
| Raw-square degree / minimum observations | Local linear / `minobs=0` | Existing pooled smoother, using KernelSmoothing; undefined fits raise errors. |
| Rotated coordinates and degrees | 45 degrees; intercept, across squared, along linear | fdapace 0.6.0 executable `RrotatedMullwlsk.cpp`, rather than its inaccurate wrapper comment. |
| Rotated bandwidth | Selected covariance bandwidth in both axes | `PC_CovE` and `RotateLwls2DV2`; no additional rotation scale factor. A raw-square override does not change it. |
| Surface integration | Work-grid domain; exact boundaries interpolated; trapezoidal weights | Existing package rule. Full-interval option follows the unrotated integrated-diagonal approach of Li and Hsing (2010). |
| Rotated integration | Observed domain; retain existing grid points with floating-point boundary buffer; divide by nominal interval length | fdapace 0.6.0 `PC_CovE`. No boundary insertion. |
| Negative estimate | Zero, with raw value and flag retained | Existing package rule. Deliberate departure from fdapace's enclosing negative-to-`1e-6` reset. |
| Undefined rotated fit / too few integration points | Error; at least two retained points | Package validation convention. No R one-point fallback or rank-deficient solve. |
| PSD projection | Benchmark only; no package keyword | Task scope. Historical FGM rule, audited at commit `6f0eac3d5cbe0fad93e88dcad28f389d89b61495`. |

All new fit choices are stored in `noise_details` and displayed by `show`.
The details include method, interval, domain, bounds, nominal length, boundary
rule, integration grid, fitted latent diagonal, and rotation settings. The
raw-square bandwidth/CV details are also retained.

## Reference values

The fixed-bandwidth reference fixture uses **R 4.5.1 / fdapace 0.6.0**.
Recreate it from the repository root with:

```sh
Rscript test/reference/fdapace_m5_sigma2.R
```

The comparisons cover 48 combinations of weighting, kernel, bandwidth, grid,
and interval. Raw variance and both fitted diagonals are compared at relative
error below `1e-8`. The [source audit](@ref fdapace-rotated-noise) describes
the quantities compared and the deliberate departures from fdapace.

## [Discovery benchmark](@id sigma2-discovery-results)

The reference run on **25 October 2026** used **Julia 1.13.2** and
FunctionalDataAnalysis **0.1.0**, source commit
`d26162c17fe1b221d0648023ca5b69728095abc1`. It covered 48 configurations
with 500 replicates each. All 120,000 estimates succeeded.
The complete [CSV results](https://github.com/mlakolar/FunctionalDataAnalysis.jl/blob/main/benchmark/sigma2/results/summary.csv)
and [per-cell table](https://github.com/mlakolar/FunctionalDataAnalysis.jl/blob/main/benchmark/sigma2/results/summary.md)
retain every configuration and estimator. The
[results README](https://github.com/mlakolar/FunctionalDataAnalysis.jl/blob/main/benchmark/sigma2/results/README.md)
records provenance and recreation instructions.

The following are equally weighted averages across the 48 configurations.
Bias and MSE use zero-truncated estimates; each cell's variance uses divisor
499. The last column counts configurations with the lowest observed MSE
among the five estimators.

| Estimator | Mean bias | Mean variance | Mean MSE | Lowest-MSE cells / 48 |
|:--|--:|--:|--:|--:|
| Surface, middle half | 0.23124201 | 0.014963528 | 0.078911898 | 0 |
| Surface, full | 0.18349587 | 0.011246044 | 0.050534863 | 0 |
| Rotated, middle half | 0.044148727 | 0.018125610 | 0.023049226 | 6 |
| Rotated, full | 0.046097678 | 0.013957901 | 0.017869929 | 41 |
| Projected FGM reference | 0.20062286 | 0.016532224 | 0.065457949 | 1 |

Rotated/full has lower MSE than rotated/middle-half in 42 of 48 cells.
The six exceptions favor rotated/middle-half: dense designs at ratio `.5`
with fixed bandwidths (both sample sizes and weightings), and dense `n=400`
at ratio `.05` with CurveCV (both weightings). The projected reference has
the lowest MSE for sparse `n=100`, ratio `.5`, subject weighting and fixed
bandwidths. Low-noise sparse estimates remain biased even with rotation.
These results concern the specified Fourier process; they do not establish
universal optimality. Dense observation- and subject-weighted cells share
data and give the same estimates, so they are not independent replications.

## Harness smoke check

Both cells use `n=100`, the mean `1+t`, three Fourier components with variances
`[1,.5,.25]`, independent Gaussian errors, and a 51-point observed-range grid.
Sparse means 2–6 random times per subject; dense means 50 common times.
Ratio means noise variance divided by integrated signal variance 1.75.
The first cell uses ratio .05 / observation weights / fixed PACE bandwidths;
the second uses ratio .2 / subject weights / five-fold CurveCV for both mean
and covariance. True noise variances are .0875 and .35, respectively.

| Cell | Estimator | Bias | Variance | MSE | Truncated / 20 |
|:--|:--|--:|--:|--:|--:|
| Sparse / default | Surface, middle half | 0.29944197 | 0.015985618 | 0.10485183 | 0 |
| Sparse / default | Surface, full | 0.26735865 | 0.014494808 | 0.085250716 | 0 |
| Sparse / default | Rotated, middle half | 0.14722968 | 0.020957519 | 0.041586222 | 2 |
| Sparse / default | Rotated, full | 0.13971713 | 0.019763211 | 0.038295927 | 1 |
| Sparse / default | Projected FGM reference | 0.23820340 | 0.014314953 | 0.070340067 | 0 |
| Dense / CV | Surface, middle half | 0.082537750 | 0.00015144770 | 0.0069563555 | 0 |
| Dense / CV | Surface, full | 0.073739356 | 0.00013757531 | 0.0055681892 | 0 |
| Dense / CV | Rotated, middle half | -0.0079882555 | 0.00013099662 | 0.00018825901 | 0 |
| Dense / CV | Rotated, full | 0.0035653155 | 0.000073695369 | 0.000082722075 | 0 |
| Dense / CV | Projected FGM reference | 0.071959639 | 0.00013463372 | 0.0053060917 | 0 |

Bias and MSE use the truncated estimate; variance uses denominator 19. Each
row has 20/20 successes. These two cells deliberately differ in several
factors and cannot isolate design, weighting, or CV effects. The low-noise
sparse cell has substantial bias even with rotation. These are harness checks,
not a full benchmark, recovery guarantee, or default recommendation.

The auditable inputs and outputs live in `benchmark/sigma2/smoke/`:
configuration CSV, two single-row result CSVs, raw/truncated replicate CSVs,
failure CSVs (headers only), generated summary CSV/markdown, and `run.json`.
The README supplies commands to reproduce them to fresh paths.

## Benchmark settings

These are explicit harness settings. They do not add package defaults:

- Process: mean `1+t`, three normalized Fourier functions, eigenvalues
  `[1,.5,.25]`; variance ratio rather than standard-deviation ratio.
- Grid/kernel: 51 equally spaced observed-range points and Gaussian, matching
  audited PACE defaults. Dense subjects share 50 equally spaced times.
- CV: five curve folds, mean candidates `[.025,.05,.1]` and covariance
  candidates `[.05,.1,.2]` times the observed range. Mean and covariance use
  the same deterministic split. Covariance CV is conditional on the supplied
  full-sample mean, as in the existing package. Raw-square bandwidth inherits
  the selected covariance bandwidth.
- Pairing/seeds: `MersenneTwister` seeds shared across weighting and bandwidth
  choices. The exact Julia version is recorded because streams can change.
- Packaging: one configuration contains all five estimators, so 48 tasks cover
  240 estimator cells. Each task writes one summary row plus raw/failure
  companions. Exact duplicate aggregation preserves weighted fits and CV
  losses; it never snaps times or discards pairs.
- Projected reference: all positive weighted eigencomponents with FGM's
  `100eps(Float64)` relative cutoff and its arithmetic index-window average.
  Common smoothed inputs isolate the diagonal rule. This is a benchmark-only
  comparison, not end-to-end FGM equivalence.
- Failures: no replacement or silent dropping. Fit `ArgumentError`s are recorded;
  successful-fit moments are explicitly conditional if any fail. Unexpected
  errors abort. Variance uses `R-1`; MSE uses `R`. Output overwrite is rejected.
- Discovery template: one CPU / 2 GB / 24 hours per task, array concurrency
  eight, one Julia/BLAS thread. Module, project, configurations and outputs are
  variables. The owner selects current allocation/partition and resource budget
  before a new run.
