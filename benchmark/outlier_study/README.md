# Outlier detection and clustering study

This harness measures how often the three outlier displays flag true and
false outliers, and how well the four clustering methods recover known
groups, with every method at its package defaults. It is outside the test
suite. The full study runs on USC Discovery. The two-cell `smoke/` run
checks only that the harness works, and the `smoke/probe/` run only feeds
the cost estimate; their numbers are not study evidence. No package
defaults change.

## Configuration and recreation

`configurations.csv` has 88 cells, each with 200 replicates:

| Study | Factor | Levels |
|:--|:--|:--|
| Outlier detection | model | fdaoutlier `simulation_model1` … `simulation_model9` (below) |
| | contamination | 5%, 10% of the curves (`round(contamination·n)`) |
| | design | dense (50-point grid on [0,1]), sparse (5–10 uniform times per curve) |
| | `n` | 100, 500 |
| Clustering | groups differ in | `mean`, `covariance` (below) |
| | groups | 2, 3 (balanced) |
| | design | dense, sparse, as above |
| | `n` | 100, 500 |

The routes in a cell get the same sample in each replicate, recreated from
the same seed:

| Route | Call |
|:--|:--|
| `boxplot` | `functional_boxplot(x)` |
| `outliergram` | `outliergram(x)` |
| `msplot` | `magnitude_shape(x; rng)` |
| `kmeans_curves` | `fit(FunctionalKMeans, x; k, rng)`, dense cells only |
| `kmeans_scores` | `fit(FunctionalKMeans, fit(FPCA, x; ncomponents=FVE(0.9)); k, rng)` |
| `kcfc` | `fit(KCFC, x; k, rng)` |
| `mixture` | `fit(FunctionalGaussianMixture, x; k, rng)` |

`k` is the true number of groups. That gives 43,200 outlier-detection fits
and 11,200 clustering fits. `FunctionalKMeans` refuses irregular curves, and
on the PACE work grid k-means of reconstructed curves is k-means of the
scores (the eigenfunctions are orthonormal there), so sparse cells have no
`kmeans_curves` route. `FVE(0.9)` is the FPCA that kCFC's initial step and
the mixture use by default (fdapace `kCFC` and `FClust`), so the three
score-based routes cluster the same scores.

No extra dependencies are needed: every script is Julia, using the package
environment. Instantiate the project with `scripts/instantiate.jl`, which
includes the KernelSmoothing master checkout, before running. Do not commit
the Manifest.

From the repository root, using new output paths:

```sh
julia benchmark/outlier_study/configs.jl /tmp/outlier-configurations.csv
julia --project=. benchmark/outlier_study/run.jl /tmp/outlier-configurations.csv 1 /path/to/results/cell-001.csv
julia --project=. benchmark/outlier_study/summarize.jl benchmark/outlier_study/configurations.csv /path/to/results /path/to/summary
```

The last command needs every configured cell. `run.jl` takes a CSV, a
one-based row index and an output path. It also accepts optional inclusive
replicate bounds and an optional route:

```sh
julia --project=. benchmark/outlier_study/run.jl CONFIG.csv INDEX OUTPUT.csv [FIRST LAST [ROUTE]]
```

Without bounds, `run.jl` runs the whole cell. With bounds alone it runs all
of the cell's routes over that block, and the last argument selects one.
Replicate seeds keep their original indices, so splitting the work does not
change the samples. Each invocation writes one CSV result row with its
configuration, replicate bounds and routes, versions, elapsed time, and
per-route outcome counts, resource summaries and metric means and counts.

Companion `.replicates.csv` and `.failures.csv` files keep every attempt with
its seed, metrics, number of true outliers, recorded settings (for example
the kCFC or EM status, iteration counts, FPCA component counts, the MS-plot
cutoff and MCD subset count) and error. Allocated bytes and wall time are
kept for failed fits too. Every `Exception` from a fit or metric computation
counts as a failure, recorded with its exception type and stage; for example
kCFC throws when its initial k-means leaves a cluster of three or fewer
curves, and the mixture throws when every EM restart degenerates.
`InterruptException` and `OutOfMemoryError` instead abort the worker.
Failures are never replaced. A fit that returns with a warning (kCFC's
`:tiny_cluster`, `:lost_cluster` or `:maxiter`, the mixture's `:unstable`)
is a success whose status is recorded; the summary counts these as "not
converged". An interrupted block keeps its companions but gets no final
result row.

Existing outputs are rejected. The summary merges the whole-cell and chunk
outputs named `cell-*.csv` in one directory and requires exactly one retained
attempt for every configured route and replicate. It rejects missing or
duplicate records, wrong seeds, metrics that do not belong to the cell's
study or lie outside their range, a wrong number of true outliers,
inconsistent aggregate rows, missing failure diagnostics, configurations
that differ from the grid and mismatched versions. Quality summaries are
conditional on success, with failure counts shown. Resource summaries
include failed attempts. Outlier rows also carry the model's outlier type.

## Scientific and measurement choices

- Outlier designs are the nine published simulation models of fdaoutlier
  0.2.1 (`simulation_model1` … `simulation_model9` in
  `R/simulation_models.R`, audited from the CRAN source), at their default
  parameters, so that detection rates are comparable with published studies
  that use these generators. Several are the designs of the papers behind
  fdaoutlier's methods, such as Dai and Genton (2018). Each curve is `X(t) = m(t) + e(t)` on [0,1], where `e` is a
  zero-mean Gaussian process with covariance `α exp(-β|s - t|^ν)`, by default
  `α = β = ν = 1`. With `k = ±1` and `u ∈ {0, 1}` equally likely:

  | Scenario | Type | Typical `m(t)` | Outlying `m(t)` |
  |:--|:--|:--|:--|
  | `model1` | shift | `4t` | `4t + 8k` |
  | `model2` | magnitude (isolated) | `4t` | `4t + 8k·1{T ≤ t ≤ T + 0.05}`, `T ~ U(0.1, 0.9)` |
  | `model3` | magnitude (partial) | `4t` | `4t + 6k·1{t ≥ T}`, `T ~ U(0.1, 0.9)` |
  | `model4` | shape | `30t(1 - t)^1.5` | `30(1 - t)t^1.5`; both with `α = 0.3`, `β = 1/0.3` |
  | `model5` | shape (covariance) | `4t` | `4t`, with `α = 5`, `β = 2`, `ν = 0.5` |
  | `model6` | mixed | `4t` | `4t + (-1)^u·1.8 + (-1)^(1-u) exp(-50(t - v)²)/√(0.02π)`, `v ~ U(0.25, 0.75)` |
  | `model7` | shape (phase) | `4t` | `4t + 2 sin(4π(t + θ))`, `θ ~ U(0.25, 0.75)` |
  | `model8` | shape (phase) | `2 sin(15πt)` | `2 sin(15πt + 2)` |
  | `model9` | amplitude | `a₁ sin 2πt + a₂ cos 2πt`, `aⱼ ~ U(3, 8)` | the same with `aⱼ ~ U(9, 10.5)` or `U(1.5, 2.5)`, equally likely |

  The types follow Hubert, Rousseeuw and Segaert (2015): a shift outlier has
  the typical shape moved by a constant, a magnitude outlier is extreme on
  part of the domain, a shape outlier has a different pattern within the
  range of the typical curves, an amplitude outlier scales the typical
  pattern, and a mixed outlier combines a shift with a shape feature.
  fdaoutlier's own `outlier_rate` default (0.05, or 0.1 for model 6) is
  replaced by the study's contamination; as with its `deterministic = TRUE`
  default, the outlying curves are a uniformly random subset of
  `round(contamination·n)` curves.

  Dense curves are the models on fdaoutlier's default 50-point grid. Sparse
  curves evaluate the same mean functions and covariance at 5–10 uniform
  times per curve. There is no measurement noise; `e` itself is rough. Truth
  is defined by the generating process: a sparse model 2 outlier whose
  interval holds no observation still counts as an outlier, and the 15π
  oscillation of model 8 cannot be resolved from 5–10 points.

  The generators draw the curve parameters with Julia's RNG in their own
  order, so samples differ from fdaoutlier's for the same seed; the
  deterministic parts are the same. `reference/fdaoutlier_models.R` runs
  fdaoutlier with the process switched off (`cov_alpha = 1e-30`), recovers
  each curve's random parameters by replaying fdaoutlier's RNG draws, and
  writes the curves with their parameters and the three covariances to
  `reference/*.csv`; `check.jl` reproduces them at relative error below
  `1e-8`.
- Clustering designs use `φ_j(t) = √2 sin(jπt)`, `j = 1, 2, 3`, orthonormal
  on [0,1]. Curves are `1 + t + Σ_j ξ_j φ_j(t) + ε` with independent Gaussian
  scores and measurement noise of standard deviation 0.1:
  - `mean`: group means 2.5 apart along `φ_1` (`ξ_1` has mean
    `2.5(g - (G + 1)/2)` in group `g`); common score standard deviations
    `(1, √0.5, 0.5)`. Adjacent groups overlap: the Bayes error of two groups
    is `Φ(-1.25) ≈ 11%`.
  - `covariance`: one mean for all groups; group `g` has score standard
    deviation 1 along `φ_g` and 0.2 along the other modes, as in the
    clustering page's example. The group means coincide, so methods that
    compare curves only through distances to centres have no signal.

  Labels are interleaved, `g = (i - 1) mod G + 1`.
- Each replicate uses `MersenneTwister(seed + replicate - 1)` for the data
  and `MersenneTwister(16(seed + replicate - 1) + r)` for the method's own
  randomness (MCD subset sampling, k-means++ seeds, EM starts), where `r` is
  the route's position in `ALL_ROUTES`. The configuration generator records
  each cell's starting seed. Julia's RNG stream depends on the Julia version,
  so reproduce with the recorded version. There are no RNG-dependent golden
  assertions.
- Every call uses the package defaults; the defaults and their reference
  sources are listed on the "Depth and outliers" and "Clustering and
  classification" pages. Sparse curves reach the outlier displays through
  the default PACE reconstruction (`representation = :fpca_reconstruction`).
- Outlier metrics: the true positive rate is the fraction of true outliers
  flagged, the false positive rate the fraction of typical curves flagged,
  and `flagged` the number of curves flagged. Clustering metric: the
  adjusted Rand index (Hubert and Arabie, 1985) of the assignments against
  the true labels, from Clustering.jl `randindex`.
- `check.jl` checks the harness against known answers: the grid and its
  committed CSV; configuration validation; the outlier models' mean
  functions and covariances against fdaoutlier (above); the parameter
  ranges and outlier types; the process variances of replayed samples and
  the Cholesky factors; orthonormality of the clustering modes and the group
  offsets; replayed samples; hand-computed true and false positive rates and
  adjusted Rand indices; failure retention; CSV round trips and aggregates;
  work-list coverage; the cost estimate on a synthetic probe; summary
  reproduction from the committed smoke run;
  merging of chunks with a counted failure; and rejection of adverse result
  directories.
- Each route runs in its own Julia child process over its replicate block.
  A warmup with 60 curves and seed 20261027 (up to three replicate seeds,
  since kCFC refuses tiny initial clusters) compiles the fit and metrics
  before timing. Garbage collection runs before each fit, and one Julia
  thread and one BLAS thread are used. `@timed` measures the complete public
  call, including the FPCA (or PACE reconstruction) the method fits.
  Simulation, metrics, startup and warmup fall outside the timer.
  `allocated_bytes` is total GC-tracked allocation during the call.
  `process_peak_rss_bytes` is the worker's cumulative `Sys.maxrss()`, which
  includes startup and JIT. The summary reports its maximum.

## Local smoke recreation

```sh
julia --project=. benchmark/outlier_study/check.jl
julia benchmark/outlier_study/configs.jl /tmp/outlier-smoke.csv --smoke
julia --project=. benchmark/outlier_study/run.jl /tmp/outlier-smoke.csv 1 /tmp/outlier-smoke/cell-001.csv
julia --project=. benchmark/outlier_study/run.jl /tmp/outlier-smoke.csv 2 /tmp/outlier-smoke/cell-002.csv
julia --project=. benchmark/outlier_study/summarize.jl /tmp/outlier-smoke.csv /tmp/outlier-smoke /tmp/outlier-smoke-summary
```

The smoke grid has two cells of 40 curves and two replicates: sparse
model 6 (mixed) outliers at 10% and dense two-group covariance clustering.
`smoke/run.json` records the local validation and recipe. Full-study
results belong in `results/`.

## Cost estimate

The probe grid (`configs.jl --probe`) has one full-size cell per study,
design and `n` (model 1 at 10%; three-group covariance clustering), with two
replicates each. `estimate.jl` takes the mean fit time per study, design,
`n` and route from a complete probe run, and the mean per-worker overhead
(startup, warmup, simulation and garbage collection), and adds them up over
the work list:

```sh
julia benchmark/outlier_study/configs.jl /tmp/outlier-probe.csv --probe
julia --project=. benchmark/outlier_study/run.jl /tmp/outlier-probe.csv 1 /tmp/outlier-probe/cell-001.csv   # ... through 8
julia --project=. benchmark/outlier_study/estimate.jl benchmark/outlier_study/configurations.csv /tmp/outlier-probe.csv /tmp/outlier-probe
```

It assumes the other models and scenarios cost as much as the probed ones,
and its timings come from the machine that ran the probe. `smoke/probe/`
keeps the local probe behind the starting resource requests below; rerun it
on a Discovery node before relying on the estimate there.

## Discovery work units

`work_units.jl` creates a complete, deterministic TSV list with one row per
cell, route and inclusive replicate block, for example
`cell-001-msplot-0001-0200.csv`:

```sh
julia --project=. benchmark/outlier_study/work_units.jl benchmark/outlier_study/configurations.csv /path/to/work-units.tsv --block-size 200
julia --project=. benchmark/outlier_study/work_units.jl benchmark/outlier_study/configurations.csv /path/to/work-units.tsv --check
```

The default block is all 200 replicates, one unit per cell and route, which
gives 272 units. Most fits take well under a second, so worker startup and
warmup, repeated for every unit, would dominate smaller blocks. The local probe in `smoke/probe/`
(`estimate.txt`) puts the full work list at about 8.1 CPU-hours: 15 s of
startup per unit, 0.035 s per replicate for simulation and garbage
collection, and fit times from under 0.01 s (dense curves) to 14 s (sparse
kCFC with `n = 500`, three groups). Sparse clustering at `n = 500` takes 3.4
of the 8.1 hours and its longest unit about 46 minutes. With
`sbatch --ntasks=8` the work list fits the four-hour request below with a
wide margin.
Keep the configurations and the 200 replicates unchanged.

`discovery.sbatch` runs GNU Parallel inside one Slurm allocation, launching
exclusive one-CPU `srun` steps through `worker.sh`. Set `JULIA_MODULE`,
`FDA_PROJECT`, `STUDY_CONFIG`, `STUDY_UNITS` and `STUDY_OUTPUT` to
site-specific absolute paths and module names. Load GNU Parallel beforehand
or through the site environment. Instantiate the project before submission;
jobs do not install dependencies. The template contains no account names or
credentials. As starting requests it asks for one node, one CPU, 16 GB and
four hours. Set the concurrency with `sbatch --ntasks=N` and keep
`--cpus-per-task=1`. `STUDY_WORKERS` can lower concurrency below
`SLURM_NTASKS`. Each step requests `STUDY_STEP_MEM` (default 1900M; local
probe workers peaked at about 1.5 GB). These requests are not a validated
budget.

GNU Parallel writes each unit's exit status to
`STUDY_OUTPUT/parallel-joblog.tsv`. An aborted unit stops new launches, and
the driver exits nonzero once active units finish. Counted fit and metric
failures are completed attempts and do not trigger a rerun. The driver saves
the configuration and work list next to the joblog and rejects changes to
them on restart. Only one allocation may use a results directory at a time.

To resume an infrastructure-aborted run, keep the same output directory,
work list and joblog. First move each aborted unit's files, those with a
`.replicates.csv` companion but no result row, into an attempt directory:

```sh
cd /path/to/results && mkdir aborted-attempt-1
for f in cell-*.replicates.csv; do b=${f%.replicates.csv}; [[ -e $b.csv ]] || mv "$b".* aborted-attempt-1/; done
```

Then resubmit with the same inputs. The driver passes `--resume-failed`
when the joblog exists. Never rerun completed blocks to replace unfavourable
outcomes. After changing block sizes or the work-list order, use a fresh
directory.

Keep a small run record with the results:

- source commit;
- configuration;
- Julia and dependency versions, including the KernelSmoothing revision;
- launch command;
- requested resources;
- returned job ID;
- outputs and checks.

Submit with a unique run tag and save the returned ID:

```sh
sbatch --parsable --job-name="outlier-study-RUN_TAG" --export=ALL benchmark/outlier_study/discovery.sbatch
```

Before summarizing, review the job accounting, the joblog and complete
result coverage. Then use the summary to fill the "Choosing a method" notes
on the "Depth and outliers" and "Clustering and classification" pages.
