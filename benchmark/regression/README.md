# Scalar-on-function regression study

This harness compares FPC regression (`FPCRegression`) with penalized
regression (`PenalizedFunctionalRegression`) for a scalar response and one
functional covariate. It asks when FPC regression loses to penalized
regression and whether the coefficient bands cover. It is outside the test
suite. The full study runs on USC Discovery. The two-cell `smoke/` run checks
only that the harness works; its numbers are not study evidence. No package
defaults change.

## Configuration and recreation

`configurations.csv` has 24 cells, each with 200 replicates and three routes:

| Factor | Levels |
|:--|:--|
| Coefficient `β(t)` | `leading`, `trailing`, `smooth` (below) |
| Covariate design | dense (101-point grid on [0,1]), sparse (2–10 uniform times per curve) |
| Training subjects `n` | 100, 500 |
| Response noise | population R² = 0.8, 0.5 |
| Held-out subjects | 200 per replicate |

That gives 14,400 fitted models. The three routes in a cell get the same
sample in each replicate, recreated from the same seed:

| Route | Call |
|:--|:--|
| `fpc` | `fit(FPCRegression, x, y)`: package defaults, marginal FPCA with `FVE(0.99)` (fdapace `FLM`) |
| `fpc_bic` | `fit(FPCRegression, x, y; selection=BIC())`: leading-component count chosen by the regression BIC |
| `pfr` | `fit(PenalizedFunctionalRegression, x, y)` on dense curves, plus `presmooth=:fpca` on sparse curves (PACE reconstruction, refund `lf(presmooth=)`); thin plate `k=10`, REML |

No extra Julia dependencies are needed, and the Python scripts use only the
standard library. Instantiate the project with `scripts/instantiate.jl`,
which includes the KernelSmoothing master checkout, before running. Do not
commit the Manifest.

From the repository root, using new output paths:

```sh
python3 benchmark/regression/configs.py /tmp/regression-configurations.csv
julia --project=. benchmark/regression/run.jl /tmp/regression-configurations.csv 1 /path/to/results/cell-001.csv
python3 benchmark/regression/summarize.py benchmark/regression/configurations.csv /path/to/results /path/to/summary
```

The last command needs every configured cell. `run.jl` takes a CSV, a
one-based row index and an output path. It also accepts optional inclusive
replicate bounds and an optional route:

```sh
julia --project=. benchmark/regression/run.jl CONFIG.csv INDEX OUTPUT.csv [FIRST LAST [ROUTE]]
```

Without bounds, `run.jl` runs the whole cell. With bounds alone it runs all
three routes over that block, and the last argument selects one of `fpc`,
`fpc_bic` or `pfr`. Replicate seeds keep their original indices, so
splitting the work does not change the samples. Each invocation writes one
CSV result row with:

- its configuration, replicate bounds and routes;
- population quantities and versions;
- elapsed time;
- per-route outcome counts, resource summaries, and metric means and counts.

Companion `.replicates.csv` and `.failures.csv` files keep every attempt with
its seed, metrics, selected tuning values (component count, λ, EDF, band
covariance), scoring grid and error. Allocated bytes and wall time are kept
for failed fits too. Every `Exception` from a fit or metric computation
counts as a failure, recorded with its exception type and stage.
`InterruptException` and `OutOfMemoryError` instead abort the worker.
Failures are never replaced. An interrupted block keeps its companions but
gets no final result row.

Existing outputs are rejected. The summary merges the whole-cell and chunk
outputs named `cell-*.csv` in one directory and requires exactly one retained
attempt for every configured route and replicate. It rejects missing or
duplicate records, wrong seeds, routes in the same replicate with different
scoring grids, inconsistent aggregate rows, missing failure diagnostics, and
mismatched versions or population quantities. Quality summaries (means, SDs
and medians of the relative errors) are conditional on success, with failure
counts shown. Resource summaries include failed attempts.

## Scientific and measurement choices

- Covariates come from `simulate_kl`. The mean is `1+t`. The eigenfunctions
  are the first nine orthonormal Fourier functions on [0,1]: the constant,
  then sine/cosine pairs up to frequency 4. Scores are independent Gaussian
  with variances `1/k²`, and independent Gaussian measurement noise has
  variance `0.05 Σ 1/k²`. In population, `FVE(0.99)` keeps eight components;
  the ninth carries 0.8% of the variance.
- Coefficients:
  - `leading`: `β = φ₁ + φ₂`, the two leading eigenfunctions.
  - `trailing`: `β = φ₉`, the trailing eigenfunction (`√2 cos 8πt`).
  - `smooth`: `β(t) = exp(-(t-0.3)²/(2·0.1²))`, a smooth bump not built from
    the FPCs, in the style of Reiss and Ogden (2007). Its part outside the
    nine-function span (`unidentified_beta_norm`, under 5% of `‖β‖`) is
    invisible to every method and sets an error floor.
- The response is `y = ∫X_latent β + ε`, where `X_latent` is the noise-free
  curve. The noise `ε` is Gaussian with variance `S(1-R²)/R²`, and the signal
  variance is `S = Σ λ_k ⟨φ_k, β⟩²`. Inner products with `β` use the
  trapezoidal rule on 20,001 points, with error below `1e-8`.
- Each replicate uses `MersenneTwister(seed+replicate-1)` for the training
  sample, then the held-out sample, then the response noise. The
  configuration generator records each cell's starting seed. Julia's RNG
  stream depends on the Julia version, so reproduce with the recorded
  version. There are no RNG-dependent golden assertions.
- Every call uses the package defaults, apart from the sparse presmoother
  and the BIC route above. The defaults and their reference sources are
  listed on the
  [Regression page](../../docs/src/regression.md). The harness adds no
  estimator. On noisy dense curves, default dense FPCA does not smooth, so
  `FVE(0.99)` keeps noise components. This is the default behaviour under
  study, not a harness choice.
- Coefficient error is `‖β̂-β‖/‖β‖`, in the L2 norm under trapezoidal weights.
  It is computed on the scoring grid: the covariate grid for dense designs,
  and for sparse designs 101 equally spaced points over the observed
  training range, where sparse fits define `β̂`.
- Prediction error is the mean squared difference between predictions for
  the 200 held-out subjects and their noise-free mean response, divided by
  `S`. Held-out curves follow the training design. Sparse held-out times are
  drawn inside the observed training range, because PACE prediction is
  defined only there. Sparse predictions therefore include the irreducible
  error of predicting from 2–10 noisy points.
- Bands are each method's default `coefficient_bands` at `alpha=0.05`,
  evaluated on the scoring grid:
  - FPC: conditional Student-t intervals given the FPCA.
  - Penalized: refund `coef.pfr` normal intervals from mgcv's `Vc`, or `Vp`
    when `Vc` is unavailable (the recorded `band_covariance`).

  Pointwise coverage is the fraction of grid points covered. Neither method
  offers a simultaneous band. `whole_curve_coverage` records whether the
  pointwise band covered `β` at every grid point; it is not a simultaneous
  coverage claim. FPC bootstrap bands (`method=:bootstrap`, 999 FPCA refits)
  are not run.
- Complexity: `ncomponents` is the FPC component count, and `term_dof` is
  that count or the penalized term's EDF. λ is recorded for `pfr`.
- `check.jl` checks the harness against known answers, with no fits:
  analytic projections, signal variances and offsets; the Gaussian mass via
  the error function; an oracle fit with zero error and full coverage;
  shifted and half-covering bands; failure retention; and paired sampling.
  `test_harness.py` checks work lists, the committed grid, launcher syntax,
  and summary rejection of adverse result directories.
- Each route runs in its own Julia child process over its replicate block.
  A warmup with 30 subjects and seed 20261026 compiles the fit and metrics
  before timing. Garbage collection runs before each fit, and one Julia
  thread and one BLAS thread are used. `@timed` measures the complete public
  `fit` call, including FPCA and smoothing selection. Simulation, metrics,
  bands, prediction, startup and warmup fall outside the timer.
  `allocated_bytes` is total GC-tracked allocation during `fit`.
  `process_peak_rss_bytes` is the worker's cumulative `Sys.maxrss()`, which
  includes startup and JIT. The summary reports its maximum.

## Local smoke recreation

```sh
julia --project=. benchmark/regression/check.jl
(cd benchmark/regression && python3 -B -m unittest test_harness)
python3 benchmark/regression/configs.py /tmp/regression-smoke.csv --smoke
julia --project=. benchmark/regression/run.jl /tmp/regression-smoke.csv 1 /tmp/regression-smoke/cell-001.csv
julia --project=. benchmark/regression/run.jl /tmp/regression-smoke.csv 2 /tmp/regression-smoke/cell-002.csv
python3 benchmark/regression/summarize.py /tmp/regression-smoke.csv /tmp/regression-smoke /tmp/regression-smoke-summary
```

The smoke grid has one dense cell (`leading`, n=40, R²=0.8) and one sparse
cell (`smooth`, n=40, R²=0.5). Each has 20 held-out subjects, two replicates
and all three routes. `smoke/run.json` records the local validation and
recipe. Full-study results belong in `results/`.

## Discovery work units

`work_units.py` creates a complete, deterministic TSV list with one row per
cell, route and inclusive replicate block, for example
`cell-001-pfr-0001-0050.csv`:

```sh
python3 benchmark/regression/work_units.py benchmark/regression/configurations.csv /path/to/work-units.tsv --block-size 50
python3 benchmark/regression/work_units.py benchmark/regression/configurations.csv /path/to/work-units.tsv --check
```

The default block is 50 replicates, which gives 288 units. Locally, one fit
on the largest (sparse, n=500) cells took about 0.5 s. Worker startup and
warmup take about 20 s and are repeated for every unit. Larger blocks
amortize startup; smaller blocks lose less work to an interruption. Keep the
configurations and the 200 replicates unchanged.

`discovery.sbatch` runs GNU Parallel inside one Slurm allocation, launching
exclusive one-CPU `srun` steps through `worker.sh`. Set `JULIA_MODULE`,
`FDA_PROJECT`, `REGRESSION_CONFIG`, `REGRESSION_UNITS` and
`REGRESSION_OUTPUT` to site-specific absolute paths and module names. Load
GNU Parallel beforehand or through the site environment. Instantiate the
project before submission; jobs do not install dependencies. The template
contains no account names or credentials. As starting requests it asks for
one node, one CPU, 16 GB and four hours. Set the concurrency with
`sbatch --ntasks=N` and keep `--cpus-per-task=1`. `REGRESSION_WORKERS` can
lower concurrency below `SLURM_NTASKS`. Each step requests
`REGRESSION_STEP_MEM` (default 1900M; local worker peaks were about 1.3 GB).
These requests are not a validated budget.

GNU Parallel writes each unit's exit status to
`REGRESSION_OUTPUT/parallel-joblog.tsv`. An aborted unit stops new launches,
and the driver exits nonzero once active units finish. Counted fit and
metric failures are completed attempts and do not trigger a rerun. The
driver saves the configuration and work list next to the joblog and rejects
changes to them on restart. Only one allocation may use a results directory
at a time.

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
sbatch --parsable --job-name="regression-RUN_TAG" --export=ALL benchmark/regression/discovery.sbatch
```

Before summarizing, review the job accounting, the joblog and complete
result coverage. Then use the summary to fill the "Choosing a regression
method" note on the Regression page.
