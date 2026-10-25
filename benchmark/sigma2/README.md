# Noise variance benchmark (issue #17)

This harness is outside the package test suite. It compares five estimators on
the **same sample in each replicate**: unrotated surface and rotated diagonal,
each on the middle half and full interval, plus a benchmark-only historical
FGM projected-diagonal reference. The [Discovery results](results/README.md)
support the package's rotated/full default. Source audits and the estimator
report are in `docs/src/fdapace.md` and
`docs/src/m5_sigma2_report.md`.

## Configuration and outputs

`configurations.csv` has 48 cells: sparse/dense × n=100/400 × variance
ratios 0.05/0.2/0.5 × obs/subj × default/CurveCV. Each cell evaluates all five
estimators for **500 replicates** (240 estimator cells, 120,000 estimates).
`configs.py` recreates that CSV. No extra Julia dependencies are needed;
Python scripts use only its standard library. Run from the repository root
after instantiating the Julia project, including its KernelSmoothing source.

```sh
python3 benchmark/sigma2/configs.py /tmp/sigma2-configurations.csv
julia --project=. benchmark/sigma2/run.jl /tmp/sigma2-configurations.csv 1 /tmp/sigma2-results/cell-001.csv
```

`run.jl` takes a CSV, a one-based row index, and an output path. It writes
**one CSV result row**, containing Welford means/M2, sums of squared errors,
success/truncation counts, settings, versions, and elapsed seconds for all five
estimators. Companion `.replicates.csv` files retain every raw and truncated
estimate, data seed, selected mean/covariance bandwidth, CV scores, and curve
fold assignments. `.failures.csv` records informative fitting errors. A failed
replicate is never replaced; its estimates are NaN and its counts remain in
the failure denominator. Shared preparation failures affect all five estimators.
Unexpected programming errors abort the task. Existing outputs are rejected.
An interrupted task leaves companions without a final result row; rerun it
to a fresh output path after recording the interruption.

After all 48 tasks finish:

```sh
python3 benchmark/sigma2/summarize.py benchmark/sigma2/configurations.csv /path/to/results /path/to/summary
```

The script requires exactly one result row for each configured cell, checks
its settings and counts, and writes `summary.csv` and `summary.md`. Bias and
MSE use the zero-truncated estimate. Variance has divisor `R-1`; MSE divides
the squared-error sum by `R`. With failures, moments are conditional on success,
and the table explicitly reports the failure count. Missing tasks raise an
error rather than disappearing from the summary.

## Scientific choices fixed for this harness

- `simulate_kl` uses mean `1+t`, the first three normalized Fourier functions
  on `[0,1]`, and Gaussian score variances `[1,.5,.25]`. The ratio is
  **noise variance / integrated signal variance**, so true sigma2 is
  `ratio*1.75`. Noise is independent Gaussian. These process choices were
  unspecified by the task; they are benchmark settings, not package defaults.
- Sparse curves have a uniform count from 2:6 and sorted uniform times.
  Dense curves have 50 common equally spaced times. Each replicate uses a
  `MersenneTwister(seed+replicate-1)` passed to `simulate_kl`. Configurations
  with the same design/n/ratio share seeds across weighting/bandwidth choices.
  Julia's RNG stream is version-dependent; reproduce with the recorded Julia
  version. The CSV records seeds; there are no RNG-dependent golden assertions.
- The work grid has 51 points over the observed range. Kernel is Gaussian;
  fixed mean/covariance bandwidths are 5%/10% of that range. These are the
  source-audited fdapace 0.6.0/PACE defaults, **not** SmoothedCovariance's
  lower-level Epanechnikov default. No times are snapped and no pairs dropped.
- CurveCV selects both mean and covariance bandwidths. Candidate fractions are
  `[.025,.05,.1]` for the mean and `[.05,.1,.2]` for covariance. Five balanced,
  shuffled curve folds use `MersenneTwister(data_seed+1000000)` for both
  selectors; covariance CV holds the fitted full-sample mean fixed, as the
  package does. Candidates/fold count are explicit benchmark choices, not R
  GCV defaults. Ties select the first candidate. Every candidate must succeed.
- The raw-square smoother inherits the chosen covariance bandwidth. All five
  estimators share the same mean, raw products, unrotated surface and raw-square
  fit. Rotated estimators use their own diagonal regressions at the same
  covariance bandwidth and fdapace's retained-grid/nominal-length rule.
- `projected_fgm` reuses the package weighted eigen step with all positive
  eigenvalues above `100eps(Float64)` times the absolute spectral maximum,
  then averages excess raw squares at indices `ceil(q/4):floor(3q/4)` and
  truncates at zero. This reproduces the audited historical FGM diagonal rule
  on shared smoothed inputs, not FGM's entire pipeline. No PSD package option
  or extra K choice is introduced.
- Exact duplicate coordinates are aggregated with weighted means, weights,
  and within-cell squared errors. Training and held-out data are aggregated
  separately within each CV fold. This is exact sufficient-statistic
  compression, not approximate binning. It is needed to make dense CV
  practical. `check.jl` compares grouped losses and all four integrated
  estimators to public package fits and independently checks PSD reconstruction.

## Local smoke recreation

Use new paths, because existing results are protected:

```sh
julia --project=. benchmark/sigma2/check.jl
python3 benchmark/sigma2/configs.py /tmp/sigma2-smoke.csv --smoke
julia --project=. benchmark/sigma2/run.jl /tmp/sigma2-smoke.csv 1 /tmp/sigma2-smoke/cell-001.csv
julia --project=. benchmark/sigma2/run.jl /tmp/sigma2-smoke.csv 2 /tmp/sigma2-smoke/cell-002.csv
python3 benchmark/sigma2/summarize.py /tmp/sigma2-smoke.csv /tmp/sigma2-smoke /tmp/sigma2-smoke-summary
```

The committed `smoke/` contains two cells × 20 replicates: sparse/n=100/
ratio=.05/obs/default and dense/n=100/ratio=.2/subj/CV, each with all five
estimators. `smoke/run.json` is the lightweight local run record. This is
harness validation, not evidence for choosing a new default.

## USC Discovery array template

The requested one-task-per-configuration array is `discovery.sbatch`.
Set `JULIA_MODULE`, `FDA_PROJECT`, `SIGMA2_CONFIG`, and `SIGMA2_OUTPUT` to
site-specific values. The file has no account names or credentials. Instantiate
the environment before submission; it does not install packages inside jobs.
The owner must select the current site account/partition and review concurrency,
memory and time. The template requests one CPU, 2 GB and 24 hours per task, with
up to eight tasks at once. This is a planning template; no cluster job was
submitted during development. The full run belongs to Mladen.

For an authorized run, keep one small run record alongside results: purpose,
source commit (plus any dirty patch), configuration path, Julia and dependency
versions, exact launch command, resources, submission intent, returned array job
ID, outputs and checks. Record intent **before** submitting and use a unique
run tag in the job name. A possible owner-run launch after setting the variables:

```sh
sbatch --parsable --job-name="sigma2-RUN_TAG" --export=ALL benchmark/sigma2/discovery.sbatch
```

Save the returned ID in that record. Review exact task accounting and result
coverage before summarizing; scheduler completion alone is not validation.
For recovery, preserve successful tasks and rerun only missing/interrupted
indices to fresh paths, recording the attempt. Keep final results and the
recreation recipe; no routine archive or remote cleanup is part of this task.
