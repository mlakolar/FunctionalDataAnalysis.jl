# Long-grid FPCA benchmark

This harness compares the package's dense and FACE routes on common grids,
and PACE and sparse FACE on 2–10 observations per curve. It is outside the
test suite. The full study runs on USC Discovery. The two-cell `smoke/`
checks only that the harness works; its timings are not benchmark evidence.
No package defaults change, and no crossover grid length is claimed before
the Discovery results are available.

## Configuration and recreation

`configurations.csv` has 20 paired cells and 200 replicates per cell:

| Design | Subjects | Grid length | Noise variance / integrated signal variance | Routes |
|:--|:--|:--|:--|:--|
| Dense | 100, 1000 | 100, 500, 2000, 5000 | 0.05, 0.20 | dense, FACE |
| Sparse | 100, 1000 | 101 work/evaluation points | 0.05, 0.20 | PACE, sparse FACE |

This gives 8,000 fitted models. Both routes in a cell receive the same
sample in each replicate, recreated from the same seed. No extra Julia
dependencies are needed; Python scripts use only the standard library.
Instantiate the project, including its KernelSmoothing `[sources]` entry,
before running. On Julia versions that do not support `[sources]`, install
KernelSmoothing from that URL as in package CI. Do not commit the Manifest.

From the repository root, using new output paths:

```sh
python3 benchmark/scaling/configs.py /tmp/scaling-configurations.csv
julia --project=. benchmark/scaling/run.jl /tmp/scaling-configurations.csv 1 /path/to/results/cell-001.csv
python3 benchmark/scaling/summarize.py benchmark/scaling/configurations.csv /path/to/results /path/to/summary
```

The last command requires all configured cells. `run.jl` takes a CSV, a
one-based row index, and an output path, with optional inclusive replicate
bounds and an optional route:

```sh
julia --project=. benchmark/scaling/run.jl CONFIG.csv INDEX OUTPUT.csv [FIRST LAST [ROUTE]]
```

Omitting the bounds runs the whole cell. Bounds alone run both routes over
that block; the final argument selects `dense` or `face` for dense cells,
or `pace` or `face_sparse` for sparse cells. Replicate seeds use their original
indices, so splitting the work does not change the simulated samples.
Each invocation writes one CSV result row with its configuration, replicate
bounds, routes, versions, elapsed time, per-route outcome counts, resource
summaries, and metric means/counts. Companion `.replicates.csv`
and `.failures.csv` files retain every attempt, seed, selected tuning values,
grid endpoints, population eigenvalues, and informative numerical errors.
Allocated bytes and wall time remain recorded for failed fits. Every
`Exception` from a fit or metric computation counts as a failure, with its
exception type and diagnostic stage, except `InterruptException` and
`OutOfMemoryError`, which abort the worker. Failures are never replaced.
Interrupted or aborted blocks have companions but no final result row.

Existing outputs are rejected. The summary merges whole-cell and chunk
outputs named `cell-*.csv` in one results directory, and requires exactly
one retained attempt for every configured route/replicate. Companion files
are not mistaken for result rows. For recovery, preserve completed blocks
and interrupted attempts; the Discovery instructions below describe explicit
restart steps. The summary rejects missing/duplicate records, mismatched
seeds/settings/versions, inconsistent aggregate rows, or missing diagnostics.
Quality means and sample SDs are conditional on success, with failure counts
shown. Resource summaries include failed attempts. Dense noise-variance
error has count zero because dense FPCA does not estimate that quantity.

## Scientific and measurement choices

- `simulate_kl` generates mean `1+t`, the first three normalized Fourier
  functions on `[0,1]`, independent Gaussian scores with variances
  `[1,0.5,0.25]`, and independent Gaussian noise. True sigma2 is
  `ratio*1.75`. These are fixed benchmark choices, not estimator defaults.
- Dense grids are equally spaced, including 0 and 1. Sparse counts are
  uniform on `2:10`, with sorted uniform times on `[0,1]`. Both sparse routes
  receive the same explicit 101-point grid spanning the observed extrema.
  This overrides their different native work-grid counts (PACE 51, sparse
  FACE 100) to compare the same numerical target. There is no extrapolation.
- Each replicate uses `MersenneTwister(seed+replicate-1)`. The configuration
  generator records every starting seed. Julia's RNG stream is version
  dependent; reproduce with the recorded Julia version. There are no
  RNG-dependent golden assertions. Seeds pair routes within a cell; samples
  in different cells are independent.
- Calls otherwise use the existing route defaults, including `FVE(0.99)`
  and each route's own `maxK`. Dense FPCA uses the sample covariance with
  denominator `n-1`, with no presmoothing, so its noisy-data accuracy describes
  that baseline. FACE uses its default mean and GCV covariance smoother;
  sparse FACE uses subject CV for mean/covariance and joint noise estimation.
  PACE uses Gaussian local linear smoothers, bandwidths 0.05/0.10 times the
  observed range, observation weighting, default noise estimation, and
  automatic binning (inactive for counts at most 10). No bandwidth search,
  fixed truth component count, supplied noise variance, or oracle mean is
  added. Default sources and departures are audited on the
  [FPCA page](../../docs/src/fpca.md): refund **0.1-40**, face **0.1-8**,
  fdapace **0.6.0**. This harness adds no reference estimator implementation.
- Population eigenpairs are computed from a three-dimensional Gram matrix
  of the known KL functions under the fit's trapezoidal weights. This is the
  exact sampled population operator, including restriction to the sparse
  observed range. It avoids conflating statistical error with quadrature
  or support differences and never allocates an `m` by `m` covariance.
  The relative eigenvalue error is the Euclidean error of the first three
  selected eigenvalues divided by the norm of the three population values.
  The eigenfunction error is the mean of three weighted L2 distances after
  sign alignment. Missing selected eigenpairs are zero, so omission incurs
  error rather than dropping a difficult component. Extra selected components
  are reflected in the recorded component count and reconstruction error.
- Reconstruction error is the mean integrated squared error of the selected
  training reconstruction against the latent noise-free curves evaluated
  from the simulated scores. It measures recovery of the observed subjects,
  not prediction on held-out subjects. Sigma2 error is signed estimate minus
  truth, in measurement units. `check.jl` verifies metrics against analytic
  Fourier covariance, an exact oracle, reversed signs, and omitted components.
- Each selected route runs in a fresh Julia child process through the selected
  replicate block.
  A tiny warmup with the same design and estimator defaults precedes timing.
  It uses 12 curves, 40 dense or 21 sparse work-grid points, noise ratio 0.20,
  and seed 20261025, independently of the scored configuration.
  Garbage collection runs before each fit, with prior fits released. One Julia
  thread and one BLAS thread are used. `@timed` measures the complete public
  `fit` call, including input copying, smoothing selection, eigenanalysis and
  training scores. Simulation, truth construction, accuracy metrics, process
  startup, and warmup are outside that fit timer.
- `allocated_bytes` is total GC-tracked allocation during `fit`, not peak live
  memory. `process_peak_rss_bytes` is `Sys.maxrss()`, the cumulative resident
  high-water mark for that route's worker, including startup, JIT, warmup,
  simulation and metrics. The summary reports its maximum across replicates,
  never a per-fit incremental peak or a sum across methods. This peak and
  fit allocations are separate memory measurements. A chunk's elapsed time
  additionally includes child startup/warmup and all replicate bookkeeping.
  Splitting into more chunks repeats startup and warmup; those remain outside
  per-fit timings. The summary reports the largest process peak across chunks.

## Local smoke recreation

```sh
julia --project=. benchmark/scaling/check.jl
python3 benchmark/scaling/configs.py /tmp/scaling-smoke.csv --smoke
julia --project=. benchmark/scaling/run.jl /tmp/scaling-smoke.csv 1 /tmp/scaling-smoke/cell-001.csv
julia --project=. benchmark/scaling/run.jl /tmp/scaling-smoke.csv 2 /tmp/scaling-smoke/cell-002.csv
python3 benchmark/scaling/summarize.py /tmp/scaling-smoke.csv /tmp/scaling-smoke /tmp/scaling-smoke-summary
```

The smoke grid contains one dense cell (`n=12,m=40,ratio=0.05`) and one
sparse cell (`n=12`, 21 evaluation points, `ratio=0.20`), with one replicate
and both routes in each. The dense grid retains full rank for FACE's default
38-function basis. `smoke/run.json` records the local validation and recipe.
Full-study results belong in a separate owner-selected directory.

## Discovery work units

`work_units.py` creates a complete deterministic TSV list, with one row per
cell, route and inclusive replicate block. Canonical output names include
all three, for example `cell-001-dense-0001-0010.csv`. Generate it before
submission, using new paths:

```sh
python3 benchmark/scaling/work_units.py benchmark/scaling/configurations.csv /path/to/work-units.tsv --block-size 10 --route-block-size dense=1
python3 benchmark/scaling/work_units.py benchmark/scaling/configurations.csv /path/to/work-units.tsv --check
```

The default block size is 10 replicates. `--route-block-size ROUTE=COUNT`
overrides it for one route, and repeatable `--cell-block-size CELL:ROUTE=COUNT`
overrides both for one cell/route. For example, `--cell-block-size 15:dense=1`
splits that long-grid dense route into single-replicate invocations without
changing the configuration. The last block can be shorter. The checker rejects
missing or overlapping replicate coverage and mismatched routes or names.

Choose block sizes using Discovery timings and memory use. A block should
fit comfortably within the allocation's remaining wall time; long-grid dense
eigendecompositions may need a block of one. Larger blocks amortize Julia
startup and warmup and create fewer files, while smaller blocks limit work
lost to interruption and improve load balancing. If one replicate exceeds a
worker's available time or memory, change the allocation or schedule that
unit separately; splitting cannot make an individual fit smaller. Keep the
scientific configurations and 200 replicates unchanged.

`discovery.sbatch` uses GNU Parallel inside one Slurm allocation, launching
exclusive one-CPU `srun` steps through `worker.sh`. Set `JULIA_MODULE`,
`FDA_PROJECT`, `SCALING_CONFIG`, `SCALING_UNITS`, and `SCALING_OUTPUT` to
site-specific absolute paths/module names. Load GNU Parallel beforehand or
through the site's environment. Instantiate the project before submission;
jobs do not install dependencies. No account names or credentials are in
the template. It requests one node, one CPU, 16 GB and two hours as starting
resource requests. Set the approved concurrency with `sbatch --ntasks=N`
and retain `--cpus-per-task=1`; `SCALING_WORKERS` may reduce concurrency below
`SLURM_NTASKS`. Every Julia and BLAS worker uses one thread. The Julia
supervisor waits for its child, so each step has one active fitting process.

The `srun --exclusive --exact -N1 -n1 -c1 --mem=0` steps use distinct CPUs
and share the allocation's memory limit. Slurm documents these step resource
options in the [srun reference](https://slurm.schedmd.com/srun.html).
Choose concurrency so the sum of active workers' peak memory fits the
allocation; shared memory does not impose an independent per-worker limit.
These requests are not a validated full-study budget or a guarantee of
completion. Mladen sizes resources on Discovery and authorizes the full
campaign before submitting it.

GNU Parallel records each work unit's exit status in
`SCALING_OUTPUT/parallel-joblog.tsv`; an aborted unit stops new launches and
the driver exits nonzero after active units finish. Counted fit/metric failures
are completed benchmark attempts, so they do not trigger a rerun. The driver
saves the small configuration and work-list inputs alongside the joblog and
rejects changes to them on restart. Freeze the source and environment for a
run as recorded below. Only one allocation may use a results directory at a
time.

To resume an infrastructure-aborted run, keep the same output directory,
work-list order and joblog. First preserve every aborted chunk's existing
files in a separate attempt directory. A unit killed by the wall-time limit
may have no joblog row, so identify aborted units by their files instead: an
aborted unit has a `.replicates.csv` companion but no result row.

```sh
cd /path/to/results && mkdir aborted-attempt-1
for f in cell-*.replicates.csv; do b=${f%.replicates.csv}; [[ -e $b.csv ]] || mv "$b".* aborted-attempt-1/; done
```

Then submit the same command and inputs again. The driver passes
`--resume-failed` when the joblog exists, so GNU Parallel reuses successful
units and launches failed or unfinished units. `run.jl` refuses to overwrite
any remaining result or companion file; preservation is an explicit recovery
step. Keep the aborted attempts as evidence. A restarted block recomputes any
finished replicates inside that aborted block with their original seeds; use smaller
blocks to bound that lost work. Do not rerun completed blocks to replace
unfavorable fit/metric outcomes, or combine duplicate attempts. After changing
block sizes or the work-list order, use a fresh directory and joblog instead
of resuming the old sequence numbers.

Keep a small run record alongside results: run ID/purpose, source commit and
any dirty patch, configuration, Julia/dependency versions and KernelSmoothing
revision, exact launch command, requested resources, submission intent,
returned job ID, outputs and checks. Record intent before submission, use a
unique run tag in the job name, and save the returned ID:

```sh
sbatch --parsable --job-name="scaling-RUN_TAG" --export=ALL benchmark/scaling/discovery.sbatch
```

Review the exact job accounting, GNU Parallel joblog and complete result
coverage before summarizing.
Scheduler completion alone does not establish successful fits. Retain final
results and the recreation recipe; this task includes no cluster submission,
resource expansion, remote archives, or cleanup. Once the Discovery summary
exists, use it to fill the FPCA page's route-selection note, reporting accuracy
and failure rates alongside time and memory. Any default change is Mladen's
separate decision.
