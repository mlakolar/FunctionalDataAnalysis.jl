# Naming and development dependencies

Types and modules use `UpperCamelCase`. Descriptive functions, fields, keyword
arguments and private helpers use `snake_case`; mutating functions end in `!`.
Keep established numerical abbreviations (`nobs`, `atol`, `rtol`), mathematical
identifiers (`X`, `M`, `Σ`, `λ`) and names required by external interfaces.

Develop and test against local `master` checkouts of workspace dependencies.
Retain package UUIDs and version numbers during this migration. Use canonical
names in new code; deprecated bindings remain available for downstream users.
Frozen reference data and recorded numerical results are not renamed.

From the package root, run the ordinary test suite with:

```sh
julia scripts/test_master.jl
```

The runner prepares `.master-test-env/v<major>.<minor>` with this package,
its test extras, and local master checkouts of workspace dependencies. Missing
workspace checkouts are cloned from master; existing checkouts must already be
on master. It supports Julia 1.10 and newer and keeps the package `Project.toml`
unchanged. CI uses the same runner.

Use `julia scripts/test_master.jl --setup-only` to prepare the environment for
focused checks; the runner prints the environment path to use with `--project`.

The ordinary suite includes deterministic band tests and a seeded smoke study
(4 datasets, 30 subjects, 19 bootstrap draws). Run the statistical coverage
regression separately with `julia scripts/test_master.jl --coverage-study`:
400 datasets, 300 subjects, 399 draws, and the original acceptance floors.
The `.90` mean-band floor allows the documented undercoverage; it does not
certify nominal 95% coverage. The larger recipe remains
`FDA_BANDS_FULL_COVERAGE=true julia scripts/test_master.jl --coverage-study`
(200 datasets, 300 subjects, 1,000 draws).

To run either study on one GitHub-hosted configuration and retain its log,
dispatch the existing CI workflow on the development branch:

```sh
gh workflow run CI.yml --ref master -f coverage_study=regression
gh workflow run CI.yml --ref master -f coverage_study=full
```

These dispatches run only the dedicated study job. Omit the input to run the
ordinary CI matrix. Manual dispatch is used while development targets `master`
and the repository's default branch remains `main`; a schedule would use the
default branch. The study outcome appears in the job log and a retained artifact.

## Bounded CI checks

The ordinary matrix explicitly uses one Julia thread and one BLAS/OpenMP thread.
One additional Ubuntu leg uses four Julia threads and one BLAS thread:

```sh
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    julia scripts/test_master.jl --threaded-smoke
```

It reuses the existing PACE and marginal/pooled multivariate R goldens at
relative error below `1e-8`, and the former-smoother comparisons at `1e-10`.
It includes no band/bootstrap coverage study or dataset download. This is a
bounded numerical smoke test, not certification of every threaded route.

The separate JET leg uses Julia **1.13.2** and JET **0.13.0**; its test-only
tool dependency is isolated from the ordinary test environment and package
`Project.toml`. Run `julia scripts/test_master.jl --jet` with that Julia build.
The eleven concrete workloads, report-site baseline, dependency filtering and
review procedure are described in [test/jet/README.md](test/jet/README.md).
An unavailable analyzer is an error, not a skipped success. Full diagnostics
are retained in the CI artifact. This runtime/tooling baseline does not pin
KernelSmoothing or alter the workspace dependency policy: all these jobs use
the same local/developed master checkouts as the ordinary suite.

Full dataset artifacts are tested only by an explicit manual dispatch:

```sh
gh workflow run CI.yml --ref master -f dataset_download=true
```

This runs only the download job. The loader uses an empty temporary artifact
depot and the shipped release URL (the package server is disabled for this
check), validates every full dataset against its committed R summaries, and
verifies the installed artifact tree hash. Existing cached artifacts cannot
make it pass without a download. It restores process state on failure and
retains the job log. Ordinary PR jobs continue to use committed samples only.
A manual trigger is intentional while development targets `master` and the
historical `main` remains the default branch. Use
`julia scripts/test_master.jl --dataset-smoke` to run the same check locally;
that command downloads the full artifact. Suite flags are mutually exclusive,
and each supports `--setup-only` for environment preparation.
