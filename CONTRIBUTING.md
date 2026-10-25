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
