# Naming and development dependencies

Types and modules use `UpperCamelCase`. Descriptive functions, fields, keyword
arguments and private helpers use `snake_case`; mutating functions end in `!`.
Keep established numerical abbreviations (`nobs`, `atol`, `rtol`), mathematical
identifiers (`X`, `M`, `Σ`, `λ`) and names required by external interfaces.

Develop and test against local `master` checkouts of workspace dependencies.
Retain package UUIDs and version numbers during this migration. Use canonical
names in new code; deprecated bindings remain available for downstream users.
Frozen reference data and recorded numerical results are not renamed.

From the package root, run the complete suite with:

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
