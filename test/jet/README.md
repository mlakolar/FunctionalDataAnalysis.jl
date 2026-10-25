# Bounded JET regression baseline

This check analyzes eleven concrete Float64 workloads from the October audit.
It is not package-wide inference certification. CI uses Julia **1.13.2** and
JET **0.13.0**, without unsupported-Julia preferences. The runner fails if JET
is unavailable. Each workload must also execute successfully before analysis.
Production code, numerical tolerances and workspace dependency policy are
unchanged: KernelSmoothing and other workspace packages still use master
checkouts, not an immutable reference chosen by this check.

## Reviewed initial baseline

The source is master `58c2d0c` (the #50 merge). Discovery job **13450309**
recorded diagnostics from the test-runner commit `2e00980`, using Julia 1.13.2,
JET 0.13.0, BSplineKit 0.19.3 and the campaign's saved KernelSmoothing master
snapshot `e8a2d231`. CI resolves current dependency masters independently;
this snapshot describes the initial evidence, not a CI dependency pin.

| Workload | Optimization reports | Call-analysis reports |
| --- | ---: | ---: |
| Mean | 0 | 0 |
| Complete-grid curve smoothing | 0 | 0 |
| Dense FPCA | 1 | 0 |
| Fixed-bandwidth PACE | 0 | 0 |
| Common-grid FACE | 2 | 0 |
| MFPCA | 24 | 0 |
| Dense FSVD | 2 | 0 |
| FPC derivative | 0 | 0 |
| Automatic sparse FACE | 4 | 0 |
| Fixed Fourier basis FPCA | 1 | 0 |
| Fourier CCA | 0 | 1 |

Every accepted site, including duplicates, is listed in `baseline.toml`. The
initial reports were reviewed as follows; counts describe this exact compiler,
source and workload, not runtime costs or comparisons with older audit counts.

- `multivariate.jl:34` is abstract metadata/ID forwarding into a copied grid
  container. It appears in dense FPCA, FACE, FSVD and MFPCA. The additional
  `smoothed_fpca.jl:262` basis-FPCA report is the analogous basis-data boundary.
- `bases.jl:189` is runtime derivative-order dispatch into BSplineKit
  collocation despite a concrete spline backend. FACE and sparse FACE report
  it directly. Dense FSVD's analysis explores FACE marginal branches although
  its exercised runtime workload uses dense marginals.
- The other sparse-FACE sites (`face_sparse.jl:242,261,264`) are heterogeneous
  automatic-mean settings, their score-step dispatch and model construction.
- MFPCA has sixteen dispatch reports: heterogeneous node `nobs`, metadata,
  B-spline construction/collocation, and marginal uncertainty/basis/mean
  conversions (`multivariate.jl:34,59`, `psplines.jl:28,29,32`,
  `bases.jl:189`, `multivariate_fpca.jl:95,98,101`). Eight further reports are
  **optimization recursion limits** on explored sparse-FACE/marginal call
  paths (`face_sparse.jl:61,100,140`, `fpca.jl:497`,
  `multivariate_fpca.jl:66,88`). These mark the analyzer's frontier, not eight
  observed infinite recursions. All are retained rather than globally ignored.
  Narrowing the heterogeneous public containers remains deferred in #33.
- CCA's call report at `smoothed_fpca.jl:371` concerns ID equality:
  `Vector{Any} == Vector{Any}` can yield `Missing`, which is not Boolean.
  This workload has `ids === nothing` on both sides and executes successfully.
  The baseline retains the potential error; it does not certify arbitrary ID
  contents or declare the report a universal false positive. ID-domain policy
  is outside this CI change.

## Filtering and regression rule

Both `report_opt` and `report_call` use
`target_modules=(FunctionalDataAnalysis,)`, matching the audit. Reports whose
error point belongs only to Base, stdlibs, JET or external dependencies are
excluded as dependency noise. Dispatch **from FDA into a dependency remains
visible**, including the BSplineKit sites above. There is no whole-module FDA
suppression, catch-and-succeed behavior or blanket allowance for call errors.
Unresolved dispatch and recursion limits restrict how far analysis can see.

The baseline compares the exact sorted multiset of report type, innermost
source basename and line. Replacing a report with a different report cannot
hide behind an unchanged count. Added, removed, moved and duplicated sites all
fail and require review. Full signatures/stacks are printed and retained in
`jet-<run-id>`; source locations alone do not replace reading those diagnostics.

## Updating the baseline

Use the same runtime, then explicitly record a candidate:

```sh
FDA_JET_RECORD_BASELINE=true julia scripts/test_master.jl --jet
julia scripts/test_master.jl --jet
```

The first command writes `test/jet/baseline.toml`; CI never sets that flag.
Review the complete diagnostics and every changed site, explain any accepted
change in the PR, update this classification, and run the second command in
an independent process. Do not replace the baseline solely to make CI green.
Keep the eleven target names intact unless the PR explicitly changes scope.
Compiler/JET upgrades require the same review. Test-tool versions are isolated
in `test/jet/Project.toml`; ordinary tests continue to support Julia 1.10+.
