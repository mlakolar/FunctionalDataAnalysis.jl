# Interface

The package implements grid, irregular, basis and multivariate functional
data, smoothing, decompositions and uncertainty summaries. Curves are rows;
components and evaluation points are columns where a matrix is returned.

## Fitting and prediction

`fit` extends StatsAPI. Prefer `project_scores` for score-only projection and
`predict_scores` for explicit score predictions with available uncertainty. Per-curve smoothers return functional
data directly, while pooled smoothers return dedicated mean/covariance
summaries and decompositions return model objects. The table records the
current legacy `StatsAPI.predict` contracts, retained with deprecation warnings
for FPCA, MultivariateFPCA and MFPCA. CCA variate prediction is unchanged. Here `n` is the new subject count, `K` a selected
component count, and `K_j` the selected count for node `j`.

| `StatsAPI.fit` call | Return value | `StatsAPI.predict` result |
|:--|:--|:--|
| `fit(LocalPolynomial, fd; grid, ...)` | `GridFunctionalData` | No method; use `evaluate` with an explicit evaluator. |
| `fit(PenalizedSpline, fd; basis, ...)` | `BasisFunctionalData` | No method; use exact `evaluate`. |
| `fit(SmoothedMean, fd; grid, ...)` | `SmoothedMean` | No method; `evaluate(mean, t)` returns a vector (or a scalar for one time). |
| `fit(SmoothedCovariance, fd, mean; ...)` | `SmoothedCovariance` | No method; use `evaluate` for the surface. |
| `fit(CrossCovariance, fd1, fd2)` | `CrossCovariance` | No method; use `evaluate` for the surface. |
| `fit(SmoothedCrossCovariance, fd1, fd2, mean1, mean2; ...)` | `SmoothedCrossCovariance` | No method; use `evaluate` for the surface. |
| `fit(FPCA, complete_grid; method=:dense, ...)` | `FPCA` | `n × K` score matrix; requires the complete training grid. |
| `fit(FPCA, grid_or_irregular; method=:pace, ...)` | `FPCA` | Named tuple `(scores, covariances)`, sizes `n × K` and `K × K × n`. |
| `fit(FPCA, grid; method=:face, ...)` | `FPCA` (common-grid FACE) | The same named-tuple shapes; requires the training grid, allowing masks. |
| `fit(FPCA, irregular; method=:face, ...)` | `FPCA` (sparse FACE) | The same named-tuple shapes; accepts grid or irregular observations inside the work grid. |
| `fit(FPCA, basis_data; ...)` | `FPCA` (exact basis representation) | `n × K` score matrix; requires the training basis. |
| `fit(FPCA, multivariate; basis=:marginal, ...)` | `MultivariateFPCA` | A vector in node order: each entry is its univariate score matrix or `(scores, covariances)` named tuple, with count `K_j`. |
| `fit(FPCA, multivariate; basis=:pooled, ...)` | `MultivariateFPCA` | A vector of `n × K_j` score matrices; requires complete training grids. |
| `fit(MFPCA, multivariate; ...)` | `MFPCA` (joint multivariate components) | One `n × K` score matrix. |
| `fit(FSVD, fd1, fd2; ...)` | `FSVD` | No prediction method; paired training scores are available through accessors. |
| `fit(FunctionalCCA, basis1, basis2; ...)` | `FunctionalCCA` | Named tuple `(variates1, variates2)`, each `n × K`. |

For raw univariate data, `method=:auto` selects dense FPCA for complete
common-grid data and PACE for irregular or masked data. `method=:face` is
explicit: grid input selects common-grid FACE; irregular input selects sparse
FACE. It is not chosen automatically by sampling density. See
[FPCA](fpca.md), [multivariate FPCA](mfpca.md),
[cross-structure](cross_structure.md) and [CCA](cca.md) for route-specific
settings and restrictions. Derivative results come from `derivative(model, ...)`,
rather than `fit`; they support reconstruction as described under
[derivatives](fpca.md#Derivatives).

## Explicit score prediction

| Route | `project_scores(model, data)` | `predict_scores(model, data)` |
|:--|:--|:--|
| Dense FPCA | `n × K` matrix | `ScorePrediction(matrix, nothing)` |
| Exact-basis FPCA | `n × K` matrix | `ScorePrediction(matrix, nothing)` |
| PACE | `n × K` matrix | `ScorePrediction(matrix, covariance)` with `K × K × n` covariance |
| Common-grid FACE (complete or masked) | `n × K` matrix | Same result type and covariance dimensions as PACE |
| Sparse FACE | `n × K` matrix | Same result type and covariance dimensions as PACE |
| Marginal MultivariateFPCA, including mixed routes | Vector of `n × K_j` matrices | Vector of node `ScorePrediction`s |
| Pooled MultivariateFPCA | Vector of `n × K_j` matrices | Vector of `ScorePrediction(matrix, nothing)` |
| Joint MFPCA | `n × K` matrix | `ScorePrediction(matrix, nothing)` |

`ScorePrediction` has named `scores` and `covariances` fields. It is not an
array and does not implicitly discard uncertainty. `nothing` means no
conditional score covariance is supplied; it does **not** assert zero
uncertainty. Joint MFPCA does not assume independent node errors or propagate
marginal covariances. PACE/FACE covariances retain their documented conditional
interpretation and exclusions. All prediction arrays are newly computed and
independent of training scores and input arrays. Modifying a result changes
that result only. There is no new FSVD prediction algorithm, and CCA retains
its separate `(variates1, variates2)` prediction contract.

Legacy `StatsAPI.predict(model, data)` keeps exactly its original matrix,
named tuple, or vector of alternatives and warns for the three FPCA model
types. Replace it explicitly with one of the two score APIs; no automatic
return-type switch is implied for a later release. Input requirements and
numerical calculations are shared with the legacy routes in the first table.

```julia
projected = project_scores(model, newdata)
prediction = predict_scores(model, newdata)
# prediction.scores is the matrix; prediction.covariances may be nothing.
```

## Statistical accessors

`StatsAPI.fitted(model)` returns `reconstruct(model)` for FPCA,
MultivariateFPCA and MFPCA: latent training curves on the stored work/display
grids with training IDs, without added noise or parameter uncertainty.
`StatsAPI.residuals(model)` subtracts those latent functions at the original
observations, before presmoothing. Grid masks, irregular times/offsets, IDs,
node order and exact basis representations are preserved. Grid/PACE/FACE
residuals use the existing piecewise-linear reconstruction; basis residuals
use exact coefficient arithmetic. Missing or imputed FACE values are not
promoted to observed residuals. Both accessors return independent data.

`StatsAPI.coef(data::BasisFunctionalData)` returns the stored curve-by-basis
coefficient matrix; copy it before mutation. There are no ambiguous model
`coef` methods, and no fitted-observation or residual methods for FSVD, CCA
or derivative summaries. Residuals contain truncation/smoothing error as well
as noise and are not independent noise estimates.

```@docs
ScorePrediction
project_scores
predict_scores
StatsAPI.fitted(::Union{FPCA,MultivariateFPCA,MFPCA})
StatsAPI.residuals(::FPCA)
StatsAPI.coef(::BasisFunctionalData)
```

## Counts, orders and array inputs

Counts and orders must be integers other than `Bool`, representable as `Int`,
within the argument's documented bounds. Zero remains valid for polynomial
and derivative orders where documented; positive counts and minimum-two
quantities retain their existing bounds. This includes component caps and
counts, CCA counts, bootstrap replicates, curve folds, bin counts, basis
sizes/orders, FACE intervals/degrees/difference orders and simulation counts.
`true` and `false` are rejected even though Julia makes `Bool <: Integer`.

One-based array views and ranges are supported; offset-indexed inputs remain
unsupported. Float32 smoke tests use relative tolerance `2e-3` and absolute
tolerance `2e-5` against Float64 calculations, including fully estimated
PACE and curve smoothing. These checks supplement the unchanged Float64 R
contracts below `1e-8`. KernelSmoothing currently computes curve-smoother
matrices in Float64; the cache retains those operators and multiplication
precision before converting output to the work-grid type. Float32 input/output
support does not imply that every backend workspace uses Float32.

## Tuning vocabulary

| Route / keyword | Meaning and default | Distinction / compatibility |
|:--|:--|:--|
| `LocalPolynomial.degree` | Polynomial degree; default `derivative + 1` | Derivative order is a separate quantity. |
| `BSplineBasis(order, ...)` | Spline order = polynomial degree + 1 | Cubic splines have order 4. This is also distinct from the derivative order in `PenalizedSpline(...; order=2)`'s roughness penalty. |
| Dense / sparse `face.knots` | Number of equal intervals, default 35 / 7 | Strictly interior knot count is `knots - 1`; basis dimension is `knots + p`, with spline degree `p=3` by default. This is not a generic interior-knot count. |
| CCA `ncan` / `ncomponents` | Number of canonical pairs, default 2 | Additive aliases; supply at most one. Unlike FPCA's `ncomponents`, this accepts a fixed count, not FVE. Stored settings retain `ncan`. |
| Sparse `face.lambda_mean` / `face.mean_loglambda` | Fixed mean penalty / log-candidate grid; defaults `nothing` / `range(-20,20; length=100)` | Candidates are exponentiated. Independent of covariance `face.lambda` / `face.loglambda`; a supplied mean or `center=false` cannot request mean tuning. |
| Dense `face.mean_smoother` | A `PenalizedSpline` smoother specification, or the documented default when `nothing` | The dense and sparse mean-smoothing mechanisms are distinct, not interchangeable keyword aliases. |
| `mean_bands(...; B, bootstrap_replicates)` | Number of whole-curve bootstrap resamples; default 1000 | Additive aliases; supply at most one. Both are bootstrap-only; the stored result field remains `B`. |
| `SmoothedCovariance.noise_method` / `noise_interval` | Default `:rotated` / `:full` | FPCA/PACE do not forward these options. Fit a covariance explicitly and supply it as shown in the [noise-options recipe](smoothing.md#Noise-options-for-PACE). |

Alias names change neither numerical defaults nor stored settings keys. See
[smoothing](smoothing.md), [FPCA](fpca.md), [CCA](cca.md) and
[uncertainty](uncertainty.md) for the route-specific controls and their sources.

## Generic ownership and storage

`nobs`, `fit`, `predict`, `reconstruct`, `fitted`, `residuals` and `coef` are
StatsAPI generics. StatsAPI
**1.8 or newer** is required for the shared `reconstruct` binding. Existing
FDA reconstruction calls and return values are unchanged; qualified imports
now identify the same generic:

```julia
import FunctionalDataAnalysis
import StatsAPI: fit, predict, reconstruct
# reconstruct(model) == FunctionalDataAnalysis.reconstruct(model)
```

`evaluate` remains owned by FunctionalDataAnalysis. When also loading
MultivariateStats or another package exporting that name, use
`FunctionalDataAnalysis.evaluate(data, times)` or import it explicitly.
`eigvals` and `svdvals` extend LinearAlgebra's generics.

FSVD exposes `svdvals`, `fve`, `left_functions`, `right_functions`,
`left_scores`, `right_scores`, `nobs` and `ncomponents`. CCA exposes
`canonical_weights`, `canonical_variates`, `canonical_correlations`,
`squared_canonical_correlations`, `nobs` and `ncomponents`.
Array accessors return the fit's own storage; modifying that storage changes
the fit. Use `copy` for arrays and `deepcopy` for basis-data weights. The
CCA paired accessors return named tuples with `left` and `right` entries
referencing the stored objects. Accessor docstrings give dimensions and
ownership; CCA's penalized correlations generally differ from empirical
correlations of its variates.

## Data interface

```@docs
FunctionalDataAnalysis.AbstractFunctionalData
FunctionalDataAnalysis.AbstractBasis
FunctionalDataAnalysis.nobs
FunctionalDataAnalysis.domain
FunctionalDataAnalysis.evaluate
FunctionalDataAnalysis.sampling_design
FunctionalDataAnalysis.quadrature_weights
```
