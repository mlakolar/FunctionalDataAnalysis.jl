# Interface

The package implements grid, irregular, basis and multivariate functional
data, smoothing, decompositions and uncertainty summaries. Curves are rows;
components and evaluation points are columns where a matrix is returned.

## Fitting and prediction

`fit` and `predict` extend StatsAPI. Per-curve smoothers return functional
data directly, while pooled smoothers return dedicated mean/covariance
summaries and decompositions return model objects. The table records the
current result contracts. Here `n` is the new subject count, `K` a selected
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

## Generic ownership and storage

`nobs`, `fit`, `predict` and `reconstruct` are StatsAPI generics. StatsAPI
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
