# Data types

Univariate data store complete or masked common grids, curve-specific
observations, or exact basis coefficients. Constructors copy observation and
coefficient arrays. Basis objects are shared and treated as immutable.
`eltype(fd)` is the numerical scalar type, while iteration yields curves.

```jldoctest
julia> using FunctionalDataAnalysis

julia> fd = GridFunctionalData([1 2 3], [0.0, 0.5, 1.0]);

julia> fd[1]
([0.0, 0.5, 1.0], [1.0, 2.0, 3.0])

julia> evaluate(fd, [0.25], PiecewiseLinear())
1×1 Matrix{Float64}:
 1.5

julia> to_grid(to_irregular(fd), fd.grid, NoInterpolation()).values == fd.values
true
```

Raw data require an explicit evaluator. `PiecewiseLinear()` returns `NaN` outside a
curve's observed range; `to_grid` records those results as missing instead.
`NoInterpolation()` requires exact `==` matches, without a tolerance.
During grid conversion it preserves absent observations in the mask and
rejects any source time not present in the target grid. All-missing grid
rows are representable, but cannot become irregular curves or be fitted.

`to_basis` is per-curve least squares on the actual observations. It does
not estimate a pooled mean, covariance, or bandwidth. Unpenalized fits use
pivoted QR; penalized fits use Cholesky and reject singular systems.

Scalar indexing and iteration of basis data retain one-curve basis objects.
They never choose a sample grid. Use `evaluate(fd[i], t)` or `to_grid(fd, t)`
to sample explicitly. Empty subject subsets throw. For basis data,
`nobs_per_curve` reports the number of coefficients, not a chosen grid size.

Multivariate data store copied univariate components for the same subjects.
Use `component(mfd, j)` to select a node; indexing always selects subjects.
`domain`, `nobs_per_curve`, and `sampling_design` return results per component.
Common-point evaluation returns an `n × p × m` array and rejects points
outside any component domain. A vector of point vectors returns a vector
of matrices. Raw components require an evaluator, which basis components
ignore. Complete common-grid components also support `to_array` in FGM order.

Components with a shared floating-point type have an inferable vector
constructor. For a type-erased component vector, explicitly supply
`MultivariateFunctionalData{T}(components)` when inference is required.
The untyped constructor promotes numerical types across mixed components.

```@docs
GridFunctionalData
IrregularFunctionalData
BasisFunctionalData
MultivariateFunctionalData
PiecewiseLinear
NoInterpolation
nobs_per_curve
ncomponents
component
to_array
to_irregular
to_grid
to_basis
Base.getindex(::GridFunctionalData, ::Integer)
Base.iterate(::AbstractFunctionalData)
```
