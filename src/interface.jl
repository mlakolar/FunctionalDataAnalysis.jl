"""
    AbstractFunctionalData{T<:AbstractFloat}

A collection of `n` curves on a common domain, with floating-point type `T`.
Concrete univariate types are `GridFunctionalData`, `IrregularFunctionalData`
and `BasisFunctionalData`. `MultivariateFunctionalData` collects components
for the same subjects. `eltype(fd)` is the numerical scalar type.
"""
abstract type AbstractFunctionalData{T<:AbstractFloat} end

"""
    AbstractBasis

A basis for representing functions on a common domain. The interface
is `nbasis(b)`, `domain(b)`, `basis_matrix(b, t; derivative)`, `gram(b)` and
`penalty(b, order)`.
"""
abstract type AbstractBasis end

"""
    nobs(fd)

Number of curves (subjects) in a functional data collection. This is the
`StatsAPI.nobs` generic.
"""
nobs

"""
    domain(x)

Return the common domain of a functional data collection or basis as the
interval `(lower, upper)`. Multivariate data return one interval per component.
"""
function domain end

"""
    evaluate(fd, t)

Evaluate each curve in `fd` at the points in `t`. Return an
`nobs(fd) × length(t)` matrix, with curves in rows and evaluation points in
columns.

Basis data evaluate exactly and accept `derivative=0`. Grid and irregular
data require an explicit `PiecewiseLinear()` or `NoInterpolation()` evaluator.
Piecewise-linear evaluation returns `NaN` outside a curve's observed range; exact
lookup throws at any unobserved time. Query points must be finite.

This generic is owned by FunctionalDataAnalysis. Use
`FunctionalDataAnalysis.evaluate` when also loading another package that
exports `evaluate`, such as MultivariateStats.

Multivariate evaluation on a common point vector returns an
`nobs × ncomponents × length(t)` array and rejects points outside any
component's domain. A vector of point vectors returns one matrix per
component. Pass an evaluator whenever any component is raw data; it is
ignored for basis components.
"""
function evaluate end

"""
    sampling_design(fd)

Return `:regular` for complete grid data or basis data. For masked grids and
irregular data, return `:dense_irregular` when the median observation count
per curve is at least `threshold` (default 20), otherwise `:sparse`. The
threshold must be finite and positive; an even sample uses the average of
the two middle counts.

Multivariate data return a `Vector{Symbol}`, passing `threshold` to each
component. For example, `all(==(:regular), sampling_design(mfd))` checks that
all components are regular; there is no aggregate multivariate design.
"""
function sampling_design end
