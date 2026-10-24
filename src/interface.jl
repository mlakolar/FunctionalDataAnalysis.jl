"""
    AbstractFunctionalData{T<:AbstractFloat}

A collection of `n` curves on a common domain, with floating-point type `T`.
Concrete types `GridFunctionalData`, `IrregularFunctionalData` and
`BasisFunctionalData` arrive in milestone 1.
"""
abstract type AbstractFunctionalData{T<:AbstractFloat} end

"""
    AbstractBasis

A basis for representing functions on a common domain. The planned interface
is `nbasis(b)`, `domain(b)`, `basismatrix(b, t; derivative)`, `gram(b)` and
`penalty(b, order)`.
"""
abstract type AbstractBasis end

"""
    nobs(fd)

Number of curves in a functional data collection. This is the `StatsAPI.nobs`
generic; concrete functional data types will provide methods in milestone 1.
"""
nobs

"""
    domain(x)

Return the common domain of a functional data collection or basis as the
interval `(lower, upper)`.
"""
function domain end

"""
    evaluate(fd, t)

Evaluate each curve in `fd` at the points in `t`. Return an
`nobs(fd) × length(t)` matrix, with curves in rows and evaluation points in
columns.
"""
function evaluate end

"""
    sampling_design(fd)

Return the observation design of `fd`: `:regular` for a common grid,
`:dense_irregular` for dense curve-specific grids, or `:sparse` for sparse
curve-specific observations.
"""
function sampling_design end
