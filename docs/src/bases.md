# Bases

All bases expose their dimension and domain, a matrix of values or derivatives,
and matrices of integrated products. Evaluation outside the domain is an error.

```jldoctest
julia> using FunctionalDataAnalysis

julia> b = MonomialBasis((0, 1), 2);

julia> basis_matrix(b, [0.0, 0.5, 1.0])
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 1.0  0.5  0.25
 1.0  1.0  1.0

julia> penalty(b, 2)
3×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  4.0
```

`BSplineBasis` uses BSplineKit's clamped basis, collocation matrices, and
Galerkin matrices. Its order is degree plus one. The wrapper stores the concrete
backend type, including the spline order, as its second type parameter.
`BSplineBasis{T}` remains a valid partial type annotation and its existing
three-argument backend constructor remains available. Code spelling out every
type parameter must account for the backend parameter. Construction with an
order known to the compiler can be inferred; an arbitrary runtime integer order
does not determine one concrete return type. Evaluation and penalties specialize
on the constructed basis, including when its order was chosen at runtime.
The Fourier basis starts with a constant and then sine/cosine pairs. It is
orthonormal on its domain. `penalty(b, :harmonic_acceleration)` for a Fourier
basis integrates products of `D³φ + (2π/domain_width)² Dφ`, the default
operator for [smoothed FPCA](fpca.md#Smoothed-FPCA). Monomials use the original time coordinate.
[`ThinPlateRegressionBasis`](@ref), mgcv's one-dimensional thin plate
regression spline basis (refund's default for penalized regression), is
documented on the [Regression](regression.md#Penalized-scalar-on-function-regression)
page.

```@docs
BSplineBasis
FourierBasis
MonomialBasis
nbasis
basis_matrix
gram
penalty
```
