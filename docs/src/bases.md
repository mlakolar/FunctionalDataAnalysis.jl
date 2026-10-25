# Bases

All bases expose their dimension and domain, a matrix of values or derivatives,
and matrices of integrated products. Evaluation outside the domain is an error.

```jldoctest
julia> using FunctionalDataAnalysis

julia> b = MonomialBasis((0, 1), 2);

julia> basismatrix(b, [0.0, 0.5, 1.0])
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
Galerkin matrices. Its order is degree plus one. Integer order is a runtime
argument; the wrapper retains a concrete return type for constructors.
The Fourier basis starts with a constant and then sine/cosine pairs. It is
orthonormal on its domain. `penalty(b, :harmonic_acceleration)` for a Fourier
basis integrates products of `D³φ + (2π/domain_width)² Dφ`, the default
operator for [smoothed FPCA](fpca.md#Smoothed-FPCA). Monomials use the original time coordinate.

```@docs
BSplineBasis
FourierBasis
MonomialBasis
nbasis
basismatrix
gram
penalty
```
