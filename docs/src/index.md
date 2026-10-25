# FunctionalDataAnalysis.jl

FunctionalDataAnalysis.jl provides functional data analysis in Julia for
dense, irregular and sparse observations. Smoothing uses KernelSmoothing.jl
and BSplineKit.jl. Univariate functional principal component analysis supports
dense data and sparse observations through PACE. FunctionalGraphicalModels.jl will
later depend on this package.

The package is **pre-release**. Milestone 0 provides documented interface
contracts and trapezoidal integration weights. Milestone 1 adds bases,
functional data containers, explicit conversions, table input, and simulation.
Milestone 2 adds per-curve and pooled smoothing, covariance estimation,
noise variance, and whole-curve bandwidth selection. Milestone 3 adds dense
and sparse PACE FPCA, with documented automatic defaults and fdapace
reference validation. See [FPCA](fpca.md) for a no-keyword example.
Milestone 4 adds marginal, pooled, and Happ--Greven multivariate FPCA and
cross-covariance surfaces. See [Multivariate FPCA (MFPCA)](mfpca.md) for the
joint construction, weights, and R reference audit.

## Installation

Julia 1.10 or later is required. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/KernelSmoothing.jl") # required on Julia 1.10
Pkg.add(url = "https://github.com/mlakolar/FunctionalDataAnalysis.jl")
```

## Development tests

From a master checkout's package root, run:

```sh
julia scripts/test_master.jl
```

This prepares the dedicated master-based test environment, including test extras
and sibling workspace dependencies, and runs the complete suite. Package version
numbers and dependency declarations remain unchanged.

## Quadrature

Use [`quadrature_weights`](@ref) to integrate sampled curves on a finite,
strictly increasing grid:

```jldoctest
julia> using FunctionalDataAnalysis

julia> quadrature_weights([0.0, 0.25, 1.0])
3-element Vector{Float64}:
 0.125
 0.5
 0.375
```
