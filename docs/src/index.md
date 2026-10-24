# FunctionalDataAnalysis.jl

FunctionalDataAnalysis.jl will provide functional data analysis in Julia for
dense, irregular and sparse observations. Planned methods include smoothing
built on KernelSmoothing.jl and BSplineKit.jl, and dense and sparse (PACE)
functional principal component analysis. FunctionalGraphicalModels.jl will
later depend on this package.

The package is **pre-release**. Milestone 0 provides documented interface
contracts and trapezoidal integration weights. Milestone 1 adds bases,
functional data containers, explicit conversions, table input, and simulation.
Smoothing and FPCA
are planned for later milestones.

## Installation

Julia 1.10 or later is required. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/FunctionalDataAnalysis.jl")
```

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
