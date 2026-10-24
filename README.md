# FunctionalDataAnalysis.jl

[![CI](https://github.com/mlakolar/FunctionalDataAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mlakolar/FunctionalDataAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/FunctionalDataAnalysis.jl/dev/)
[![codecov](https://codecov.io/gh/mlakolar/FunctionalDataAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mlakolar/FunctionalDataAnalysis.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Functional data analysis in Julia, with planned support for dense, irregular
and sparse observations, smoothing built on KernelSmoothing.jl and BSplineKit.jl,
and dense and sparse (PACE) functional principal component analysis. The
package is pre-release. Milestone 0 provides the package skeleton, documented
interface contracts and trapezoidal quadrature weights. Milestone 1 adds
bases, functional data containers, explicit conversions, table input and
simulation generators. Per-curve smoothing, pooled mean and covariance
estimation, and dense and sparse PACE FPCA are implemented. `fit(FPCA, data)`
chooses documented defaults and records every resolved choice on the fit.
FunctionalGraphicalModels.jl
will later depend on this package.

## Installation

Requires Julia 1.10 or later. Install the pre-release package from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/FunctionalDataAnalysis.jl")
```

## Roadmap

- Milestone 0: package skeleton, CI, docs.
- Milestone 1 (done): data types (grid, irregular, basis, multivariate), conversions,
  Tables.jl input, simulation generators.
- Milestone 2: smoothing (curve smoothing, pooled mean, covariance surface,
  noise variance, curve-level bandwidth selection) on KernelSmoothing.jl.
- Milestone 3 (done): FPCA, dense and sparse (PACE), with automatic defaults
  and fdapace reference validation.
- Milestone 4: multivariate FPCA; FunctionalGraphicalModels.jl migrates.
- Milestone 5: registration, functional regression, depth and outliers,
  plotting recipes.
