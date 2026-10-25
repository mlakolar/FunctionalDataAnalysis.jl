# FunctionalDataAnalysis.jl

[![CI](https://github.com/mlakolar/FunctionalDataAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/FunctionalDataAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/FunctionalDataAnalysis.jl/dev/)
[![codecov](https://codecov.io/gh/mlakolar/FunctionalDataAnalysis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/FunctionalDataAnalysis.jl)
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
Milestone 4 adds marginal, pooled, and Happ--Greven multivariate FPCA and
cross-covariance surfaces for the FunctionalGraphicalModels.jl migration.

## Installation

Requires Julia 1.10 or later. Install the pre-release package from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/FunctionalDataAnalysis.jl")
```

## Development and testing

Use a `master` checkout for this package and its workspace dependencies. From
the package root, run the complete test suite with:

```sh
julia scripts/test_master.jl
```

The runner resolves sibling master checkouts and test extras in a dedicated
`.master-test-env` environment, leaving the package dependency declarations
unchanged. Missing workspace dependencies are cloned from their master branches.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the naming and development policy.

## Licence

The package code is MIT-licensed (see `LICENSE`). The datasets loaded by
`load_dataset` keep their upstream terms: growth, Canadian weather and gait
(R fda 6.3.0), CD4 and DTI (R refund 0.1-40) are GPL (>= 2), and Tecator
(fda.usc 2.2.0, GPL-2) is redistributed under its original StatLib permission
note. The licence texts and notes ship with the data in `data/licenses` and in
the artifact.

## Roadmap

- Milestone 0: package skeleton, CI, docs.
- Milestone 1 (done): data types (grid, irregular, basis, multivariate), conversions,
  Tables.jl input, simulation generators.
- Milestone 2: smoothing (curve smoothing, pooled mean, covariance surface,
  noise variance, curve-level bandwidth selection) on KernelSmoothing.jl.
- Milestone 3 (done): FPCA, dense and sparse (PACE), with automatic defaults
  and fdapace reference validation.
- Milestone 4 (done in FDA): marginal, pooled, and Happ--Greven (`MFPCA`)
  multivariate FPCA; sample and smoothed cross-covariance surfaces with
  multivariate blocks. FGM estimator migration is maintained in its own repository.
- Milestone 5 (in progress): plotting coordinates and Plots.jl recipes;
  six classic datasets through lazy artifacts, with offline samples;
  derivatives, confidence bands, and the noise-variance benchmark.
- Milestone 6: FACE and penalized FPCA.
- Milestones 7–9: regression; depth and clustering; registration.
