# Smoothing

Per-curve smoothing accepts grid or irregular observations. Masked values are
unobserved. Local polynomials require a scalar bandwidth and an explicit work
grid. They leave points outside each curve's observed range missing. An
undefined estimate within that range raises an error suggesting a larger
bandwidth. Kernels come from KernelSmoothing.jl.

```@example smoothing
using FunctionalDataAnalysis
import KernelSmoothing
import StatsAPI

t = collect(range(0.0, 1.0; length=21))
fd = GridFunctionalData(permutedims(hcat(1 .+ t, 2 .- t)), t; ids=[:a, :b])
local_fit = smooth_curves(fd, LocalPolynomial(bandwidth=0.3); grid=t)
# The StatsAPI entry point returns the same kind of data object.
local_fit = StatsAPI.fit(LocalPolynomial, fd; bandwidth=0.3, grid=t)

basis = BSplineBasis(4, (0.0, 1.0), 6)
spline_fit = smooth_curves(fd, PenalizedSpline(basis; lambda=0.1))
selected = smooth_curves(fd,
    PenalizedSpline(basis; lambda=GCV([0.001, 0.01, 0.1, 1.0])))
selected.metadata.selection
```

Both methods return functional data, with copied ids. Their `metadata` named
tuple records all settings and is included in `show`. Constructors for grid
and basis data also accept a copied `metadata` named tuple. Metadata describes
the fit at creation; changing the public data arrays does not refit it.
Local fits record bandwidth, degree, kernel, work grid, and `minobs=0`.
Spline fits record basis, selected or fixed lambda, and derivative order.
Both use equal observation weights, no binning, and no folds.

GCV chooses one common lambda by summing the per-curve criteria
`n_i * RSS_i / (n_i - tr(H_i))^2`. Scores use actual observed times, including
for curves with different observation counts. Every candidate and its score
is retained in `metadata.selection`; fixed lambda sets `selection=nothing`.
Singular candidate fits or residual degrees of freedom no larger than
`sqrt(eps(T))*n_i` get infinite scores. No curve is dropped. All-infinite
scores raise an error. Exact ties choose the first supplied candidate.
This criterion concerns prediction within curves and does not account for
within-curve error correlation.

Pointwise confidence bands for the pooled mean are outside this API. They
require cluster-robust variances, which KernelSmoothing.jl does not yet provide.
Pooled mean, covariance, noise variance, and curve-level bandwidth selection
are scheduled for the remaining milestone 2 tasks.

KernelSmoothing.jl is currently unregistered. On Julia 1.10, install it before
instantiating this project (newer Julia versions use the root `[sources]`):

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/KernelSmoothing.jl")
```

KernelSmoothing must be registered before FunctionalDataAnalysis can be registered.

```@docs
LocalPolynomial
PenalizedSpline
GCV
smooth_curves
```
