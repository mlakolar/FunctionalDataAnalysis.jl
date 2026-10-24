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
Covariance and noise variance are scheduled for the next milestone 2 task.

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

## Pooled means and curve-level CV

`StatsAPI.fit(SmoothedMean, fd; grid, bandwidth, weighting=:obs)` pools all
observed values. `:obs` weights each observation equally. `:subj` uses `1/N_i`
per observation so each curve has total weight one. Data-object observation
and curve weights are not used by this API. Empty curves raise an error.
The work grid is explicit; there is no binning or nearest-neighbour bandwidth
floor. The sampling design is retained as `design`.

```@example smoothing
mean_fit = StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=0.3)
evaluate(mean_fit, [0.25, 0.75])
cv_mean = StatsAPI.fit(SmoothedMean, fd; grid=t,
    bandwidth=CurveCV([0.2, 0.3, 0.5]; folds=nobs(fd)), weighting=:subj)
cv_mean.selection
```

`evaluate` returns linear interpolation on the stored grid, including its
endpoints. It throws outside that range or for nonfinite queries. The pooled
smoother itself can estimate beyond observed times when the kernel window has
enough distinct times; undefined local fits raise an error naming the point.

`CurveCV` predicts held-out curves at their actual observation times, leaving
out whole curves. Its error is the total weighted squared error divided by
the total observation weight, using the same `:obs` or `:subj` scheme as the
fit. Every candidate must yield finite predictions in every fold; otherwise
selection throws and suggests a larger bandwidth. Ties select the first
candidate. For K-fold selection, supply `rng`, for example
`CurveCV([0.3, 0.5]; folds=2, rng=MersenneTwister(42))` after `using Random`.
Folds are shuffled and balanced in curve counts. The RNG state is copied, so
reusing a selector repeats its split without advancing the caller's RNG.
Leave-one-curve-out (`folds=nobs(fd)`) is deterministic and requires no RNG.
The result stores candidate bandwidths, errors, per-curve fold assignments,
fold count, and all smoothing settings; `show` displays them.

The fixed-bandwidth mean is checked against fdapace 0.6.0 `Lwls1D`, using
Epanechnikov local linear fits without binning, for both weighting schemes.
The reference script is `test/reference/fdapace_m2.R`. Run it from the repository
root to recreate the version-labelled CSVs. The covariance fixture is reserved
for task 3: it uses ordered within-curve off-diagonal residual products, centered
at direct mean estimates at observation times, and `Lwls2D(method="plain")`.
Its subject weights are `1/(N_i*(N_i-1))`. No covariance comparison is claimed
by the mean tests.

```@docs
SmoothedMean
CurveCV
```
