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

Pooled-mean uncertainty is available through [`mean_bands`](@ref):
whole-curve bootstrap intervals and simultaneous bands, or analytic pointwise
intervals using KernelSmoothing's curve-cluster sandwich variance. See
[mean uncertainty](uncertainty.md#The-mean) for assumptions and fixed-bandwidth
calibration.
Covariance and noise variance are estimated as described below.

KernelSmoothing.jl is currently unregistered. From this package's development
checkout, prepare the sibling master dependencies on Julia 1.10 and newer with:

```sh
julia scripts/instantiate.jl
```

The bootstrap develops the sibling KernelSmoothing master checkout and clones it
from master if missing. Run `julia scripts/test_master.jl` for the complete suite
in its dedicated environment with test extras.

KernelSmoothing must be registered before FunctionalDataAnalysis can be registered.

```@docs
LocalPolynomial
PenalizedSpline
GCV
smooth_curves
```

Polynomial `degree` and spline `order` count different things: a local
quadratic uses `degree=2`, while a cubic `BSplineBasis` uses `order=4`.
The `order=2` in `PenalizedSpline` instead selects the derivative in its
integrated roughness penalty. These are separate controls, not aliases.

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
root to recreate the version-labelled CSVs. The covariance fixture uses ordered within-curve off-diagonal residual products, centered
at direct mean estimates at observation times, and `Lwls2D(method="plain")`.
Its subject weights are `1/(N_i*(N_i-1))`. The covariance tests compare this quantity to relative error below `1e-8`.
The mean is stored at all input times for this comparison, so centering agrees
with the direct fdapace mean rather than a coarser interpolated mean.

```@docs
SmoothedMean
CurveCV
```

## Covariance and noise variance

`StatsAPI.fit(SmoothedCovariance, fd, mean_fit; grid, bandwidth, diagonal_bandwidth,
noise_method=:rotated, noise_interval=:full)` smooths all
ordered within-curve residual products with distinct observation indices.
This excludes measurement noise on the raw diagonal. The product kernel uses
the same scalar bandwidth in both coordinates. Local linear fits solve a
three-coefficient weighted least-squares system; singular windows throw and
name the query point. The final surface is symmetrized exactly. It is not
projected onto the positive-semidefinite cone.

`:obs` assigns weight one to each pair. `:subj` assigns
`1/(N_i*(N_i-1))` so each curve has total pair weight one. Each curve must have
at least two observations; empty and singleton curves raise errors.
The supplied mean is a `SmoothedMean` fit or one complete `GridFunctionalData`
curve, including a mean stored by multivariate FPCA. It is linearly
interpolated at the original observation times.
Its stored grid must cover all those times. Stored data weights are unused.

A `CurveCV` selector leaves entire curves out of the covariance smoother and
predicts their raw products, minimizing weighted squared prediction error.
The supplied mean remains fixed across folds; CV is conditional on that mean.
This does not refit the mean within each training fold. Split rules, tie rules,
and errors for undefined candidates follow the pooled-mean conventions above.

`binning=n` snaps original times onto the nearest of `n` equally spaced
points spanning the observed domain. Midpoint ties go upward. Residuals are
computed at the original times before snapping. Pair multiplicities and
original observation-index diagonal exclusions are preserved, including
products whose two distinct observations snap to the same bin. No observations
are discarded. Within each curve and bin cell, pair weights are summed and
products are replaced by their weighted mean. This preserves the fixed-bandwidth
fit at snapped times. Within-cell squared errors are retained, so CV still
scores the original raw products. Default `binning=nothing` disables snapping.
Exact duplicate coordinates are combined even without binning, retaining total
weights, weighted products and scatter. Fixed-bandwidth fits combine pairs as
they are formed; original diagonal observations remain separate. CV splits
subjects first, then combines training and held-out coordinates separately.

Forming products takes `O(P)` work, where `P = sum(N_i*(N_i-1))`.
For fixed bandwidths, pair storage uses the number of distinct coordinate
pairs. For each CV fold, smoothing costs the product of the numbers of distinct
training and held-out coordinate pairs, per candidate. In the absence of
duplicates this retains the `O(P^2)` bound. With binning,
`Q = sum(min(N_i*(N_i-1), n^2))` bounds the occupied curve/bin cells before
fold-specific compression. Neither form of aggregation removes the cost of
initially forming the products.

The noise estimator smooths raw residual squares by a univariate local linear
fit with `minobs=0`. By default it uses the covariance bandwidth (after any
CV selection), as fdapace does; theory only requires the diagonal bandwidth to
be of no larger order than the covariance bandwidth, so this default is
consistent (Li and Hsing, 2010, Corollary 3.5). Pass `diagonal_bandwidth` to
choose it separately: a positive number, a `CurveCV`, or a KernelSmoothing
regression rule (see below). `noise_details.diagonal_source` records
`:covariance` or `:user`. Diagonal CV predicts the raw squared residuals of held-out
curves at their original observation times, using the pooled-mean weighting
and fold rules. The supplied mean remains fixed. When binning is enabled,
training times are snapped as in the final diagonal smoother; held-out query
times remain original. Its weights are one
for `:obs` and `1/N_i` for `:subj`. With the former default selected explicitly
as `noise_method=:surface_diagonal, noise_interval=:middle_half`, this fit
`V(t)` and the covariance `G(s,t)` give the raw noise estimate
`(2/(b-a)) * integral(V(t)-G(t,t), t=a+(b-a)/4..b-(b-a)/4)`.
Here `[a,b]` is the work-grid range. Trapezoidal quadrature includes the exact
middle-half boundaries, interpolating `V-G` linearly when needed. Negative
estimates are truncated to zero. `noise_details` records the untruncated value,
truncation flag, diagonal bandwidth, fold count, selection candidates, errors
and assignments, fitted squares, and integration settings.
`diagonal_bandwidth` also appears directly on the fit.
The stored `weights` integrate over the full covariance work grid.

With `noise_interval=:full`, the surface-diagonal method
integrates over the whole work-grid domain and divides by its length,
following the unrotated integrated diagonal estimator of Li and Hsing (2010).
The default is now `noise_method=:rotated, noise_interval=:full`, chosen from
the issue #17 Discovery benchmark (48 designs × 500 replicates) on 25 October
2026. PACE inherits this choice. The latent diagonal follows
Yao, Müller and Wang (2005), as implemented by fdapace 0.6.0 `PC_CovE` and
`RotateLwls2DV2`: rotate by 45 degrees, fit a quadratic across the diagonal
and a linear term along it, and use the covariance bandwidth in both rotated
axes. This option uses existing work-grid points within the selected portion
of the **observed** domain and divides by that portion's nominal length.
It does not insert interval endpoints; at least two retained points are
required. The covariance surface is unaffected. The raw-square bandwidth can
still be overridden independently. Negative estimates truncate to zero.
Singular rotated fits throw an informative error suggesting a larger bandwidth.
For a small design that cannot support the rotated quadratic, explicitly use
`noise_method=:surface_diagonal`; there is no automatic fallback.
The [source audit](@ref fdapace-rotated-noise) details the construction and
the differences from fdapace's enclosing FPCA routine.

`noise_details.noise_method` and `.noise_interval` record these choices.
The details also store the integration domain, nominal length, boundary rule,
latent diagonal at the integration points, and rotation settings. `show`
includes them.
The standalone harness lives in `benchmark/sigma2/`; its configuration grid,
Discovery array template, recreation commands, and summary script are described
in that directory's README. [Noise-variance estimators (issue #17)](m5_sigma2_report.md)
gives the Discovery results, benchmark-selected defaults and a separate harness
smoke check.

References: Yao, Müller and Wang (2005), *Functional Data Analysis for Sparse
Longitudinal Data*, [doi:10.1198/016214504000001745](https://doi.org/10.1198/016214504000001745);
Li and Hsing (2010), *Uniform convergence rates for nonparametric regression
and principal component analysis in functional/longitudinal data*,
[doi:10.1214/10-AOS813](https://doi.org/10.1214/10-AOS813).

```@example smoothing
using Random
sparse = simulate_kl(MersenneTwister(42), 80; mean=t -> 1 + t,
    eigenfunctions=[t -> 1.0], eigenvalues=[0.5], noise_sd=0.2,
    sampling=SparseSampling((0.0, 1.0), 6:9)).data
workgrid = collect(range(0.0, 1.0; length=11))
pooled = StatsAPI.fit(SmoothedMean, sparse; grid=workgrid, bandwidth=0.8)
covariance = StatsAPI.fit(SmoothedCovariance, sparse, pooled;
    grid=workgrid, bandwidth=0.8, diagonal_bandwidth=0.6, weighting=:subj)
(size(covariance.surface), covariance.noise_variance, evaluate(covariance, 0.3, 0.6))
```

`evaluate(covariance, s, t)` uses bilinear interpolation within the stored grid.
Nonfinite queries and extrapolation raise errors. All tuning settings, the
sampling design, CV results, and noise details appear in `show`.

```@docs
SmoothedCovariance
```

### Noise options for PACE

`noise_method` and `noise_interval` belong to `SmoothedCovariance` and are
not forwarded by `fit(FPCA, ...; method=:pace)`. To choose a nondefault noise
estimator, fit the covariance explicitly and supply the fitted object:

```julia
alternative = StatsAPI.fit(SmoothedCovariance, sparse, pooled;
    grid=workgrid, bandwidth=0.8, diagonal_bandwidth=0.6, weighting=:subj,
    noise_method=:surface_diagonal, noise_interval=:middle_half)
pace = StatsAPI.fit(FPCA, sparse; method=:pace, grid=workgrid,
    mean=pooled, covariance=alternative, ncomponents=1)
```

This reuses the fitted surface and `alternative.noise_variance`; it does not
re-estimate the noise with PACE's defaults. The mean/covariance objects and
their work grids must be compatible with the observations. An explicit
`noise_variance` overrides the object's estimate, as in the existing FPCA API.
Supplying only a `(grid, surface)` named tuple instead requires that explicit
noise value. The unchanged automatic default remains `noise_method=:rotated,
noise_interval=:full`; use the explicit recipe only to select an alternative.

## [Cross-covariance](@id smoothing-cross-covariance)

For complete common-grid curves,
`StatsAPI.fit(CrossCovariance, fd1, fd2)` returns the sample cross-covariance
with denominator `n-1`. Each node is centered by its own sample mean; each
node may have a different grid. Missing observations raise an error.

For sparse, irregular, masked, or explicitly smoothed grid data, fit
`SmoothedCrossCovariance` with two supplied means. For each subject, it forms
all Cartesian residual products between the nodes and reuses the bivariate
local linear smoother on `grid1 × grid2`. The surface is generally not
symmetric. A positive scalar bandwidth uses the same width in both axes;
`bandwidth=(h1,h2)` chooses them separately. The default kernel is Gaussian,
following fdapace 0.6.0 `GetCrCovYX`. Bandwidths and work grids are explicit;
an explicit-candidate `CurveCV` leaves out whole subjects, with the two means
held fixed. This selection deliberately differs from fdapace's Gaussian
GCV default. The [source audit](fdapace.md) details every difference.

Same-time pairs are **kept** by default. This assumes independent measurement
errors between the two nodes. When errors at shared times are correlated,
use `exclude_same_time=true`; the test uses exact original times before any
binning. Each subject must retain at least one pair. `weighting=:obs` gives
each retained pair weight one; `:subj` gives `1/(N1_i*N2_i)`, without changing
the denominator after same-time exclusions. Data-object weights are unused.
Empty curves, mismatched counts, mismatched available IDs, inadequate work
or mean grids, and singular windows raise informative errors.

The optional `binning` integer snaps each axis over its own observed range
and reuses covariance pair aggregation within subject. Multiplicities and
within-cell squared errors are retained; CV predicts raw products at the
snapped coordinates. Default `nothing` performs no snapping. Exact duplicate
coordinates are combined with their weights and scatter in either case;
whole-subject CV splits training and held-out pairs before combining them.
`evaluate(c,s,t)` interpolates bilinearly inside the stored grids and throws
outside them. Both grids and their quadrature weights, the two copied means,
all tuning settings, and CV results are stored on the fit and shown.

```@example smoothing
# Two nodes on the same subjects. The mean objects can also be the
# one-curve grid means stored by a MultivariateFPCA fit.
other = GridFunctionalData(2 .* fd.values .+ 0.1, t; ids=ids(fd))
other_mean = StatsAPI.fit(SmoothedMean, other; grid=t, bandwidth=0.5)
cross = StatsAPI.fit(SmoothedCrossCovariance, fd, other, mean_fit, other_mean;
    grid1=t, bandwidth=(0.4, 0.6))
evaluate(cross, 0.25, 0.75)
sample_cross = StatsAPI.fit(CrossCovariance, fd, other)
```

`cross_covariances(mfd; bandwidth, ...)` returns a matrix of fits with the
existing `SmoothedCovariance` fits on the diagonal and cross fits elsewhere.
It fits each unordered node pair once and transposes the result, enforcing
`G[j,k](s,t) = G[k,j](t,s)` exactly at stored grid points. Reverse blocks have
independent arrays. The helper accepts one shared scalar bandwidth or
`CurveCV`; direct cross fits support separate axis widths.

Supply `fpca=a_multivariate_fpca_fit` to reuse its node means, or provide a
vector `means`. Without supplied means, complete grid nodes use sample means
and irregular/masked nodes use Gaussian local linear means at 5% of their
observed range, as multivariate marginal FPCA does. `mean_bandwidth` requests
explicit mean smoothing and can be shared or per node. A shared real `grid`
vector or vector of node grids overrides the mean grids / FPCA default grids.
The diagonal blocks retain their own off-diagonal observation-index rule and
noise estimator. This helper always returns smoothed blocks; it does not
silently switch to sample covariance for dense inputs.

```@example smoothing
nodes = MultivariateFunctionalData([fd, other])
blocks = cross_covariances(nodes; bandwidth=0.6)
blocks[1,2].surface == transpose(blocks[2,1].surface)
```

```@docs
CrossCovariance
SmoothedCrossCovariance
cross_covariances
```

## Automatic univariate pilots

Bandwidth keywords remain required. The pooled mean's `bandwidth` and the
noise diagonal's `diagonal_bandwidth` also accept regression rules from
KernelSmoothing, such as `KernelSmoothing.RuleOfThumb()` or
`KernelSmoothing.IMSEOptimal()`. Each rule receives the smoother's pooled
responses, times, observation weights, kernel, and degree. For the diagonal,
responses are raw squared residuals and times are snapped when binning is used.
The mean stores `selection=(rule, bandwidth, details)`; the diagonal stores it
in `noise_details.selection`. Density-only rules and rules returning pointwise
vectors (currently `MSEOptimal()`) raise `ArgumentError`.

These rules assume independent observations. For sparse designs that is
asymptotically adequate, because each curve contributes about one point to a
smoothing window and the leading variance matches the independent case (Zhang
and Wang, 2016, Corollaries 3.1 and 3.2). For dense designs they ignore the
within-curve correlation term. Li and Hsing (2010, Section 4.1) also found
IMSE-optimal bandwidths worked less well for ``\sigma^2`` than slightly smaller
ones. Treat these rules as pilots and use `CurveCV`, which leaves out whole
curves and does not depend on the within-curve correlation, for the final
choice. Theory requires only that the diagonal bandwidth be of no larger order
than the covariance bandwidth (Li and Hsing, 2010, Corollary 3.5), so
undersmoothing the diagonal is allowed.

```julia
pilot = KernelSmoothing.RuleOfThumb()
automatic = StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=pilot)
cv = CurveCV(; pilot, folds=nobs(fd))
selected = StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv)
```

Pilot-based `CurveCV` computes `h_pilot` on the full pooled data at fit time,
then evaluates `multipliers .* h_pilot` with the existing curve-level CV.
Defaults are `multipliers=2.0 .^ range(-1, 2; length=10)`, `folds=10`, and
`rng=nothing`; supply an RNG unless leaving one curve out. Selection retains
`pilot`, `h_pilot`, `pilot_details`, copied `multipliers`, candidate `bandwidths`,
CV `errors`, and fold `assignments`. Explicit `CurveCV(bandwidths; ...)` keeps
its existing behavior. Both forms work for the mean and noise diagonal.

The covariance surface uses a bivariate smoother. KernelSmoothing rules are
univariate, so covariance `bandwidth` rejects rules and pilot-based `CurveCV`.
Use `CurveCV` with explicit candidates for that bandwidth.

## Derivatives

A basis expansion can be differentiated without refitting. `derivative(fd, ν)`
keeps its coefficients, ids, and metadata and replaces its basis with
`DerivativeBasis(fd.basis, ν)`. The new functions are the analytic derivatives
of the original basis functions. Their Gram and roughness matrices use
BSplineKit Galerkin integration for B-splines and closed forms for Fourier and
monomial bases. The Gram matrix may be singular, for example when the original
basis contains a constant. This is different from fda 6.3.0 `deriv.fd`, which
projects sampled derivatives back into the original basis; see the
[reference audit](fdapace.md#Derivative-source-audit).

Raw observations do not define a derivative. `derivative` on grid or irregular
data throws and suggests smoothing or projection to a basis first. Estimate a
velocity directly with a local polynomial instead:

```@example smoothing
position = GridFunctionalData(reshape(1 .+ 2t .+ t.^2, 1, :), t)
velocity = smooth_curves(position,
    LocalPolynomial(; bandwidth=0.3, derivative=1); grid=t)
velocity.metadata
mean_velocity = StatsAPI.fit(SmoothedMean, position;
    grid=t, bandwidth=0.3, derivative=1)
evaluate(mean_velocity, [0.25, 0.75])
```

The default polynomial degree is `ν+1`, so degree minus derivative order is
odd (Fan and Gijbels, 1996, *Local Polynomial Modelling and Its Applications*).
A degree at least `ν` can be supplied explicitly. Both per-curve and pooled
fits pass the derivative order to KernelSmoothing.jl and retain it in their
settings and display. A polynomial of degree at most the fitted degree has its
analytic derivative reproduced, subject to floating-point error and sufficient
local support. Curve fits keep out-of-range values missing. A singular local
fit raises an error naming the curve/point.

Positive-order pooled mean fits require a fixed bandwidth. The existing
value-prediction CV criterion and value-oriented bandwidth selectors do not
select derivative bandwidths. This restriction prevents an unstated tuning
rule; zero-order mean fits retain their existing selectors. `evaluate`,
`mean_data`, and `plot(mean_velocity)` use the stored derivative values.
Only `SmoothedMean` fits with `derivative=0` can center observations in FPCA,
covariance, or cross-covariance estimation. Supplying a positive-order mean
derivative as a centering mean raises `ArgumentError`.

```@docs
DerivativeBasis
derivative
```
