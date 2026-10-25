# Uncertainty

`fitted_bands` describes uncertainty in individual latent PACE curves.
`mean_bands` describes uncertainty in the population mean estimated from
independent curves. These have different targets and assumptions.

## Individual PACE curves

```julia
bands = fitted_bands(pace; alpha=0.05, coverage=:interval, curves=[1, 3])
simultaneous = fitted_bands(pace; coverage=:band)
# After loading Plots:
bandplot(bands)
```

For curve i, the stored conditional score covariance is Ωᵢ. The variance
of the prediction error for the selected-K latent curve is

```math
v_i(t) = \phi_K(t)^\mathsf{T}\Omega_i\phi_K(t).
```

Pointwise intervals are the reconstructed curve plus or minus
`quantile(Normal(), 1-alpha/2) * sqrt(vᵢ(t))`. Simultaneous bands replace the
normal critical value with `sqrt(quantile(Chisq(K), 1-alpha))`. The package
computes these critical values without a new runtime dependency.
Under the conditional Gaussian model, the score ellipsoid provides at least
nominal coverage for the truncated curve. Its envelope can be conservative.
It is simultaneous within one curve and one node; it does not adjust for
selecting multiple curves or nodes.

The mean, eigenfunctions, covariance, noise variance and component selection
are treated as fixed estimates. Omitted-component bias and uncertainty in
these estimates are not included. Measurement noise is not added to the
band: the target is a latent trajectory, not a future noisy observation.
This is the model-based fitted-curve uncertainty described by refund, rather
than a corrected band that accounts for estimating the FPCA model.
Dense integration scores do not carry conditional covariance. Fit with
`method=:pace` to request this calculation.

`FittedBands` holds copied `grid`, `fitted`, `lower`, `upper`, and `variance`
arrays. Each row follows `curves`. It records `alpha`, `coverage`, selected
`ncomponents`, score `method`, and `critical_value`. A custom grid uses the
same explicit piecewise-linear mean and eigenfunction evaluation as
`reconstruct`. `fitted_bands(multivariate_fit)` returns `(; nodes)`, with
separate node results. Joint Happ–Greven `MFPCA` uncertainty is not implemented.
The stored settings record the selected-score variance source, latent target,
interpolation and PSD cleanup rule. Negative conditional eigenvalues are
clipped only within `100eps(T) * max(1, maximum(abs, eigenvalues))`, where T
is the score-covariance element type; larger negative values throw.

## The mean

```julia
using Random
m = StatsAPI.fit(SmoothedMean, fd; grid=workgrid, bandwidth=h)
uncertainty = mean_bands(fd, m; B=1000, rng=MersenneTwister(42))
# Recreate the bootstrap from its stored initial RNG state:
repeated = mean_bands(fd, m; B=uncertainty.B, rng=uncertainty.rng_state)
# After loading Plots:
bandplot(uncertainty; coverage=:band)
```

The bootstrap samples whole curves with replacement, retaining all their
observed times and within-curve dependence. It refits with the stored scalar
bandwidth, kernel, polynomial degree, weighting and work grid. **Bandwidth
selection is not repeated.** Supply the original training data. A refit must
reproduce the stored mean; this compatibility check cannot certify data
provenance.

`coverage=:interval` gives pointwise percentile intervals from bootstrap
means, with R type-7 empirical quantiles. `coverage=:band` uses

```math
q^*_{1-\alpha} = \operatorname{quantile}_{1-\alpha}
\left\{\max_{t\in\mathcal G}
\frac{|\widehat\mu_b^*(t)-\widehat\mu(t)|}{s^*(t)}\right\}_{b=1}^B,
\qquad
\widehat\mu(t)\ \pm\ q^*_{1-\alpha}s^*(t),
```

where `s*(t)` is the sample standard deviation of the B bootstrap means.
`coverage=:both` returns both types. The original fitted mean centers the
sup-t deviations; no nested bootstrap is used. A zero standard error gives
a zero standardized deviation only if every replicate equals the original
mean there. Otherwise the simultaneous calculation throws.

The grid is held fixed even when a resample has a narrower observed range.
Kernel estimates on that grid must remain defined. An undefined replicate
throws with its number; there are no retries, dropped draws or removed grid
points. A large enough bandwidth may be needed for sparse designs.

The RNG is copied, its initial state is stored, and the caller's state is
preserved. `MeanBands` stores the B refitted means, bootstrap standard errors,
original mean and copied smoother, interval/band endpoints, RNG state and
settings. RNG reproduction requires the same Julia/RNG implementation.
The RNG must support state copying; `RandomDevice` is rejected.

Both intervals and bands require independent subjects and negligible
smoothing bias relative to sampling error. Simultaneity applies to the
supplied finite grid. Degras (2011) supplies functional mean-band theory for
fine-grid data; the whole-curve bootstrap here is a separate calibration
and is not a theorem for arbitrary sparse designs. Coverage should be
checked for the design and smoothing regime being used.

### Analytic pointwise option

```julia
analytic = mean_bands(fd, m; method=:sandwich)
```

KernelSmoothing supports curve-cluster variance. This option uses
`ClusterVariance(1)`, its default CR1 adjustment, with subjects as clusters.
It reuses the original smoother's settings and returns normal intervals
around the stored mean using **conventional** cluster standard errors.
No robust bias correction is applied, so smoothing bias must still be small.
A window with fewer than two contributing clusters has undefined variance
and throws. No window or grid point is dropped.

Only `coverage=:interval` is available for this method and is its default.
Explicit `B` and `rng` throw because no resampling occurs. The result has
`B=0`, `rng_state=nothing`, an empty bootstrap matrix, and records CR1,
conventional inference, curve clusters and `minobs=0`. Analytic simultaneous
mean bands are not implemented.

## Defaults and references

| Choice | Default and source |
|:--|:--|
| Fitted alpha | 0.05, milestone 5 specification; fdapace defaults to `NULL` |
| Fitted coverage | `:interval`, milestone 5 specification; fdapace defaults to `"band"` when intervals are requested |
| Fitted grid, K, curves | Stored work grid, selected K, all training curves, as in fdapace 0.6.0 |
| Mean method | `:bootstrap`, milestone 5 specification |
| Bootstrap alpha, B, coverage, RNG | 0.05, 1000, `:both`, copied `Random.default_rng()`, milestone 5 Gap Map |
| Mean B/rng keyword sentinels | `nothing` resolves to those bootstrap defaults; permits rejecting explicit bootstrap settings for sandwich intervals |
| Mean smoothing | Original stored settings, fixed bandwidth, milestone 5 specification |
| Percentile quantiles | R type 7, fdapace 0.6.0 `GetMeanCI` |
| Sandwich variance | KernelSmoothing CR1 with curves as clusters; conventional inference preserves the stored mean target |
| Sandwich coverage | `:interval`, the available analytic method |
| Plot coverage | All types stored in the supplied result; no new inference runs |
| Ribbon opacity | 0.2, package display choice; Plots attributes can override it |

The [fdapace source audit](fdapace.md#fdapace-confidence-bands) records its
formulas, rank behavior and the reference fixture's explicit corrections.
The golden CSVs use R 4.5.1 and fdapace 0.6.0. Recreate them with
`Rscript test/reference/fdapace_m5_bands.R`; fitted curves and both types of
endpoints are compared at relative error below `1e-8`. The critical-value
fixture also checks central and tail χ² quantiles against R's `qchisq`.

References:

- Yao, Müller and Wang (2005), [Functional Data Analysis for Sparse Longitudinal Data](https://utstat.utoronto.ca/fyao/2005-jasa.pdf),
  equations (7)–(8), Theorem 4 (pointwise), Theorem 5 (simultaneous).
- Degras (2011), [Simultaneous confidence bands for nonparametric regression with functional data](https://www3.stat.sinica.edu.tw/sstest/oldpdf/A21n412.pdf),
  *Statistica Sinica* 21, 1735–1765.
- refund 0.1-40, [`fpca.sc`](https://github.com/cran/refund/blob/0.1-40/R/fpca.sc.R),
  model-based curve covariance; model-estimation correction is discussed in
  Goldsmith, Greven and Crainiceanu (2013), cited there.

## API

```@docs
FittedBands
MeanBands
fitted_bands
mean_bands
band_data
bandplot
```
