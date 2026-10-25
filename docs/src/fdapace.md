# Comparison with fdapace

This audit reads the **CRAN fdapace 0.6.0 source**, not defaults recalled from
earlier versions. CRAN's `DESCRIPTION` reports version 0.6.0, dated 2024-07-02
and published 2024-07-03. The source and current version were checked on
2026-10-24. The reference fixtures use R 4.5.1 and fdapace 0.6.0.

Sources: [CRAN package description](https://cran.r-project.org/web/packages/fdapace/DESCRIPTION),
[versioned source archive](https://cran.r-project.org/src/contrib/fdapace_0.6.0.tar.gz),
and the [CRAN source mirror at version 0.6.0](https://github.com/cran/fdapace/tree/0.6.0).
The archive's SHA-256 is
`1d3e80d885ab8a091b5a99f7edff1e043ad878030f98eadb53d1b3062c5d3bcb`.
File paths below are relative to that archive. Function names identify the
executable source; several comments and help-text defaults disagree with it.

## [Cross-covariance: GetCrCovYX (milestone 4 task 3)](@id fdapace-cross-covariance)

This section audits the executable fdapace **0.6.0** sources
[`R/GetCrCovYX.R`](https://github.com/cran/fdapace/blob/0.6.0/R/GetCrCovYX.R),
[`R/GetRawCrCovFuncFunc.R`](https://github.com/cran/fdapace/blob/0.6.0/R/GetRawCrCovFuncFunc.R),
[`R/BinRawCov.R`](https://github.com/cran/fdapace/blob/0.6.0/R/BinRawCov.R),
[`R/BwNN.R`](https://github.com/cran/fdapace/blob/0.6.0/R/BwNN.R), and
[`R/Lwls2D.R`](https://github.com/cran/fdapace/blob/0.6.0/R/Lwls2D.R).
The locally installed R 4.5.1 / fdapace 0.6.0 function bodies were also
inspected. These are defaults of `GetCrCovYX`, distinct from `FPCA` defaults.

| Item | fdapace 0.6.0 executable behavior | FDA contract |
|:--|:--|:--|
| Sparse inputs | `Ly1,Lt1,Ly2,Lt2`: lists aligned by subject; both means are required, with values on each node's sorted unique pooled times. `GetRawCrCovFuncFunc` linearly interpolates means to observations and forms every within-subject Cartesian residual product. | Grid or irregular nodes; explicit checks for equal subject counts, matching available IDs, finite residuals, and observation coverage. Means are smoothed fits or one-curve complete grid data, including task 1 FPCA means. |
| Dense inputs | With both value matrices and no times or means, return `rawCC$rawCCov=cov(Ly1,Ly2,use="pairwise.complete.obs")`; no smoothed surface. Divisor is `n-1`. | `CrossCovariance` uses separate sample means and divisor `n-1`. Missing observations throw to prevent silent pairwise deletion. Nodes may have different grids. |
| Kernel | Formal default `kern="gauss"`; the help comment saying FALSE is incorrect. `smoothRCC2D` calls `Lwls2D(...,crosscov=TRUE)`, so it does not symmetrize. | `SmoothedCrossCovariance` and the block helper default to Gaussian; any supported kernel can be supplied. The diagonal helper blocks receive that same kernel. |
| Bandwidths | `bw1=NULL,bw2=NULL`. Two numeric bandwidths bypass selection; otherwise both are selected. Automatic selection rejects non-Gaussian kernels. | Required scalar, axis tuple `(h1,h2)`, or explicit-candidate `CurveCV`. Separate axis bandwidths match the audited fixed-bandwidth method. |
| Bandwidth selection | Gaussian GCV; default `bwRoutine="l-bfgs-b"`. `getBWidths` constructs ten equally spaced values per axis from that node's `BwNN(...,onlyCov=TRUE,k=5)` to its `k=45` value. The bounds are computed from **within-node** time-pair clouds. The default bounded optimizer starts at 95% of the upper bounds and uses `maxit=51`. `bobyqa` and `grid-search` are explicit alternatives. | Whole-subject `CurveCV` with explicit scalar candidates reuses milestone 2 and respects subject dependence. No automatic GCV or candidate grid is silently substituted. This intentional departure requires an explicit bandwidth or selector. |
| Same-time pairs | `rmDiag=FALSE`. When true, remove pairs with exactly equal coordinates after exact-duplicate aggregation. This concerns equal times, not matching observation indices. | `exclude_same_time=false`, assuming independent cross-node measurement errors. Set true to remove correlated same-time noise; equality is tested on original times before optional snapping. |
| Pair weights | Raw pairs have weight one. If at least 20% of time-pair rows duplicate earlier rows and `useGAM=FALSE`, `BinRawCov` groups **exact** duplicates, storing means, counts, and within-cell RSS. `smoothRCC2D` supplies the counts as weights. There is no subject-equal weighting option here. | `:obs` is identical for fixed-bandwidth smoothing. Optional `:subj` uses `1/(N1_i*N2_i)` per retained pair, without renormalizing after exclusions. No data weights are silently used. |
| Binning | Exact-duplicate grouping leaves the fixed-bandwidth surface unchanged. It does not snap distinct times. | Default `binning=nothing`; duplicate rows with weight one already give the same fit. Explicit integer binning reuses milestone 2: snap each axis to its own observed-range bin grid; aggregate within subject, preserving multiplicities and RSS. This changes the fitting locations and is an explicit departure. |
| Output | A 51 by 51 `smoothedCC`, on separate equally spaced grids spanning each node's observed range. `smoothGrid` is a 51 by 2 matrix containing the two axes, **not** the 2601 product points. Also returns processed `rawCC`, `bw`, and `score`. | Explicit `grid1,grid2` for direct fits, stored with their separate quadrature weights. The block helper defaults to FPCA/mean grids. Evaluation is bilinear within the two grid ranges. |

`GCVgauss2D` interpolates the fitted surface back to raw (or grouped) pair
coordinates. Its numerator is the unweighted sum of squared differences of
the raw/group means; the denominator is
`(1-r1*r2*0.398942^2/(N*bw1*bw2))^2`, where `N` is the number of raw/group
values and `r1,r2` are the output-grid ranges. Thus grouping counts affect
smoothing but are not used as multiplicities in this GCV loss. The code
returns this Gaussian-form score even for fixed non-Gaussian bandwidths,
with a source comment warning that it can be incorrect. FDA compares only
the fixed-bandwidth **surface**, not this diagnostic score.

The source's optional grid-search loop indexes `1:length(bwCandidates)` on
a 100 by 2 matrix, rather than its 100 rows. Its advertised missing-`minqa`
fallback also uses `==` rather than assignment. Neither path is used or
replicated here. `useGAM=FALSE` is the fdapace default; the optional GAM
route is outside this task.

`test/reference/fdapace_m4.R` generates fixed sparse two-node observations
and `m4_fdapace_crosscov.csv` using `GetCrCovYX` itself, Epanechnikov,
`bw1=.55,bw2=.8`, and both `rmDiag` values. Supplied linear means agree
exactly under interpolation. Tests compare the full returned 51 by 51
surface in its original orientation at relative error below `1e-8`.
The fixture exercises exact duplicate grouping. FDA also combines exact
duplicate coordinates with their original weights, without snapping times.
The generating script and CSV headers record R and fdapace versions.

## Defaults and data routes

Here `n` is the number of curves and `m` is the number of work-grid points.
The package implements dense and PACE FPCA, including no-keyword fits.
PACE defaults use 51 equally spaced work-grid points, a Gaussian kernel,
observation weighting, and fixed mean/covariance bandwidths of 5%/10% of the
observed range. Each choice is overridable, stored, and shown. This audit
identifies the matching conventions and the differences retained by design.

| Choice | fdapace 0.6.0: executable source | FunctionalDataAnalysis.jl and reason |
|:--|:--|:--|
| Work grid / `nRegGrid` | `R/SetOptions.R`, `SetOptions`: Dense and DenseWithMV use the number of unique pooled times after `signif(t,6)`; Sparse and `p>>n` use 51. `R/FPCA.R`, `FPCA` then creates an equally spaced grid over the observed range. | Dense uses the supplied data grid and its stored weights, without rounding or resampling. It preserves the data representation. PACE uses 51 equally spaced points over the observed range unless a work grid is supplied, including when PACE is explicitly requested for common-grid data. |
| `max_k` | `SetOptions`: `min(nRegGrid-2,n-2)`, additionally capped at 20 **only for `methodMuCovEst="smooth"`**; values below 1 are reset to 1. An explicit `max_k` above `nRegGrid-2` is reduced with a message. | The dense default is `max(1,min(m-2,n-2))`; PACE uses `max(1,min(20,m-2,n-2))`, matching the respective audited routes. An explicit dense cap may use up to `min(m,n-1)` and an explicit PACE cap up to `m`; requested components above the cap or retained positive rank throw. |
| `methodSelectK` | `SetOptions`: `"FVE"`. `R/SelectK.R`, `SelectK` also accepts `"AIC"`, `"BIC"`, or a positive integer. | `ncomponents=FVE(0.99)` or a positive integer. PACE also accepts `AIC()` and `BIC()`. |
| `FVEthreshold` | `SetOptions`: 0.99 for `FPCA`. The standalone `SelectK` function separately defaults to 0.95. | `FVE(0.99)` follows the `FPCA` entry point. |
| Kernel | `SetOptions` initially chooses `"epan"` for Dense and `"gauss"` for every other data type, including DenseWithMV. A later invalid-kernel recovery branch chooses Epanechnikov for DenseWithMV. | The dense sample mean and covariance require no kernel. Optional presmoothing records the supplied smoother and kernel. PACE defaults to Gaussian, following the audited non-dense default. Direct pooled-smoother calls retain their Epanechnikov default; every kernel remains overridable. |
| `methodBwMu`, `methodBwCov` | `SetOptions`: both are `"Default"`, with fixed `userBwMu=0.05*range` and `userBwCov=0.10*range`. GCV is an explicit option, **not the current default**. | Dense has no smoothing bandwidth by default. PACE uses the same fixed 5%/10% defaults on estimated inputs. Numbers, whole-curve `CurveCV`, and the mean smoother's KernelSmoothing selectors pass through to the existing smoothers. Existing `CurveCV` deliberately differs from fdapace's selectors. |
| `kFoldMuCov` | `SetOptions`: 10. `R/GetSmoothedMeanCurve.R`, `GetSmoothedMeanCurve` passes it to mean CV; `R/GetSmoothedCovarSurface.R`, `GetSmoothedCovarSurface` passes it to covariance CV. | Dense uses no folds. Existing `CurveCV` also defaults to ten folds. It requires an RNG when `folds<nobs(fd)` and uses deterministic leave-one-curve-out when they are equal; folds above the number of curves throw. Holding out entire curves respects subject dependence. |
| `useBinnedData` | `SetOptions`: `"AUTO"`; the exact rule is below. `useBinnedCov` is a different option: true except when `n<128` and the mean number of observations is below 3. | Dense does not bin. PACE defaults to `binning=:auto` and uses the audited bin-count decision below. It reuses the existing covariance smoother's weight-preserving pair snapping and aggregation, while leaving the mean and work grid unchanged. Direct covariance-smoother calls retain `binning=nothing`. |
| `methodMuCovEst` | `SetOptions`: Sparse uses `"smooth"`; Dense, DenseWithMV, and `p>>n` use `"cross-sectional"`. | Complete common-grid data use cross-sectional estimates by default. Irregular or masked data use PACE pooled smoothers; `method=:pace` forces this route on regular data too. |
| `methodXi` | `SetOptions`: Dense and `p>>n` use `"IN"`; DenseWithMV uses `"CE"`. Sparse normally uses `"CE"`, but uses `"IN"` if every curve has **more than 20** observations and the largest spacing, including either boundary gap to the pooled range, is at most 6% of that range. | Dense scores use the stored quadrature weights. PACE always uses conditional-expectation scores. There is no additional data-driven switch between these formulas. |
| Measurement error | `SetOptions`: `error=TRUE`, even for Dense. `R/GetCovDense.R`, `GetCovDense` uses a second-difference noise estimate on its cross-sectional route. | Dense stores `noise_variance=nothing` and defaults to `presmooth=nothing`. It assumes negligible measurement error, following Ramsay and Silverman (2005), Chapter 8. Noisy dense curves need explicit presmoothing or PACE. |
| Smoothed noise estimator | `PC_CovE` uses the rotated diagonal; `SetOptions` supplies the middle-half `rotationCut=c(.25,.75)`, while the internal formal default is `c(0,1)`. | `SmoothedCovariance` and PACE default to the rotated diagonal over the full observed domain, selected by the issue #17 Discovery benchmark. Explicit `noise_method=:surface_diagonal` and `noise_interval=:middle_half` retain the milestone 2 rule. |
| `rho` | `SetOptions`: `methodRho="vanilla"`, `userRho=NULL`. This means no extra regularization; details are below. | `rho` regularization is not implemented. No ridge is silently added. |

The remembered claim that fdapace always uses `max_k=min(20,m-2)` is therefore
incorrect for 0.6.0. So is a description of GCV as its default bandwidth
selector. The dense component cap follows the audited fdapace default, replacing
the milestone plan's cap of 20.

## Bandwidth candidate grids

These are optional fdapace selectors. They are not run when its fixed
`"Default"` bandwidths are used.

`R/GCVLwls1D1.R`, `GCVLwls1D1` constructs ten geometrically spaced mean
bandwidths. Let `r` be the pooled time range, `p` the local polynomial degree,
and `Minb(t,q)` the largest difference at lag `q` among the sorted unique
times (`R/Minb.R`, `Minb`). Its starting bandwidth is:

- Sparse: `h0=2.5*Minb(t,p+2)`, first multiplying `Minb` by 0.75 if it exceeds
  `r/4`.
- DenseWithMV: `h0=2*Minb(t,p+1)`.
- Other designs: `h0=1.5*Minb(t,p+1)`.

A NaN start becomes `0.2*r` for a Gaussian kernel and otherwise errors.
After setting `h0=min(h0,r)`, the candidates are
`sort(h0*q.^(0:9))`, where `q=(r/(4*h0))^(1/9)`. Thus their endpoints are
`h0` and `r/4`, in either order. If every GCV score is infinite, it retries
on 20 equally spaced points from the largest candidate to `r`. A remaining
all-infinite grid errors. Exact ties choose the largest bandwidth.

`R/GCVLwls2DV2.R`, `GCVLwls2DV2` uses
`r=sqrt(2)*diff(range(obsGrid))`. The initial minimum comes from
`R/GetMinb.R`, `GetMinb`, whose current path calls `BwNN(t,k=3,onlyCov=TRUE)`:
the worst-case distance to three unique neighbors in the pooled within-curve
time-pair cloud, using the maximum coordinate distance. A Gaussian kernel
multiplies that start by 0.2. After `h0=min(h0,r/4)`, the ten candidates are
`h0*q.^(0:9)`, with `q=(r/(4*h0))^(1/9)` when `h0<r/4`; the equality branch
targets `r/2`. Candidates are evaluated from largest to smallest and an
invalid local fit interrupts the descending search. With the default
`useBW1SE=FALSE`, ties choose the lowest candidate index (the smallest
bandwidth), regardless of the descending evaluation order. The code computes
a larger retry grid when the optimum is at the
edge, but `maxIter=1` means that grid is not evaluated in a second iteration.

Covariance `"CV"` uses this same candidate machinery, with folds of **raw
covariance pairs**, not folds of curves. Mean `"CV"` instead leaves out
whole curves (`R/CVLwls1D.R`, `CVLwls1D`). Its ten candidates are
`2.5*r/U*(U/2.5)^((j-1)/10)` for `j=1:10`, shifted so that the smallest is
`h0`, where `U` is the number of unique times. The mean-CV start is
`Minb(t,p+2)` for Dense and 2.5 times that otherwise, reduced by 0.75 if it
exceeds `r/4`. The optional `"GMeanAndGCV"` rule combines the selected GCV
bandwidth and a minimum bandwidth geometrically. Existing package `CurveCV`
uses supplied candidates and holds out whole curves for both mean and
covariance; it is a deliberately different selection rule.
[`pace_bandwidth_candidates`](@ref) exposes the ten audited GCV mean and
covariance candidates for optional whole-curve `CurveCV`; it does not run
fdapace's GCV or pair-level covariance CV. Its default kernel convention is
Gaussian, matching PACE, and an explicit kernel selects the corresponding
candidate convention.

## The automatic binning rule

`R/GetBinnedDataset.R`, `GetBinnedDataset` calls `R/GetBinNum.R`, `GetBinNum`
when `useBinnedData="AUTO"` and `numBins=NULL`. Let `m_i` be each curve's
observation count. It sets `m=median(m_i)` for Sparse and `m=max(m_i)` for
other designs, and then:

1. If `m<=20`, it does not bin, regardless of `n`.
2. Otherwise, `m>400` initially chooses 400 bins.
3. If `n>5000`, set `mstar=max(20,400+19*(5000-n)/2250)`. Use
   `ceil(mstar)` bins if `mstar<m`; otherwise do not bin.
4. If neither applicable condition selected a bin count, do not bin.

An explicitly supplied `numBins` bypasses this automatic count. `FPCA` skips
this binning step for `usergrid=TRUE` or `useBinnedData="OFF"`. When binning
does occur, it can reduce `nRegGrid`. The dense fixtures explicitly disable
binning so they test the stated input grid.

PACE adopts this **bin-count decision**. The helper's mean-candidate and
bin-count rules use the audited `R/IsRegular.R`, `IsRegular` classification:
with `u` unique pooled observed times and `N` total observations, let
`coverage=N/(n*u)`. Full coverage is Dense when `u>=8` and Sparse otherwise.
Incomplete coverage above `0.8` is DenseWithMV; all remaining cases are Sparse.
This reference classifier does not alter the package's `method=:auto`
dispatch, which uses dense FPCA for complete `GridFunctionalData` and PACE
for irregular or masked data. The resolved bin count and decision inputs are
stored on the fit. It reuses milestone 2's covariance-pair aggregation: times
are snapped, weighted products are combined within each subject and cell,
and multiplicities and diagonal exclusions are preserved. This differs from
fdapace's preprocessing of the raw observations within curves. The package
leaves the pooled mean and work-grid length unchanged. Consequently binned
fits need not reproduce fdapace's complete binned-data path.

## Noise variance and conditional-expectation scores

On fdapace's smoothed route, `R/PC_CovE.R`, `PC_CovE` estimates the diagonal
observed variance by one-dimensional local linear smoothing of squared
residuals. It estimates latent diagonal variance using the rotated
off-diagonal smoother `RotateLwls2DV2`, rather than reading the diagonal of
the final two-dimensional surface. It trapezoidally integrates their
difference on the retained regular-grid points inside `rotationCut`, and
divides by the **nominal** interval length
`(b-a)*(rotationCut[2]-rotationCut[1])`. `SetOptions` defaults this interval
to the middle half, `c(0.25,0.75)`; the internal `PC_CovE` function's own
formal default is instead `c(0,1)`. `GetSmoothedCovarSurface` resets a
negative result to `1e-6`, but leaves zero and tiny positive values unchanged.

The package's default `SmoothedCovariance` uses the rotated diagonal over the
full observed domain (`noise_method=:rotated, noise_interval=:full`), following
the issue #17 Discovery benchmark and Mladen's decision on 25 October 2026.
PACE inherits these defaults. This matches the internal `PC_CovE` formal
interval, while fdapace's `FPCA` entry point uses the middle half. Negative
estimates truncate to **zero**. The raw-square bandwidth defaults to the
covariance bandwidth and can be supplied separately. The milestone 2 rule
remains available with `noise_method=:surface_diagonal,
noise_interval=:middle_half`; it interpolates exact interval endpoints.
The rotated construction and departures are detailed below.
On fdapace's cross-sectional route, `GetCovDense` instead
estimates noise by the mean squared second difference divided by
`choose(4,2)=6`, and subtracts it from the sample covariance diagonal unless
`userSigma2` is supplied. Our dense route does not perform this operation.

`R/GetCEScores.R`, `GetCEScores` replaces a NULL noise variance by zero, forms
`Sigma_Y=fittedCov+sigma2*I`, and interpolates to each subject's times.
`src/GetIndCEScoresCPP.cpp`, `GetIndCEScoresCPP` solves this system with an
Eigen LDLT factorization. There is **no** zero/tiny-noise threshold, ridge
floor, or pseudoinverse in this active score path. A singular or poorly
conditioned system is therefore not automatically regularized. An unused
`pinv` helper in the separate leave-one-out C++ source is not the active
solver.

The covariance used in that `Sigma_Y` is the **full positive-semidefinite
fitted covariance up to `max_k`**, not only the selected `K` components:
`GetEigenAnalysisResults` constructs `eigObj$fittedCov` before selecting K,
and `FPCA` passes that object to `GetCEScores`. `FVEfittedCov` controls the
separate user-returned `fittedCovUser`; it does not change this score
covariance. The numerator of the conditional expectation and its returned
score covariance use the selected eigenfunctions and eigenvalues.

PACE follows that full-covariance score denominator. It uses a Cholesky
factorization, rather than fdapace's LDLT solve. A non-positive-definite
subject system throws and suggests supplying `noise_variance`. It uses the
nonnegative `SmoothedCovariance` value, including a truncation to zero, or
the supplied nonnegative number. It does not copy fdapace's absolute `1e-6`
replacement for a negative variance: that replacement depends on the data's
units. No noise floor is applied during scoring.

The optional non-vanilla `rho` path in `R/FPCA.R`, `FPCA` **replaces** the
noise variance passed to `GetCEScores`; it is not an additional diagonal
term on top of that noise variance. `R/GetRho.R`, `GetRho` first updates a
residual variance twice. For `methodRho="trunc"`, its 50 candidates span
one to ten times that updated variance. For `"ridge"`, they are
`exp(seq(-13,-1.5,length.out=50))*R`, where
`R=sqrt((integral(mu^2)+sum(lambda))/(b-a))`. It minimizes squared
discrepancy between integrated reconstructed variance and `sum(lambda)`.
Above 2048 curves, `FPCA` estimates it on a random subset of 2048. This
regularization, penalized/smoothed FPCA, and FACE remain unimplemented.

## [Rotated noise variance: milestone 5, issue #17](@id fdapace-rotated-noise)

The 2026-10-25 audit inspected the versioned fdapace **0.6.0** archive above,
including executable C++, and the installed R 4.5.1 function bodies:
[`R/RotateLwls2DV2.R`](https://github.com/cran/fdapace/blob/0.6.0/R/RotateLwls2DV2.R),
[`src/RrotatedMullwlsk.cpp`](https://github.com/cran/fdapace/blob/0.6.0/src/RrotatedMullwlsk.cpp),
[`R/PC_CovE.R`](https://github.com/cran/fdapace/blob/0.6.0/R/PC_CovE.R),
[`R/SetOptions.R`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R), and
[`R/GetSmoothedCovarSurface.R`](https://github.com/cran/fdapace/blob/0.6.0/R/GetSmoothedCovarSurface.R).
The wrapper comment claiming quadratic terms in both directions is inaccurate.

For a pair `(s,t)`, the rotated coordinates are
`u=(s-t)/sqrt(2)` across the diagonal and `v=(s+t)/sqrt(2)` along it.
At a diagonal query `(q,q)`, the C++ weighted regression has exactly the
columns `1, u^2, v-sqrt(2)*q`. There is no linear across-diagonal term,
quadratic along-diagonal term, or cross term. The product kernel acts on
`u/h` and `(v-sqrt(2)*q)/h`. Compact kernels restrict both coordinates to
their bandwidth windows; Gaussian uses all pairs. Julia scales the polynomial
columns by bandwidth for conditioning, which preserves the fitted intercept.
It reuses the covariance smoother's three-column normal-equation solver.

`PC_CovE` takes **only the first covariance bandwidth** and supplies that
scalar to both the rotated smoother and the raw-square `Lwls1D` smoother.
The wrapper repeats it in both rotated axes. There is no extra `sqrt(2)`
bandwidth multiplier and no bandwidth selection inside `PC_CovE`.
Julia likewise uses its selected covariance bandwidth in both rotated axes;
its existing `diagonal_bandwidth` override still affects only the raw squares.

The integration domain is `[a,b]=extrema(obsGrid)`. A cut `(l,r)` keeps only
existing `regGrid` points satisfying
`q > a+(b-a)*l-buffer` and `q < a+(b-a)*r+buffer`, where
`buffer=10*.Machine$double.eps*max(abs(obsGrid))`.
The trapezoidal integral is divided by the **nominal** length `(b-a)*(r-l)`,
even if the retained points cover less. No boundary points are inserted.
`SetOptions` supplies `(0.25,0.75)`; the internal `PC_CovE` formal default is
`(0,1)`. Julia exposes these as `:middle_half` and `:full` (default), with the
buffer computed at the fit's floating-point precision. Its surface-diagonal
estimator retains its original work-grid domain and interpolated boundaries.

`noise_method=:rotated` matches this `PC_CovE` calculation at fixed bandwidths.
The public covariance surface still uses unrotated local linear smoothing.
Intentional departures are explicit:

- Negative noise estimates truncate to zero, with the raw value and flag
  retained. fdapace's enclosing surface routine changes a negative estimate
  to `1e-6`; `PC_CovE` itself returns the raw estimate.
- Undefined or singular fits throw, including windows with one point.
  fdapace's C++ returns that point's response in a one-point window and
  does not perform Julia's rank checks. At least two retained integration
  points are required in Julia.
- The existing observation/subject weights, supplied mean interpolation,
  optional binning, and explicit raw-square bandwidth override remain in
  force. These are extensions of the fixed-bandwidth calculation, not a
  claim of identical end-to-end R FPCA tuning.

`test/reference/fdapace_m5_sigma2.R` generates `test/data/m5_fdapace_sigma2.csv`
from the unchanged milestone-2 fixture. It calls `PC_CovE` itself, for both
intervals, two bandwidths (`0.35,0.65`), Epanechnikov/Gaussian kernels, and
aligned, unaligned, and extended work grids. Observation and subject weights
enter through the `BinnedRawCov` weighted-input interface without moving or
merging times. Raw squares and weights are sorted together before entry:
`PC_CovE` sorts diagonal values but does not itself permute `win2`.
All raw variances and fitted diagonals are compared at relative error below
`1e-8`. The fixture tests the retained-grid/nominal-length rule directly.

The benchmark-only `projected_fgm` reference is source-audited against FGM's
historical `bsplinekit` commit
[`6f0eac3d5cbe0fad93e88dcad28f389d89b61495`, `src/doubly.jl`](https://github.com/mlakolar/FunctionalGraphicalModels.jl/blob/6f0eac3d5cbe0fad93e88dcad28f389d89b61495/src/doubly.jl).
That code keeps weighted covariance eigenvalues exceeding
`100eps(Float64)*max(maximum(abs,eigenvalues),floatmin(Float64))`, reconstructs
the covariance from all such components, and estimates noise by the arithmetic
mean of raw-square excesses at indices `ceil(q/4):floor(3q/4)` (clamped to
`1:q`), followed by truncation at zero. The benchmark reuses Julia's weighted
eigen step with relative cutoff `100eps(Float64)` and this index-average rule.
It compares the diagonal rule on the same smoothed inputs as the other four
estimators, including the requested weighting and bandwidth choices; it is
not a rerun of FGM's complete pipeline. The relative-cutoff expression agrees
with historical FGM for the non-subnormal spectra of these simulations.
There is no projected-diagonal package option.

[Noise-variance estimators (issue #17)](m5_sigma2_report.md) gives the Discovery
results, benchmark-selected defaults and the two-cell harness smoke check.
The Discovery benchmark led to `:rotated` with `:full` as the package defaults.

## Eigenanalysis, signs, and component selection

`R/GetEigenAnalysisResults.R`, `GetEigenAnalysisResults` computes ordinary
matrix eigenvectors of the covariance surface `G`. It keeps eigenvalues
`d>=0`, including exact zeros, drops strictly negative ones, and truncates
the remaining descending list to `max_k`. It does not record a dropped
negative mass or apply a relative positive-eigenvalue cutoff.

It reports eigenvalues `lambda=h*d`, with the first regular-grid spacing
`h=grid[2]-grid[1]`. Each ordinary eigenvector is separately divided by its
trapezoidal L2 norm. The result has individual trapezoidal norm one but
need not have **mutually orthogonal** eigenfunctions under those weights.
Its sign satisfies the **unweighted** `sum(phi*muWork)>=0`. It has no
near-zero fallback. If `muWork=NULL`, the internal function substitutes
`1:m`; the `FPCA` entry point supplies its estimated mean.

The package diagonalizes `sqrt(W)*G*sqrt(W)`, then divides the eigenvectors
by `sqrt(w)` using the same stored trapezoidal weights as the scores.
Eigenfunctions are mutually orthonormal under this quadrature. It retains
eigenvalues above `eigen_tol*maximum(abs,spectrum)` with default
`eigen_tol=1e-10`, and records the number, absolute mass, and share of the
strictly negative spectrum. The relative cutoff is an explicit numerical
choice, rather than a statistical tuning rule from fdapace. Negative mass
is measured before the `max_k` cap.

The sign agrees with fdapace when the mean overlap is nonzero. If its
absolute value is at most
`sign_tol*norm(phi)*norm(mu)`, with default `sign_tol=1e-12`, the largest
absolute eigenfunction entry is made positive; ties use the first entry.
This fallback follows the convention used by FunctionalGraphicalModels.jl.
Both numerical tolerances are stored on the fit. They are explicit package
choices that the milestone leaves to the implementation.

Both implementations compute FVE **after** the `max_k` truncation:
`cumsum(lambda)/sum(lambda)` over that retained positive spectrum. Thus the
last retained value is one even if the cap discarded positive components.
`GetEigenAnalysisResults` first uses `>=` to find its threshold crossing;
`R/SelectK.R`, `SelectK` later uses `cumFVE>threshold-eps`. The package uses
the specified `>=` boundary. `R/SubsetFPCA.R`, `SubsetFPCA` truncates the
reported `cumFVE` to selected K; our fit retains the whole curve for
inspection. The fixtures use an extra full-K run to preserve that curve.

For AIC/BIC, `SelectK` evaluates `GetLogLik(K)+C*K`, with `C=2` or `log(n)`
and `R/GetLogLik.R`, `GetLogLik` returning minus twice the log likelihood
up to a constant. It stops at the first increase instead of searching all
candidate K. The likelihood uses the K-component covariance and noise,
which is distinct from the full fitted covariance used by CE scoring.
When `methodSelectK` is numeric, `SetOptions` first sets `FVEthreshold=1`,
and `SelectK` then checks the fixed requested K against available components.

PACE stores the criterion for every candidate from 1 through `max_k`, then
applies the first-increase rule. Beyond the retained positive rank, extra
zero-variance components leave the likelihood unchanged and increase only
the penalty. It additionally records the
global minimizer. Its criterion includes the Gaussian normalization constant
`sum_i N_i*log(2pi)`, whereas fdapace's `GetLogLik` omits this constant.
Reference files retain both conventions. The offset changes neither the
first-increase choice nor the global minimizer.

## PACE supplied-input and prediction reference checks

The same reference script also writes `m3_sparse_input.csv`,
`m3_sparse_mean.csv` (work grid and mean), `m3_sparse_surface.csv`,
`m3_sparse_noise.csv`, `m3_sparse_heldout.csv`, and `m3_fdapace_pace.csv`. The sparse fixture
contains 40 deterministic curves with 2 to 8 observations. Its fixed-K run
uses `userMu`, `userCov`, `userSigma2`, `error=TRUE`, `methodXi="CE"`,
`useBinnedData="OFF"`, and `methodRho="vanilla"`. Separate AIC, BIC, and
FVE runs record selected component counts and the full available criterion
curves. R and fdapace versions appear in the CSV header comments.

The PACE fixture uses sine eigenfunctions that vanish at both work-grid
endpoints. Rectangular and trapezoidal quadrature therefore agree for this
surface. The Julia golden test exercises the public `fit(FPCA, ...)`
interface directly, comparing eigenvalues, eigenfunctions, training scores,
conditional score covariances, fitted covariance, prediction, and component
selection at relative error below `1e-8`. It requires no alternative
eigen-step convention. Separate analytic tests validate public PACE scores
and uncertainty for affine eigenfunctions with nonzero boundary values.

Three further source details affect these comparisons:

- `GetUserMeanCurve` interpolates `userMu` with a spline. This package's
  supplied-mean contract uses linear interpolation. The fixture mean is
  affine so both conventions agree.
- `predict.FPCA` first interpolates the covariance onto the union of training
  observation times. `GetCEScores` then adds the noise diagonal on that
  union and interpolates the resulting observed covariance to new times.
  At new times absent from the union, this can interpolate the noise
  diagonal itself. This package interpolates the latent projected surface
  and adds `noise_variance*I` at the new subject's actual times. Held-out
  reference curves therefore use times present in the training union but
  absent from the work grid, isolating the documented quantity.
- `SubsetFPCA` truncates `xiEst`, `phi`, and `lambda` to the selected K but
  leaves `xiVar` at its full retained size. The reference records the selected
  upper-left block of `xiVar`, matching this package's `K × K × n` contract.

## End-to-end and default reference checks

`test/reference/fdapace_m3.R` also writes `m3_fdapace_end2end.csv` and
`m3_fdapace_defaults.csv`, using R 4.5.1 and fdapace 0.6.0. Their versioned
header comments describe the settings. Both align noise with
`userSigma2=0.12`, isolating mean, covariance and score comparisons from noise
estimation. These fixtures predate the rotated default; fdapace's enclosing
`FPCA` still uses a different integration interval and negative-value rule.

The fixed-bandwidth run estimates its mean and covariance from the sparse
fixture. It uses `userBwMu=0.30`, `userBwCov=0.40`, `kernel="epan"`,
`useBinnedData="OFF"`, `useBinnedCov=FALSE`, observation weighting,
`methodXi="CE"`, `methodRho="vanilla"`, fixed `K=3`, and `max_k=5`.
Its 41-point work grid equals the observed-time union. Thus mean evaluation
before forming residual products agrees in both implementations. Public
Julia fits reproduce those mean and covariance values at relative error
below `1e-8`. The eigenanalysis and CE golden comparisons explicitly use
fdapace's rectangular operator and separate trapezoidal normalization through
the internal compatibility arguments described for the dense reference.
They compare the documented fdapace quantities at the same `1e-8` tolerance.
They do not assert equality with the package's trapezoidal operator.

The defaults reference preserves a native `FPCA` call with only
`userSigma2=0.12` supplied. It additionally records the direct smoothed mean
at the 51 work-grid points and a mean-aligned `FPCA` call. This alignment is
necessary because native fdapace first smooths its mean on the pooled
observation-time union and interpolates it to the work grid. The Julia fit
smooths directly at the work grid, then linearly interpolates that mean to
observation times when forming covariance residuals. The aligned R call
supplies `userMu` at the observed-time union equal to that interpolated direct
mean. Direct mean smoothing, the aligned covariance, and the compatible
eigenanalysis and CE scores are golden comparisons at relative error below
`1e-8`. The ten geometric candidate vectors are also compared with the
audited formulas using fdapace's actual `GetMinb`/`BwNN` calculation.

The complete public fits retain their original mean evaluation, quadrature,
and eigenfunction normalization. Their comparisons with native fdapace are
explicit **cross-method checks**, not golden tests:

| Quantity | Fixed-bandwidth relative error / bound | Defaults relative error / bound |
|:--|--:|--:|
| Work-grid mean | Golden: below `1e-8` | `2.90e-5` / `5e-5` |
| Work-grid covariance | Golden: below `1e-8` | `1.71e-4` / `3e-4` |
| Eigenvalues | `2.18e-3` / `5e-3` | `5.73e-4` / `1e-3` |
| Eigenfunctions | `1.01e-1` / `1.2e-1` | `3.29e-2` / `4e-2` |
| Scores | `4.19e-3` / `1e-2` | `5.09e-3` / `1e-2` |

The bounds describe these fixtures and their specific finite-grid conventions.
They are not universal approximation guarantees or relaxed golden tolerances.
The supplied-noise comparison does not validate equality of estimated noise
variances. Sparse simulation recovery tests separately check that estimator.

## Dense denominator and reference checks

`R/GetMeanDense.R`, `GetMeanDense` uses the cross-sectional mean.
`R/GetCovDense.R`, `GetCovDense` calls
`cov(ymat,use="pairwise.complete.obs")`. Without missing values its
denominator is **n-1**, matching this package. The separate supplied-mean
branch also divides centered products by `n-1` for complete data.
FunctionalGraphicalModels.jl currently divides by n; milestone 4's migration
must account for the resulting eigenvalue and score-variance scaling.

Run `Rscript test/reference/fdapace_m3.R` from the repository root to
recreate `test/data/m3_dense_input.csv` and
`test/data/m3_fdapace_dense.csv`. Each CSV begins with R and fdapace
versions. The script has no random draws. It uses 18 rank-five curves on
21 grid points, `dataType="Dense"`, cross-sectional estimation,
`error=FALSE`, integration scores, `max_k=5`, no binning, and a fixed K of 3.
A second `FVE(0.99)` run selects K=4; a full-K run records the full FVE curve.
The first three reference eigenvalues are approximately
`4.448802620311068`, `1.165542244194415`, and `0.3479570319234146`.

Julia's reference comparison reproduces fdapace's rectangular operator
weights and separate trapezoidal eigenfunction normalization through
internal eigen-step arguments. Means, covariances, eigenvalues,
eigenfunctions, FVE, and scores are compared at relative tolerance `1e-8`.
This validates the documented fdapace quantity. It does not claim that our
default trapezoidal eigendecomposition equals fdapace on a finite grid.
Separate grid-refinement checks test convergence of the two conventions.

## Plotting source audit

Milestone 5 task 1 audits **fdapace 0.6.0**, using the CRAN source tag and
the locally installed R package of that version. Relevant source files:
[CreateModeOfVarPlot.R](https://github.com/cran/fdapace/blob/0.6.0/R/CreateModeOfVarPlot.R),
[CreateScreePlot.R](https://github.com/cran/fdapace/blob/0.6.0/R/CreateScreePlot.R),
[CreateDesignPlot.R](https://github.com/cran/fdapace/blob/0.6.0/R/CreateDesignPlot.R),
[DesignPlotCount.R](https://github.com/cran/fdapace/blob/0.6.0/R/DesignPlotCount.R), and
[CreatePathPlot.R](https://github.com/cran/fdapace/blob/0.6.0/R/CreatePathPlot.R).

- Modes: the default `k=1`, `rainbowPlot=FALSE` route draws the mean and
  shaded regions bounded by multipliers -2, -1, 1, and 2. It has no formal
  multiplier argument. The optional rainbow route uses 257 values from -2
  to 2, despite its prose mentioning -3 and 3. Its default y range instead
  comes from ±3 standard deviations of the **first** component, even when
  plotting another component. Our API requires k, returns the four default
  boundary curves, and lets Plots scale to the actual curves; it does not
  reproduce that clipping-prone y range or implement the rainbow option.
- Scree: bars are `c(cumFVE[1], diff(cumFVE))`, with a cumulative line and
  points, on a vertical range of 0 to 1.05. `CreateScreePlot` uses the
  returned, selected-K `cumFVE`. Our stored full retained spectrum is
  exposed instead, with selected K and its threshold recorded separately.
- Design: defaults are `isColorPlot=TRUE`, `noDiagonal=TRUE`, and
  `addLegend=TRUE`. `DesignPlotCount` adds one count per subject for every
  **ordered** observed pair, then zeros the diagonal. The color path caps
  displayed counts at four and uses categories 1, 2, 3, 4+; its default
  colors are black, blue, green, red and marker sizes increase by category.
  The noncolor path records pair presence only. Fully observed common
  grids longer than 101 points take a special 101-square approximation
  that bypasses diagonal removal. We keep exact pairs, omit the diagonal
  uniformly by default, expose uncapped counts, and use Plots' color scale
  with the same count categories, so backend themes and color overrides
  remain available. There is no dense-grid approximation.
- Paths: missing subset selects all curves; K defaults to the fitted
  selection through `fitted.FPCA`, on `workGrid`. Original points are shown
  when `inputData` exists; `obsOnly=FALSE` and `showMean=FALSE`. Our API
  takes one curve index and returns its original points and existing
  `reconstruct` output. Original input is copied onto FPCA fits, including
  before dense presmoothing. Grid interpolation follows the existing
  Julia reconstruction contract, rather than introducing a plotting solver.

`Rscript test/reference/plotting_m5.R` regenerates `test/data/m5_design.csv`
with R 4.5.1 and fdapace 0.6.0. It records exact ordered-pair counts with
and without diagonals. Other plot-data checks use analytic rank-two curves
and direct means, eigenvalues, eigenfunctions, and conditional expectations;
they require no plotting package or R execution in CI.

## Reproducing the PACE fixtures

`Rscript test/reference/fdapace_m3.R` regenerates the PACE golden fixtures
with R 4.5.1 and fdapace 0.6.0.

References: Yao, Müller, and Wang (2005), *Functional Data Analysis for
Sparse Longitudinal Data*, JASA 100, 577–590;
[doi:10.1198/016214504000001745](https://doi.org/10.1198/016214504000001745).
Ramsay and Silverman (2005), *Functional Data Analysis*, second edition,
Chapter 8; [doi:10.1007/b98888](https://doi.org/10.1007/b98888).

## Derivative source audit

The derivative reference is CRAN **fdapace 0.6.0**, specifically
[`FPCAder.R`](https://github.com/cran/fdapace/blob/0.6.0/R/FPCAder.R),
[`SetDerOptions.R`](https://github.com/cran/fdapace/blob/0.6.0/R/SetDerOptions.R),
`Lwls1D.R`, `CPPlwls1d.cpp`, `ConvertSupport.R`, and `GetCEScores.R`.
The method references are Liu and Müller (2009), *Estimating derivatives for
samples of sparsely observed functions, with application to online auction
dynamics*, and Dai, Müller and Tao (2018), *Derivatives and principal component
analysis of curves*, *Statistica Sinica* 28, 779–801.

### Audited rules

- `SetDerOptions` defaults to `method="FPC"`, order `p=1`, and
  `kernelType="gauss"`, independently of the original FPCA kernel.
- If neither derivative bandwidth is given, a common `bw` is used. Without an
  explicit `bw`, it is `p*0.10*diff(range(workGrid))` when nonnull `sigma2`
  divided by the sum of selected `lambda` is at least 0.01; otherwise it is
  `p*0.05*diff(range(workGrid))`. Separate `bwMu` and `bwCov` require both.
- `FPC` supports orders one and two. `muDer` smooths the stored `mu` and
  `phiDer` smooths each stored `phi` on `workGrid`, both with polynomial degree
  **equal to p**, derivative order `p`, and unit point weights. Original scores
  reconstruct derivatives as `muDer + phiDer %*% xiEst[i,]`. Order two emits an
  experimental warning. The C++ fitter multiplies the derivative coefficient
  by `p!`, with the sign correction for its reversed polynomial coordinates.
  Gaussian windows include all inputs; compact kernels use their bandwidth
  window. The package uses KernelSmoothing.jl's equivalent derivative fit and
  validates unsupported kernels/orders rather than using a fallback kernel.
- `DPC` supports first derivatives. It re-estimates the raw pooled mean:
  degree one at `obsGrid` for centering, degree two/derivative one at `workGrid`
  for `muDer`. It forms raw covariance bins but does not use them subsequently
  in the implemented DPC branch. `cov10` is the columnwise first difference
  of `smoothedCov`, divided by the first grid spacing, then interpolated from
  `seq(min(workGrid),max(workGrid),length=nWorkGrid-1)` to `workGrid`.
  This support assignment uses the endpoints, not interval midpoints.
  Rowwise degree-two derivative smoothing gives `cov11`, then symmetrization.
- Native DPC diagonalizes `cov11`, keeps nonnegative eigenvalues, scales them
  by the first grid spacing, and chooses the first cumulative FVE reaching
  **0.9999**. Eigenvectors are normalized by trapezoidal integrals. Its
  cross-covariance `zeta=crossprod(cov10,phiDer)*gridSize` supplies CE score
  numerators; the original fitted covariance supplies the denominator, with
  noise `max(sigma2,rho)` when `rho` exists. For the score prior, however, it
  substitutes eigenvalues of the twice-**unsmoothed-differenced** covariance.
  Negative conditional score-covariance eigenvalues are clipped to zero.
  In a dense cross-sectional fit `sigma2` can be NULL; the native DPC call then
  fails in its `ifelse` noise expression unless that field is supplied.
- `FPC1` is the only additional implemented branch. It supports first order,
  re-estimates the pooled mean as DPC does, and uses `Lwls2DDeriv` (total degree
  one, partial orders `(1,0)`) on binned raw off-diagonal covariance products.
  It multiplies that derivative surface by `phi/lambda` and grid spacing.
  This optional method is not implemented here because it needs a separate
  bivariate derivative smoother. Unrecognized method strings in this package
  raise `ArgumentError`.

### Package DPC quantity and reference validation

This package uses the linear local-polynomial derivative operator `D` on both
arguments: `C10=D*G`, `C11=D*G*D'`. The mean derivative applies the corresponding
mean-bandwidth operator to the **stored mean**, respecting supplied means.
Dense `G` is the sample covariance. PACE `G` is the full positive-semidefinite
projection retained by the original fit. This preserves consistency between
the derivative prior and its curve-score cross-covariance, supports nonuniform
work grids, and avoids introducing a second differentiation implementation.
The eigenstep uses stored quadrature weights `W`, as all package FPCA does.
For derivative functions `Psi`, the PACE numerator is the interpolation of
`G*D'*W*Psi`; its denominator is the original `G(t_i,t_i)+sigma2*I`.
The same derivative eigenvalues supply the score prior. This is the
conditional-expectation construction of the derivative process, not native
fdapace's unsmoothed-spectrum substitution. No posterior clipping is needed
for the resulting positive-semidefinite joint covariance.

Recreate `test/data/m5_fdapace_der.csv` with:

```sh
Rscript test/reference/fdapace_m5.R
```

The script records its R and fdapace versions in the CSV header. It reuses the
fixed dense and sparse milestone 3 input CSVs (adding `0.2*t^2` to the supplied
sparse mean so its second derivative is nonzero), `bwMu=.12`, `bwCov=.15`, Gaussian
kernels, and three selected components. Original eigensystems are aligned to
the package's trapezoidal convention in R before `FPCAder` runs. `FPC_order1` and
`FPC_order2` rows refer to FPC derivative orders 1 and 2. Both orders compare the mean, functions, original
spectrum/scores, and reconstructed curves at relative error below `1e-8`.

For DPC, native `FPCAder` runs on both designs (dense `sigma2` is explicitly
set to zero for its noise-expression issue). Its mean, functions, eigenvalues
and scores are retained as `native_DPC`, exposing the native difference.
The native mean derivative is compared directly with `fit(SmoothedMean, ...)`
on the original pooled observations, using degree two, derivative order one,
the Gaussian kernel and `bandwidth=.12`, at relative error below `1e-8`.
This checks native `FPCAder` output independently of its covariance and score
prior. Native DPC re-estimates this mean even when the original FPCA fit has
a supplied mean; the package DPC differentiates the stored mean instead.
For the dense common-grid fixture, these quantities coincide, and the package
DPC mean derivative is also compared directly with native DPC at `1e-8`.
`DPC` rows compare the documented aligned quantity above: fdapace `Lwls1D`
in both covariance arguments, the stored mean, weighted eigenanalysis, and
fdapace `GetCEScores` with the matching derivative prior. Tests compare mean,
derivative covariance, functions, eigenvalues, scores and curves at relative
error below `1e-8`; sparse conditional covariances have the same tolerance.
The native DPC spectrum is explicitly checked to differ from this aligned
spectrum. No tolerance is relaxed to claim native DPC equivalence.

The other basis reference is **fda 6.3.0**,
[`R/deriv.fd.R`](https://github.com/cran/fda/blob/6.3.0/R/deriv.fd.R).
Its default `int2Lfd(1)` requests a first derivative. It evaluates derivatives
on `10*nbasis+1` equally spaced points and calls `project.basis` into the
**original** basis. This package instead retains exact derivative basis
functions and original coefficients, as required by the derivative API;
it does not perform this projection. Analytic polynomial, Fourier, and
in-span B-spline tests validate this deliberately different quantity.

## [Confidence bands (milestone 5 task 3)](@id fdapace-confidence-bands)

Audited executable source: fdapace **0.6.0**
[`R/fitted.FPCA.R`](https://github.com/cran/fdapace/blob/0.6.0/R/fitted.FPCA.R),
[`R/SubsetFPCA.R`](https://github.com/cran/fdapace/blob/0.6.0/R/SubsetFPCA.R),
[`R/GetCEScores.R`](https://github.com/cran/fdapace/blob/0.6.0/R/GetCEScores.R),
[`src/GetIndCEScoresCPP.cpp`](https://github.com/cran/fdapace/blob/0.6.0/src/GetIndCEScoresCPP.cpp),
and [`R/GetMeanCI.R`](https://github.com/cran/fdapace/blob/0.6.0/R/GetMeanCI.R).
The installed R function bodies and versioned archive agree. There is no
`GetCIs` helper in this version; fitted bands are calculated inline.

| Item | fdapace 0.6.0 behavior | FDA behavior |
|:--|:--|:--|
| Alpha | `NULL`: no bands unless explicitly supplied; accepts values in (0,1). | Explicit uncertainty API defaults to 0.05 per milestone 5. |
| Coverage | Missing/NULL means `"band"`; `"interval"` is pointwise. Every other string falls through to the band formula without validation. | `:interval` default per task; `:band` supported; other values throw. |
| K | Defaults to `length(object$lambda)`, the selected rank. Explicit K changes the reconstruction and χ² degrees of freedom but does not subset `phi` or `xiVar` in the variance calculation. | Always selected K for the reconstruction, variance basis and critical value. |
| Conditional covariance | `xiVar`: Λ − ΛΦᵢᵀΣᵢ⁻¹ΦᵢΛ, from the CE score solve. Σᵢ uses the full retained PSD covariance and measurement-error variance, not just the selected components. | Reuses the stored selected-score covariance; the full retained covariance was already used by PACE. No new solve is run. |
| PSD adjustment | Eigendecompose `xiVar` and replace all negative eigenvalues by zero. | Clip only roundoff-negative eigenvalues; materially indefinite stored covariances throw. |
| Pointwise formula | Reconstruction ± `qnorm(1-alpha/2)` times `sqrt(diag(phi %*% omega %*% t(phi)))`. | Same formula. |
| Simultaneous formula | Reconstruction ± `sqrt(qchisq(1-alpha,K))` times the same standard deviation. | Same formula, separate per curve and per node. |
| Dense data | Returns only fitted values even when alpha is supplied. | Throws because integration scores have no conditional covariance; suggests `method=:pace`. |
| Mean helper | `GetMeanCI(level=.95,R=999)` resamples curves and returns type-7 percentile intervals. Sparse samples can lose unsupported grid points; it warns. | Gap Map default B=1000, pointwise percentiles and/or a finite-grid sup-t band; fixed smoother choices; any undefined draw throws without dropping points. |

Two source issues require explicit corrections to exercise native
`fitted.FPCA` on the milestone 3 supplied-input sparse fixture:

1. `SubsetFPCA` truncates `phi`, `xiEst`, and `lambda`, but leaves `xiVar`
   at the full retained rank. With selected K smaller than `max_k`, native
   fitted bands fail with nonconformable matrices. The reference script
   verifies that failure and subsets only `xiVar` to its selected upper-left
   block. The CE denominator still contains all five retained components.
2. Supplied `userMu` leaves `bwMu=NULL`. `fitted.FPCA` nevertheless smooths
   `muObs` and `phiObs` using `bwMu` and then discards both intermediates.
   The reference sets `bwMu=.3` solely to make these unused calculations
   defined. It does not change the reconstruction or band variance.

`test/reference/fdapace_m5_bands.R` records these corrections and generates
`m5_fdapace_bands.csv` at K=1,3,5 for both coverage types. K=5 needs no
covariance subsetting. Tests compare native fitted values and both endpoint
matrices at relative error below `1e-8`. `m5_chisq.csv` separately records
R `qchisq` values for integer ranks 1–200 and central/tail probabilities.
FDA uses an integer/half-integer gamma recurrence, a positive lower-tail
series, and bracketed inversion, starting from KernelSmoothing's public
Gaussian CDF. No transitive package internals or additional runtime
dependencies are used.

For interpretation, refund **0.1-40**
[`R/fpca.sc.R`](https://github.com/cran/refund/blob/0.1-40/R/fpca.sc.R)
labels its fitted-curve covariance as model-based. Its `VarMats` expression
is `sigma2 * Z * inv(Zobs'Zobs + sigma2*inv(D)) * Z'`, the corresponding
conditional prediction-error covariance under its truncated model. It does
not represent uncertainty from estimating the FPCA model. FDA uses the
same conditional interpretation, not refund's separate corrected-band
procedures or its simulated-max critical values.

KernelSmoothing **0.1.0**, source tree
`3178afc11653d934bdb8aec4a68539fe1473ec46`, provides
`ClusterVariance(1)` and `cluster` observation labels in
`src/locpoly.jl`. Its default CR1 multiplies the sum of squared cluster
influences by `(N-1)/(N-k) * G/(G-1)`, with N observations, k local
polynomial coefficients and G clusters in the window. FDA's
`method=:sandwich` supplies subject labels, the stored smoothing choices,
`minobs=0`, and conventional standard errors, keeping the original mean
estimate as the interval center. Insufficient clusters produce an error.

## Cross-structure source audit

Reference: **fdapace 0.6.0**, CRAN source tarball
`fdapace_0.6.0.tar.gz`, SHA-256
`1d3e80d885ab8a091b5a99f7edff1e043ad878030f98eadb53d1b3062c5d3bcb`.
The authoritative sources are
[`FSVD.R`](https://github.com/cran/fdapace/blob/0.6.0/R/FSVD.R),
[`SetSVDOptions.R`](https://github.com/cran/fdapace/blob/0.6.0/R/SetSVDOptions.R),
[`FCCor.R`](https://github.com/cran/fdapace/blob/0.6.0/R/FCCor.R), and
[`DynCorr.R`](https://github.com/cran/fdapace/blob/0.6.0/R/DynCorr.R).
See [Cross-structure](cross_structure.md) for usage and reference comparisons.

| Choice | Audited source behavior | Package behavior |
|:--|:--|:--|
| FSVD component cap | `SetSVDOptions`: `min(20,length(Ly1)-2)`, despite help text saying n−1. | Same default; at least three subjects are required. A cap may exceed a grid's rank; selection uses only retained positive singular values. |
| FSVD selection | `methodSelectK="FVE"`, `FVEthreshold=.9999`. FVE uses squared singular values after the cap. Numeric `methodSelectK` sets `FVEthreshold=1`, disabling FVE selection. Invalid integers can reset to FVE(.95); excessive counts can be reduced with a warning. | `FVE(.9999)` or a positive integer; invalid counts and rank-exceeding requests throw. The full capped cumulative curve and selected count are stored. |
| Marginal FPCA when options are NULL | Calls `SetOptions` with the SVD node's inferred `dataType` and `nRegGrid` (51 by default), checks it, then calls `FPCA` for each node. | Calls the existing `fit(FPCA,fd)` for each node. Two complete grids use dense FPCA; otherwise both marginals force `method=:pace`, including a complete-grid partner, to provide measurement-error variances for joint conditional scores. Explicit dense overrides on this route throw. Requested and effective options are stored in FSVD settings. FVE(.99), grids, bandwidths, component caps and documented quadrature departures remain in force. Override other tuning using `fpca_options1/2`. |
| Dense cross covariance | Centers each node by its FPCA mean; divides by `n-2+mudf`, where `mudf` counts supplied SVD means. This is different from `GetCrCovYX`'s dense sample covariance. | Reuses `CrossCovariance` and rescales from n−1 to **n−2**. No separate dense SVD mean overrides; presmoothing is explicit on input data. |
| Sparse cross covariance | `GetCrCovYX`, Gaussian by default, local linear with exact-duplicate aggregation, `rmDiag=FALSE`, `useGAM=FALSE`. Missing bandwidths trigger Gaussian GCV using the default `l-bfgs-b` routine. `SetSVDOptions` does not implement a `methodBwCov` field despite the FSVD help text mentioning it. | Reuses `SmoothedCrossCovariance`, Gaussian, observation-pair weights, equal-time products retained, no snapping by default. Bandwidth or explicit whole-subject `CurveCV` is required, preserving the existing cross-covariance contract instead of adding pair-level GCV. Marginal kernels are independently controlled through their FPCA options. |
| Singular operator | Ordinary matrix SVD of `CrCov`; retain strictly positive values, cap, select, then multiply values by `sqrt(first_spacing1*first_spacing2)`. Functions receive separate trapezoidal normalization. | SVD of `sqrt(W1)*C*sqrt(W2)`, divided by square-root weights for the functions. Uses the same stored quadrature for operator, orthogonality and scores, including unequal nonuniform grids, as in package FPCA. Relative singular cutoff `1e-10` follows its numerical tolerance. |
| Sign | `flip=FALSE`, meaning no flip of R's arbitrary SVD signs. | Largest absolute left entry positive, paired right entry and scores flipped together; first index breaks ties. |
| Scores | `noScores=FALSE` in source, despite the help text saying TRUE. Dense uses `GetINScores`. Sparse conditions jointly on both processes. | Always computes scores: stored-weight integration for dense; joint conditional expectation otherwise. |
| Sparse CE | PSD-projects the stacked **smoothed** covariance by dropping negative ordinary eigenvalues. Adds marginal noise before support conversion. Numerators integrate marginal fitted covariance against singular functions and use singular values times opposite functions in the cross block. `regulRS="sigma2"`. | Reuses the existing eigen step with unit weights for this unweighted PSD projection, the PACE interpolation and Cholesky solver, and marginal full projected covariances for the numerators. Those integrals use stored marginal weights. Noise is added before linear interpolation. No new solver, smoother or rho rule. Singular observation covariance throws. |
| FCCor bandwidth | Required `bw`; length one replicates to five, otherwise requires five. Order: two means, two variances, cross-product. | Same; vector/tuple length one or five, or scalar. All values finite and positive. |
| FCCor kernel and grids | Actual `kern='epan'`, despite the help text saying Gaussian. Means on sorted unique pooled times; residual products smoothed at `Tout`, which defaults to those times. Observation weights one. | Same five local linear calculations through KernelSmoothing. Paired observation times must agree; an explicit output grid stays inside the observed range. |
| FCCor nonpositive variance | Replaces values ≤0 with NaN and warns; does not clip correlation. | Same, with separate affected-point masks retained. |
| DynCorr | Paired dense common-grid matrices. Subtracts each curve's time mean, then population mean. Divides each residual by its time RMS and returns individual time-averaged products. Uses trapezoidal integration divided by domain length. Missing values trigger `approx(...,rule=2)` interpolation. | Same centering, normalization and integration with stored weights. Returns both individual correlations and their arithmetic mean. Complete data required; no silent interpolation. Zero residual RMS throws instead of producing undefined values. There are no tuning defaults. |

The dense FSVD R fixture passes `shrink=FALSE` explicitly. With explicit raw
FPCA option lists, `FSVD` passes those lists to `GetINScores` without filling
all defaults; absent `shrink` otherwise causes an invalid logical operand.
This supplies the normal default and does not change the score formula.
The sparse fixture uses fixed bandwidths and supplied marginal inputs to
separate cross-structure validation from marginal bandwidth selection.
