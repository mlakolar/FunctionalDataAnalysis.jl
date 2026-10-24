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
The fixture exercises fdapace's duplicate grouping; FDA uses unsnapped,
unaggregated raw observation pairs, which have the same weighted fit.
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
| `maxK` | `SetOptions`: `min(nRegGrid-2,n-2)`, additionally capped at 20 **only for `methodMuCovEst="smooth"`**; values below 1 are reset to 1. An explicit `maxK` above `nRegGrid-2` is reduced with a message. | The dense default is `max(1,min(m-2,n-2))`; PACE uses `max(1,min(20,m-2,n-2))`, matching the respective audited routes. An explicit dense cap may use up to `min(m,n-1)` and an explicit PACE cap up to `m`; requested components above the cap or retained positive rank throw. |
| `methodSelectK` | `SetOptions`: `"FVE"`. `R/SelectK.R`, `SelectK` also accepts `"AIC"`, `"BIC"`, or a positive integer. | `ncomponents=FVE(0.99)` or a positive integer. PACE also accepts `AIC()` and `BIC()`. |
| `FVEthreshold` | `SetOptions`: 0.99 for `FPCA`. The standalone `SelectK` function separately defaults to 0.95. | `FVE(0.99)` follows the `FPCA` entry point. |
| Kernel | `SetOptions` initially chooses `"epan"` for Dense and `"gauss"` for every other data type, including DenseWithMV. A later invalid-kernel recovery branch chooses Epanechnikov for DenseWithMV. | The dense sample mean and covariance require no kernel. Optional presmoothing records the supplied smoother and kernel. PACE defaults to Gaussian, following the audited non-dense default. Direct pooled-smoother calls retain their Epanechnikov default; every kernel remains overridable. |
| `methodBwMu`, `methodBwCov` | `SetOptions`: both are `"Default"`, with fixed `userBwMu=0.05*range` and `userBwCov=0.10*range`. GCV is an explicit option, **not the current default**. | Dense has no smoothing bandwidth by default. PACE uses the same fixed 5%/10% defaults on estimated inputs. Numbers, whole-curve `CurveCV`, and the mean smoother's KernelSmoothing selectors pass through to the existing smoothers. Existing `CurveCV` deliberately differs from fdapace's selectors. |
| `kFoldMuCov` | `SetOptions`: 10. `R/GetSmoothedMeanCurve.R`, `GetSmoothedMeanCurve` passes it to mean CV; `R/GetSmoothedCovarSurface.R`, `GetSmoothedCovarSurface` passes it to covariance CV. | Dense uses no folds. Existing `CurveCV` also defaults to ten folds. It requires an RNG when `folds<nobs(fd)` and uses deterministic leave-one-curve-out when they are equal; folds above the number of curves throw. Holding out entire curves respects subject dependence. |
| `useBinnedData` | `SetOptions`: `"AUTO"`; the exact rule is below. `useBinnedCov` is a different option: true except when `n<128` and the mean number of observations is below 3. | Dense does not bin. PACE defaults to `binning=:auto` and uses the audited bin-count decision below. It reuses the existing covariance smoother's weight-preserving pair snapping and aggregation, while leaving the mean and work grid unchanged. Direct covariance-smoother calls retain `binning=nothing`. |
| `methodMuCovEst` | `SetOptions`: Sparse uses `"smooth"`; Dense, DenseWithMV, and `p>>n` use `"cross-sectional"`. | Complete common-grid data use cross-sectional estimates by default. Irregular or masked data use PACE pooled smoothers; `method=:pace` forces this route on regular data too. |
| `methodXi` | `SetOptions`: Dense and `p>>n` use `"IN"`; DenseWithMV uses `"CE"`. Sparse normally uses `"CE"`, but uses `"IN"` if every curve has **more than 20** observations and the largest spacing, including either boundary gap to the pooled range, is at most 6% of that range. | Dense scores use the stored quadrature weights. PACE always uses conditional-expectation scores. There is no additional data-driven switch between these formulas. |
| Measurement error | `SetOptions`: `error=TRUE`, even for Dense. `R/GetCovDense.R`, `GetCovDense` uses a second-difference noise estimate on its cross-sectional route. | Dense stores `noise_variance=nothing` and defaults to `presmooth=nothing`. It assumes negligible measurement error, following Ramsay and Silverman (2005), Chapter 8. Noisy dense curves need explicit presmoothing or PACE. |
| `rho` | `SetOptions`: `methodRho="vanilla"`, `userRho=NULL`. This means no extra regularization; details are below. | `rho` regularization is not implemented. No ridge is silently added. |

The remembered claim that fdapace always uses `maxK=min(20,m-2)` is therefore
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

The package's existing `SmoothedCovariance` shares the middle-half
integrated variance principle. It uses the diagonal of its fitted covariance
surface, inserts the exact interval endpoints by linear interpolation, and
truncates a negative result to **zero**. Its diagonal bandwidth defaults to
the covariance bandwidth and can be supplied separately. These differences
were established in milestone 2 and are not claims of literal `PC_CovE`
equivalence. On fdapace's cross-sectional route, `GetCovDense` instead
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
fitted covariance up to `maxK`**, not only the selected `K` components:
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

## Eigenanalysis, signs, and component selection

`R/GetEigenAnalysisResults.R`, `GetEigenAnalysisResults` computes ordinary
matrix eigenvectors of the covariance surface `G`. It keeps eigenvalues
`d>=0`, including exact zeros, drops strictly negative ones, and truncates
the remaining descending list to `maxK`. It does not record a dropped
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
is measured before the `maxK` cap.

The sign agrees with fdapace when the mean overlap is nonzero. If its
absolute value is at most
`sign_tol*norm(phi)*norm(mu)`, with default `sign_tol=1e-12`, the largest
absolute eigenfunction entry is made positive; ties use the first entry.
This fallback follows the convention used by FunctionalGraphicalModels.jl.
Both numerical tolerances are stored on the fit. They are explicit package
choices that the milestone leaves to the implementation.

Both implementations compute FVE **after** the `maxK` truncation:
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

PACE stores the criterion for every candidate from 1 through `maxK`, then
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
`userSigma2=0.12`, because the package's existing noise estimator differs
from `PC_CovE` as described above.

The fixed-bandwidth run estimates its mean and covariance from the sparse
fixture. It uses `userBwMu=0.30`, `userBwCov=0.40`, `kernel="epan"`,
`useBinnedData="OFF"`, `useBinnedCov=FALSE`, observation weighting,
`methodXi="CE"`, `methodRho="vanilla"`, fixed `K=3`, and `maxK=5`.
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
`error=FALSE`, integration scores, `maxK=5`, no binning, and a fixed K of 3.
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

## Task 2 development validation record

The 2026-10-24 development checks used Julia 1.13.2 and a temporary project at
`/private/tmp/fda-m3-pace-tests`, with this package and the local
KernelSmoothing.jl checkout developed into that environment. Only the affected
test files were run:

| Test file | Passing assertions |
|:--|--:|
| `test/fpca.jl` | 178 |
| `test/pace.jl` | 128 |
| `test/pace_golden.jl` | 68 |

For example, the PACE checks were run with:

```sh
julia --project=/private/tmp/fda-m3-pace-tests -e 'using FunctionalDataAnalysis, Test, Random, LinearAlgebra; include("test/pace.jl")'
```

`Rscript test/reference/fdapace_m3.R` regenerated the golden fixtures with
R 4.5.1 and fdapace 0.6.0. The existing dense CSVs remained byte-for-byte
unchanged. The controller runs the full test suite on Julia 1.10 and the
current release, Aqua, and the docs build. Those controller gates were not
run during this development step.

References: Yao, Müller, and Wang (2005), *Functional Data Analysis for
Sparse Longitudinal Data*, JASA 100, 577–590;
[doi:10.1198/016214504000001745](https://doi.org/10.1198/016214504000001745).
Ramsay and Silverman (2005), *Functional Data Analysis*, second edition,
Chapter 8; [doi:10.1007/b98888](https://doi.org/10.1007/b98888).
