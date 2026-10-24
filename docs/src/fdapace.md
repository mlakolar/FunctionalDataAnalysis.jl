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

## Defaults and data routes

Here `n` is the number of curves and `m` is the number of work-grid points.
The current package implements the dense FPCA route. The PACE defaults remain
for the next task; the existing smoothers require explicit work grids and
bandwidths. Listing fdapace's PACE behavior here does not choose those defaults
for this package.

| Choice | fdapace 0.6.0: executable source | FunctionalDataAnalysis.jl and reason |
|:--|:--|:--|
| Work grid / `nRegGrid` | `R/SetOptions.R`, `SetOptions`: Dense and DenseWithMV use the number of unique pooled times after `signif(t,6)`; Sparse and `p>>n` use 51. `R/FPCA.R`, `FPCA` then creates an equally spaced grid over the observed range. | Dense uses the supplied data grid and its stored weights, without rounding or resampling. It preserves the data representation. PACE grid defaults are deferred. |
| `maxK` | `SetOptions`: `min(nRegGrid-2,n-2)`, additionally capped at 20 **only for `methodMuCovEst="smooth"`**; values below 1 are reset to 1. An explicit `maxK` above `nRegGrid-2` is reduced with a message. | The dense default is `max(1,min(m-2,n-2))`, matching fdapace's dense cross-sectional default with no cap of 20. An explicit cap may use up to `min(m,n-1)`, the sample covariance rank bound; requested components above the cap throw. |
| `methodSelectK` | `SetOptions`: `"FVE"`. `R/SelectK.R`, `SelectK` also accepts `"AIC"`, `"BIC"`, or a positive integer. | `ncomponents=FVE(0.99)` or a positive integer. AIC/BIC selection is reserved for the PACE task. |
| `FVEthreshold` | `SetOptions`: 0.99 for `FPCA`. The standalone `SelectK` function separately defaults to 0.95. | `FVE(0.99)` follows the `FPCA` entry point. |
| Kernel | `SetOptions` initially chooses `"epan"` for Dense and `"gauss"` for every other data type, including DenseWithMV. A later invalid-kernel recovery branch chooses Epanechnikov for DenseWithMV. | The dense sample mean and covariance require no kernel. Optional presmoothing records the supplied smoother and kernel. Existing local-polynomial smoothers default to Epanechnikov; PACE kernel defaults are deferred. |
| `methodBwMu`, `methodBwCov` | `SetOptions`: both are `"Default"`, with fixed `userBwMu=0.05*range` and `userBwCov=0.10*range`. GCV is an explicit option, **not the current default**. | Dense has no smoothing bandwidth by default. Existing pooled smoothers require an explicit positive bandwidth or selector; PACE defaults are deferred. This avoids pretending that existing `CurveCV` reproduces fdapace's selectors. |
| `kFoldMuCov` | `SetOptions`: 10. `R/GetSmoothedMeanCurve.R`, `GetSmoothedMeanCurve` passes it to mean CV; `R/GetSmoothedCovarSurface.R`, `GetSmoothedCovarSurface` passes it to covariance CV. | Dense uses no folds. Existing `CurveCV` also defaults to ten folds. It requires an RNG when `folds<nobs(fd)` and uses deterministic leave-one-curve-out when they are equal; folds above the number of curves throw. Holding out entire curves respects subject dependence. |
| `useBinnedData` | `SetOptions`: `"AUTO"`; the exact rule is below. `useBinnedCov` is a different option: true except when `n<128` and the mean number of observations is below 3. | Dense does not bin. The existing covariance smoother defaults to `binning=nothing`; explicit binning records its grid and preserves pair weights. No automatic binning rule is imported. |
| `methodMuCovEst` | `SetOptions`: Sparse uses `"smooth"`; Dense, DenseWithMV, and `p>>n` use `"cross-sectional"`. | Regular, completely observed grid data use the cross-sectional sample mean and covariance. Masked or irregular data require the PACE task. |
| `methodXi` | `SetOptions`: Dense and `p>>n` use `"IN"`; DenseWithMV uses `"CE"`. Sparse normally uses `"CE"`, but uses `"IN"` if every curve has **more than 20** observations and the largest spacing, including either boundary gap to the pooled range, is at most 6% of that range. | Dense scores use the stored quadrature weights. PACE will use conditional-expectation scores; there is no data-driven switch between scoring formulas in the current dense fit. |
| Measurement error | `SetOptions`: `error=TRUE`, even for Dense. `R/GetCovDense.R`, `GetCovDense` uses a second-difference noise estimate on its cross-sectional route. | Dense stores `noise_variance=nothing` and defaults to `presmooth=nothing`. It assumes negligible measurement error, following Ramsay and Silverman (2005), Chapter 8. Noisy dense curves need explicit presmoothing or the later PACE route. |
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

References: Yao, Müller, and Wang (2005), *Functional Data Analysis for
Sparse Longitudinal Data*, JASA 100, 577–590;
[doi:10.1198/016214504000001745](https://doi.org/10.1198/016214504000001745).
Ramsay and Silverman (2005), *Functional Data Analysis*, second edition,
Chapter 8; [doi:10.1007/b98888](https://doi.org/10.1007/b98888).
