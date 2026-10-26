# Regression

The package fits four Gaussian functional regression models. None of them
selects a smoothing parameter automatically; penalized models with automatic
smoothing are planned separately.

| Response | Covariates | Model | Estimator | Reference |
|:--|:--|:--|:--|:--|
| Scalar | Functional (and scalar) | `y = α + Σ∫β_j X_j + zᵀγ + ε` | [`FPCRegression`](@ref) on FPC scores | fdapace `FLM`, fda.usc `fregre.pc` |
| Functional | Functional (and scalar) | `Y(s) = μ(s) + Σ∫β_j(s,t)X_j(t)dt + zᵀγ(s) + ε(s)` | [`FPCRegression`](@ref) with a functional response | fdapace `FLM` |
| Functional | Functional at the same time (and scalar) | `Y(t) = β₀(t) + Σβ_j(t)X_j(t) + zᵀγ(t) + ε(t)` | [`ConcurrentRegression`](@ref), kernel smoothing | fdapace `FCReg` (formerly `ConcurReg`) |
| Functional | Scalar only | `Y(t) = Σ x_k β_k(t) + ε(t)` | [`FunctionOnScalarRegression`](@ref), fixed-λ basis expansion | fda `fRegress` |

All four estimators follow the StatsAPI interface: `fit`, `predict`, `coef`,
`fitted`, and, where the model defines them, `residuals`, `vcov`, `stderror`,
`nobs`, `dof` and `dof_residual`. Each one stores every tuning choice on the
fit and shows it with `show`.

## Scalar-on-function regression

`FPCRegression` fits a Gaussian scalar response to one or more functional
covariates and optional scalar covariates:

```math
y_i = \alpha + z_i^\top\gamma +
\sum_j\sum_{k=1}^{K_j} b_{jk}\xi_{ijk} + \epsilon_i,
\qquad \beta_j(t)=\sum_{k=1}^{K_j}b_{jk}\phi_{jk}(t).
```

FPCA scores describe centered curves. Dense fits use the stored quadrature
weights, including on nonuniform grids. Basis fits use exact basis inner
products. Sparse fits use the existing PACE conditional expectation scores.
The regression uses an intercept and ordinary least squares, with no regression
smoothing parameter. The intercept is for centered functional covariates and
uncentered scalar covariates. A deliberately uncentered basis FPCA retains its
own score convention; its intercept must be interpreted in that convention.

### Fit and inspect a model

The offline Tecator sample contains three spectra and their fat percentages.
One component leaves one residual degree of freedom. This example illustrates
the interface; use the full dataset for a substantive analysis.

```@example regression
using FunctionalDataAnalysis
import StatsAPI
tecator = load_dataset(:tecator; sample=true)
model = StatsAPI.fit(FPCRegression, tecator.data, tecator.covariates.Fat;
    ncomponents=1)
(StatsAPI.coef(model), model.sigma2, StatsAPI.r2(model))
```

```@example regression
beta = only(coefficient_functions(model))
values = evaluate(beta, tecator.data.grid)
pointwise = only(coefficient_bands(model))
coordinates = band_data(pointwise) # band_plot(pointwise) uses the existing ribbon recipe
predictions = predict(model, tecator.data; interval=:prediction)
(StatsAPI.fitted(model), predictions.lower, predictions.upper)
```

Pass a tuple of functional datasets, or `MultivariateFunctionalData`, for
multiple predictors. Domains and sampling designs may differ. Subjects must
have the same row order; conflicting subject IDs throw. For example,
`fit(FPCRegression, (x1, x2), y; scalar_covariates=Z,
ncomponents=(2, 1))` fits two and one components, respectively.

`coef` orders parameters as intercept, scalar coefficients, then the score
coefficients of each functional covariate. `vcov` and `stderror` use that same
order. `coefficient_functions` returns a tuple of `BasisFunctionalData`, even
for one predictor. This preserves exact basis eigenfunctions when available.
`fitted`, `residuals`, `nobs`, `dof`, `dof_residual` and `r2` follow StatsAPI.
Here `dof` counts regression coefficients, and `sigma2` is RSS divided by
`dof_residual`. Rank deficiency, nonfinite inputs, zero response variation and
nonpositive residual degrees of freedom throw informative errors.
Nullable response and scalar containers are accepted when every entry is finite
and observed; actual missing values throw and no subjects are dropped.
The solve uses Julia's pivoted QR. Iterative proximal or coordinate descent
methods, including ProximalSolvers.jl and CoordinateDescent.jl, would add
iteration and stopping choices to this unpenalized least-squares problem;
no optimization solver or runtime dependency is added.

### FPCA and component choices

| Choice | Default and source |
|:--|:--|
| Regression model | Gaussian OLS with intercept, fdapace 0.6.0 `FLM`/`GetBR2New` |
| FPCA per covariate | A fresh `fit(FPCA, x_j)`, using that covariate's existing defaults |
| Component selection | `selection=:fpca`, keeping each marginal FPCA's selection, as `FLM` does |
| Fresh FPCA component rule | `FVE(0.99)`, fdapace 0.6.0 `SetOptions.R` |
| Scalar covariates | None; explicit vector or matrix when included |
| Supplied fit/options/count | `fpca=nothing`, `fpca_options=(;)`, `ncomponents=nothing`: keep the marginal defaults |
| Prediction intervals | `interval=:none`, matching ordinary R prediction without requested intervals |
| Requested interval coverage | `alpha=0.05`, equivalent to fda.usc's `level=0.95` |
| Interval distribution | Student t with residual degrees of freedom, fda.usc 2.2.0 `predict.fregre.fd`/R `lm` |
| Prediction evaluator | `nothing`: native FPCA scoring contract, with exact evaluation of basis curves for raw fits |
| Coefficient interval grid | Each fitted FPCA work grid; a display choice, not a new inner product |
| Coefficient interval method | `method=:conditional`; `:bootstrap` reproduces fdapace `FLMCI` (`B=999`, its `R`) |

`fpca_options=(; max_k=5)` passes settings to FPCA. For different settings per
predictor, pass a tuple of named tuples. All smoothing, noise and eigenanalysis
choices remain in each marginal FPCA, and are shown with the regression fit.
See [FPCA](fpca.md) for defaults and existing departures from R, including the
estimated measurement-noise floor. Regression does not add a second smoother
or conditional expectation solver.

A supplied `fpca` may be one fit or a tuple/vector with a fit or `nothing` per
covariate. The model copies supplied fits. It uses stored scores only for
observations equal to the fit's stored training data; different observations
are projected with that fit. This also preserves fitted preprocessing and FACE
completion for training observations. Tuning options for supplied fits throw.
`ncomponents` may choose a fixed count or an existing FPCA selector per fresh
fit; for supplied fits it may retain or truncate the available harmonics using
a fixed count or FVE. It cannot recover discarded eigenfunctions.

Optional regression selection uses `selection=AIC()`, `BIC()` or `:loocv`.
Use `fpca_options=(; max_k=...)` to limit fresh candidate bases. AIC and BIC
minimize `n*log(RSS/n) + c*dof`, with `c=2` or `log(n)`. The additive Gaussian
likelihood constant is omitted. LOOCV minimizes mean squared held-out response
error and refits FPCA in every fold, so it costs `n` FPCA fits per covariate
for each candidate count tuple; keep `max_k` small with several covariates.
Supplied fits are rejected for LOOCV
because the procedure that produced them is not recoverable in general.
Selection searches all positive leading-component counts, jointly across
predictors. It records candidate counts, criteria, failures and the winner.
Nonestimable candidates receive infinite criterion with an explicit reason;
if none is estimable, fitting throws. Exact ties choose the first
lexicographic count. Candidate search can be expensive with many predictors.

These are leading-component models. fda.usc `fregre.pc.cv` instead normally
performs stepwise component subset selection. It defaults to `kmax=8` and
`criteria="SIC"`; neither overrides fdapace's FVE default here. Its AIC/SIC
values are the corresponding package criterion divided by `n`. Its CV source
refits FPCA but additionally divides each held-out residual by `n-ck`, where
`ck=n-out$df.residual` for the fold. The package deliberately uses ordinary
LOOCV prediction MSE, without this extra scaling, to measure held-out prediction
error directly. The reference script compares that documented quantity using
real `fregre.pc` fold fits and also saves the scaled R quantity.

### Prediction and uncertainty

Native prediction reuses each stored FPCA fit. PACE accepts grid and irregular
observations within its domain. Dense FPCA requires its complete training grid.
Basis FPCA requires its training basis. New basis curves can be evaluated
exactly on a raw fit's grid. Other conversions require an explicit
`evaluator=PiecewiseLinear()` or `NoInterpolation()`; every curve must cover
the fitted grid. For a basis fit this explicitly requests numerical projection
using its stored work-grid quadrature, which approximates exact basis
integration. No interpolation or basis approximation is chosen implicitly.

`coefficient_bands`, `fitted_bands` and `predict(...; interval=:confidence)`
use the conditional OLS covariance `sigma2*(D'D)^(-1)`, computed through pivoted
QR and triangular solves. A confidence interval describes the conditional
mean response. `interval=:prediction` adds independent Gaussian response noise.
Returned interval objects record their target, alpha, critical value and
uncertainty convention. These are pointwise intervals, with no simultaneous
coverage claim. They exclude uncertainty in estimated means, eigenfunctions,
sparse latent scores, truncation and model selection, and omit discarded modes.
In particular, sparse intervals condition on the PACE score estimates.

fdapace `FLM` itself returns no standard errors. Its separate `FLMCI` defaults
to `level=0.95` and `R=999` subject-bootstrap refits of FPCA and regression,
reporting R `quantile` (type 7) percentile intervals.
`coefficient_bands(model; method=:bootstrap, B=999, rng)` performs the same
refits with the stored FPCA options and component rules; see
[Bootstrap intervals](@ref regression-bootstrap). The conditional default follows
fda.usc's linear-model covariance and Student-t prediction formula.

## Function-on-function regression

Pass a univariate functional response to `FPCRegression`. As in fdapace 0.6.0
`FLM`, the response is reduced to its own FPCA scores `η_ik`. Each response
score is regressed by OLS on the same design of covariate scores (and scalar
covariates), giving a `p × K_Y` coefficient matrix `B`:

```math
\beta_j(t,s) = \sum_{k,l} \phi_{jk}(t)\, B_{jk,l}\, \psi_l(s),
\qquad \hat Y_i(s) = \mu_Y(s) + \sum_l (D B)_{il}\,\psi_l(s).
```

```@example fof
using FunctionalDataAnalysis
import StatsAPI
grid = collect(range(0.0, 1.0; length=21))
u = 2pi .* (1:30) ./ 30
x = GridFunctionalData(cos.(u) .* transpose(sinpi.(grid)) .+
    0.5 .* sin.(2u) .* transpose(sinpi.(2grid)), grid)
y = GridFunctionalData(1 .+ (0.8cos.(u) .- 0.3sin.(2u)) .* transpose(sqrt(2) .* sinpi.(grid)) .+
    0.3 .* sin.(5u) .* transpose(cospi.(2grid)), grid)
fof = StatsAPI.fit(FPCRegression, x, y)
(ncomponents(fof), size(StatsAPI.coef(fof)), StatsAPI.r2(fof))
```

```@example fof
surface = only(coefficient_functions(fof))   # (; grid, response_grid, values)
intervals = only(coefficient_bands(fof))     # pointwise conditional surfaces
curves = StatsAPI.predict(fof, x[1:3])       # GridFunctionalData on the response grid
r2_by_time = pointwise_r2(fof)
size(surface.values), size(curves.values)
```

The response FPCA is controlled like the covariates: `response_fpca` supplies
a fit, `response_fpca_options` tunes a fresh one, and `response_ncomponents`
fixes or truncates its count. Surfaces have the covariate grid `t` in rows and
the response grid `s` in columns, the orientation of fdapace's `betaList`. The
intercept function `μ_Y(s) + B_{0·}ψ(s)'` is
`reconstruct(fof.response_fpca, reshape(StatsAPI.coef(fof)[1, :], 1, :))`.

| Choice | Default and source |
|:--|:--|
| Response reduction | Fresh `fit(FPCA, y)` with its defaults, including `FVE(0.99)`: fdapace `optnsListY=list()` |
| Coefficients | One OLS per response score sharing the design, fdapace `GetBR2New` |
| `r2` | `1 - Σ_k RSS_k / Σ_k TSS_k` on response scores, `GetBR2New`'s `R2` |
| `coef`, `vcov`, `stderror` | `p × K_Y` matrix; `Cov(vec B) = SigmaE ⊗ (D'D)^(-1)`, with `SigmaE = E'E/(n-p)` |
| `sigma2` | `tr(SigmaE)`: integrated residual variance within the retained response components |
| `fitted`, `predict` | Curves `μ_Y + scores*ψ'` on the response work grid or a supplied `grid` |
| `residuals` | `(η - η̂)ψ'`: residuals of the response reconstructed from its retained components |
| Regression selection | Only `selection=:fpca`; AIC, BIC and LOOCV throw, as `FLM` selects by FPCA only |

`pointwise_r2(fof)` returns `1 - Σ_i r_i(s)² / Σ_i (Ỹ_i(s) - mean(s))²` for the
reconstructed responses `Ỹ`. fdapace reports only the scalar `R2`. Where every
retained response eigenfunction vanishes (for example at the endpoints of
sine eigenfunctions), the ratio is `0/0` and the value is `NaN`.

Conditional intervals use the variance
`φ_j(t)'(D'D)^(-1)_{jj}φ_j(t) · ψ(s)'SigmaE ψ(s)` for `β_j(t,s)`. Prediction
intervals for a new response curve add `ψ(s)'SigmaE ψ(s)`. These intervals
condition on both FPCAs. They exclude truncation and measurement error of the
response, which are not part of its retained components.

### [Bootstrap intervals](@id regression-bootstrap)

`coefficient_bands(model; method=:bootstrap, B=999, alpha=0.05, rng=nothing)`
follows fdapace 0.6.0 `FLMCI` for scalar and functional responses. Each
replicate resamples subjects with replacement and refits every FPCA and the
regression with the stored options and component rules. FVE selection may
therefore choose a different number of components in each replicate. The
refitted coefficient functions are evaluated on the requested grids, and R
type-7 quantiles give pointwise percentile intervals. `se` is the bootstrap
standard deviation and `critical_value` is `nothing`.

As in `FLMCI`'s dense branch, a resample is redrawn until every functional
variable keeps its original observed time range, so refitted work grids agree.
After 1000 consecutive rejections the call throws. `FLMCI`'s sparse branch
instead samples from all but `2(p+1)` curves, then appends curves forced to
contain each boundary. The package uses plain subject resampling with the
range check for every design. Bootstrap intervals require fresh FPCA fits and
`selection=:fpca`. Supplied fits throw because their fitting procedure cannot
be repeated. `rng=nothing` uses `Random.default_rng()`. The RNG is copied, so
the caller's state is unchanged, and its initial state is stored as
`settings.rng_state`. A replicate that fails to refit throws with its number;
no draw is dropped. `FLMCI` passes `optnsListY` as the covariate options of its
point estimate, an apparent slip; the package uses the fitted model itself.

## Concurrent regression

[`ConcurrentRegression`](@ref) implements fdapace 0.6.0 `FCReg`. fdapace 0.6.0
no longer contains `ConcurReg`; `FCReg` is its successor with the same
two-dimensional smoothing estimator. Dense and sparse designs use the same
code, and covariates and the response may be observed at different times:

```math
\beta(t) = C_{XX}(t,t)^{-1} C_{XY}(t,t), \qquad
\beta_0(t) = \mu_Y(t) - \sum_j \beta_j(t)\mu_j(t).
```

```@example concurrent
using FunctionalDataAnalysis
import StatsAPI
grid = collect(range(0.0, 1.0; length=11))
u = 2pi .* (1:30) ./ 30
x = GridFunctionalData(cos.(u) .* transpose(1 .+ grid) .+
    0.5 .* sin.(2u) .* transpose(cospi.(grid)) .+ transpose(grid), grid)
y = GridFunctionalData(transpose(1 .+ 0.5grid) .+ x.values .* transpose(1.2 .- 0.8grid) .+
    0.1 .* sin.(7u) .* transpose(cospi.(3grid)), grid)
concurrent = StatsAPI.fit(ConcurrentRegression, x, y;
    mean_bandwidth=0.15, covariance_bandwidth=0.2)
StatsAPI.coef(concurrent)   # intercept row, then one row per covariate
```

Each functional variable is centered by a local linear mean evaluated at its
own sorted unique observation times. Linear interpolation, with constant
extrapolation as in `approxfun(rule=2)`, gives the mean elsewhere. All
within-subject products of centered values are smoothed by the existing
bivariate local linear smoother (KernelSmoothing.jl) with unit weights and the
covariance bandwidth on both axes. Only the diagonal `(t,t)` is evaluated.
With `measurement_error=true`, a variable's own products drop equal-time pairs
(`FCReg`'s `rmDiag`), removing noise variance from `C_XX` and `C_YY`. Scalar
covariates add varying coefficients. Their products with functional variables
are smoothed by the univariate local linear smoother; their own covariances
are sample covariances. `pointwise_r2` returns `FCReg`'s
`R2(t) = C_YX β / C_YY`, clipped to `[0,1]`.

| Choice | Default and source |
|:--|:--|
| Kernel | Gaussian, `FCReg`'s `kern="gauss"` |
| Measurement error | `measurement_error=true`, `FCReg`'s `measurementError=TRUE` |
| Bandwidths | Required: `FCReg` has no defaults for `userBwMu` and `userBwCov` |
| Output grid | `grid=nothing`: the sorted union of observation times, `MvCov`'s default (`FCReg` requires `outGrid`) |
| Mean grid | Sorted unique observation times of each variable, `demean` |
| Local polynomial | Degree 1, unit pair weights, no binning, as in `Lwls1D`/`Lwls2D` |
| One-dimensional diagonal smoothing | Not offered: `FCReg`'s `diag1D` default is `"none"` |
| Intervals | Subject bootstrap, `B=999`, `alpha=0.05`, borrowed from `FLMCI`; `FCReg` has none |

`predict(concurrent, x)` returns `GridFunctionalData` on the fitted grid.
Covariates completely observed on exactly that grid, and basis curves, are
used directly. Other raw curves need an explicit
`evaluator=PiecewiseLinear()` or `NoInterpolation()` and must cover the grid.
Sparse curves usually do not, so `fitted` is available for dense data only.
`coefficient_bands(concurrent; B=999, rng)` resamples subjects and refits the
means and covariances with the stored bandwidths, kernel, grid and noise
setting. It returns pointwise percentile intervals for the intercept and every
coefficient; the bandwidth is not reselected and smoothing bias is ignored.

## Function-on-scalar regression

[`FunctionOnScalarRegression`](@ref) implements fda 6.3.0 `fRegress` with
scalar covariates and a fixed `fdPar` penalty. The response must be
`BasisFunctionalData`; convert raw curves explicitly, for example with
`to_basis` or `smooth_curves`. Each coefficient function uses the response
basis, and the coefficients solve

```math
\bigl(X^\top X \otimes G + \operatorname{blockdiag}(\lambda_k R_k)\bigr)\, b
= \operatorname{vec}(G\, Y^\top X),
```

with exact Gram matrix `G` and penalty matrices `R_k` from the package's
bases.

```@example fos
using FunctionalDataAnalysis
import StatsAPI
basis = BSplineBasis(4, (0.0, 1.0), 7)
u = 2pi .* (1:30) ./ 30
z = cos.(u)
coefs = 1 .+ z .* transpose(sin.(1:7) ./ 2) .+ 0.1 .* sin.(3u) .* transpose(cos.(1:7))
fos = StatsAPI.fit(FunctionOnScalarRegression, z, BasisFunctionalData(coefs, basis); lambda=1e-4)
intercept, slope = coefficient_functions(fos)
bands = coefficient_bands(fos)          # pointwise intervals, intercept first
StatsAPI.predict(fos, [0.5, -0.5])      # BasisFunctionalData
```

| Choice | Default and source |
|:--|:--|
| Intercept | `intercept=true`, the constant `fRegress.formula` adds |
| Coefficient basis | The response basis, `fRegress.formula`'s default for scalar covariates; other bases are not offered |
| Penalty | `lambda=0` (`fdPar`'s default), one value or one per coefficient, fixed |
| Penalty operator | `penalty_order=nothing`: `fdPar`'s default, order `norder-2` for B-splines, harmonic acceleration for Fourier, order 0 for monomials |
| Residual coefficient covariance | `residual_covariance=:residual`: `E'E/(n-q)` from coefficient residuals (`fRegress.stderr` requires it from the user) |
| Interval distribution | Student t with `n-q` degrees of freedom for the estimate; normal for a supplied covariance |
| Interval grid | `max(51, 10*nbasis+1)` equally spaced points, `fRegress.stderr`'s plotting grid |

Standard errors follow `fRegress.stderr`,
`Var(b) = C^(-1) (X'X ⊗ G Σ_c G) C^(-1)`, where `Σ_c` is the covariance of one
response's basis coefficients (`y2cMap*SigmaE*y2cMap'` in fda). A symmetric
positive semidefinite `nbasis × nbasis` matrix may be supplied instead of the
residual estimate; `n-q` ignores the penalty. `predict(fos, z;
interval=:confidence)` gives `fRegress.stderr`'s fitted-curve standard errors;
`:prediction` adds `φ(t)'Σ_c φ(t)` for a new response curve. `fitted` and
`residuals` are exact `BasisFunctionalData` in the response basis.

## Reference source audit and validation

The audited sources are [fdapace 0.6.0](https://cran.r-project.org/src/contrib/fdapace_0.6.0.tar.gz)
(`FLM.R`, `FLMCI.R`, `SetOptions.R`, `GetBR2New`, `FCReg` with its `demean`,
`MvCov`, `uniCov`, `GetCrCovYZ`, `GetRawCrCovFuncFunc` and `Lwls2D` helpers),
[fda.usc 2.2.0](https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz)
(`fregre.pc.R`, `fdata2pc.R`, `predict.fregre.fd.r`, `inprod.fdata.R`,
`ops.fda.usc.R`) and [fda 6.3.0](https://cran.r-project.org/src/contrib/fda_6.3.0.tar.gz)
(`fRegress.fd`, `fRegress.double`, `fRegress.formula`, `fRegress.stderr`,
`fRegressArgCheck`, `fdPar`, `inprod`, `bsplinepen`, `times.fd`). The reference
generators use R 4.5.1.

`FLM` fits marginal FPCA without changing its defaults, concatenates `xiEst`
and scalar columns, then calls `lm`. Sparse predictors therefore use PACE CE
scores through the ordinary FPCA route. Held-out curves use `predict.FPCA`
with the training component count. `fregre.pc` defaults to `l=1:3`, `lambda=0`
and unit observation weights. The package follows fdapace's FVE default because
it adapts the count to the functional predictor. A fixed `ncomponents=3`
provides the corresponding three-leading-component model. With a functional
response, `FLM` fits the response FPCA with `optnsListY` and calls `GetBR2New`,
which runs one `lm` per response score. Its `alpha` subtracts a rectangular
integral of `μ_j β_j`; the package's intercept function is the centered one.

There are inherited discretization differences. fdapace uses rectangular
covariance eigenanalysis and then trapezoidal harmonic normalization; fda.usc
uses Euclidean SVD with uniform-grid normalization. Its inner-product helper
is named `int.simpson2`, but the global `int.method` default is `"TRAPZ"`.
The package consistently uses stored quadrature weights. Dense and sparse
golden fixtures use sine subspaces vanishing at the endpoints, where these
eigensteps agree. Sparse fixtures supply the same mean, covariance and noise
to isolate the regression and CE route from existing smoother differences.
The package's centered intercept is compared with the R score-regression
intercept: `FLM$alpha` instead subtracts a rectangular integral of `mu*beta`.

`FCReg` centers each variable with `GetSmoothedMeanCurve` at its unique times,
then calls `MvCov` with `center=FALSE`. The functional-functional blocks smooth
raw products with `Lwls2D(crosscov=TRUE)` and exclude equal-time pairs of a
variable with itself under `measurementError`. Functional-scalar blocks use
`Lwls1D`. Its coefficient solve is `qr.solve` at each output point. Its
`summaryConReg` helper is empty; no intervals or bandwidth defaults exist.

`fRegress.fd` forms its normal equations with `inprod`, which integrates by
Romberg extrapolation with relative tolerance `1e-4` unless both arguments
are B-spline objects passed with integer derivative orders. With interior
knots this leaves relative errors near `1e-5` in the fitted coefficients. The
package uses exact Gram matrices. `bsplinepen` computes B-spline penalties
exactly for bases with interior knots, but for a B-spline basis without
interior knots it substitutes a monomial basis and returns that basis's
penalty matrix, which does not apply to B-spline coefficients. Penalized
`fRegress` fits on such bases therefore differ from the B-spline penalty used
here. `fRegress.stderr` integrates with a sum over 501 equally spaced points
that gives full weight to both endpoints; the exact inner products here
change its covariance by up to about 8% in the reference fixtures. Fitted
curves are projected onto the response basis on 501 points by `smooth.basis`;
the package's fitted coefficients are exact.

Recreate the small fixtures from the repository root with
`Rscript test/reference/regression_m7.R` (fdapace 0.6.0 and fda.usc 2.2.0) and
`Rscript test/reference/regression_m7b.R` (fdapace 0.6.0 and fda 6.3.0).
`test/regression.jl` compares scalar-response coefficient functions, scalar
coefficients, fitted values, residuals, residual variance, R², held-out
predictions and conditional intervals at relative error below `1e-8`. It also
checks full-refit LOOCV and AIC/BIC against R, and independent analytic
nonuniform-grid, basis and sparse conditional-expectation examples.
`test/regression_functional.jl` compares, at relative error below `1e-8`:

- dense and sparse functional-response `FLM` fits: surfaces, fitted and
  predicted curves, response scores, score coefficients, `R2`, pointwise R²
  (away from the vanishing endpoints) and the conditional standard errors;
- dense (with a scalar covariate, and without measurement error) and sparse
  `FCReg` coefficients, intercepts and `R2(t)`;
- `fRegress` itself for an unpenalized B-spline basis without interior knots,
  where Romberg integration is exact, and otherwise the same normal equations
  recomputed in R with exact `bsplinepen` (or exact Gauss–Legendre) matrices,
  together with the `fRegress.stderr` covariance, pointwise standard errors
  and predictions using exact inner products.

Analytic checks cover noise-free surfaces, bootstrap intervals that collapse
when every resample refits the same model, a replay of the bootstrap draws,
and orthonormal-basis least squares. No R installation is required to use
the package or run its tests.

```@docs
FPCRegression
ConcurrentRegression
FunctionOnScalarRegression
coefficient_functions
coefficient_bands
pointwise_r2
StatsAPI.predict(::FPCRegression, ::Any)
StatsAPI.coef(::FPCRegression)
StatsAPI.vcov(::FPCRegression)
StatsAPI.fitted(::FPCRegression)
StatsAPI.residuals(::FPCRegression)
StatsAPI.dof(::FPCRegression)
StatsAPI.dof_residual(::FPCRegression)
StatsAPI.r2(::FPCRegression)
StatsAPI.stderror(::FPCRegression)
StatsAPI.predict(::ConcurrentRegression, ::Any)
StatsAPI.coef(::ConcurrentRegression)
StatsAPI.nobs(::ConcurrentRegression)
StatsAPI.fitted(::ConcurrentRegression)
StatsAPI.predict(::FunctionOnScalarRegression, ::Any)
StatsAPI.coef(::FunctionOnScalarRegression)
StatsAPI.vcov(::FunctionOnScalarRegression)
StatsAPI.stderror(::FunctionOnScalarRegression)
StatsAPI.nobs(::FunctionOnScalarRegression)
StatsAPI.dof(::FunctionOnScalarRegression)
StatsAPI.dof_residual(::FunctionOnScalarRegression)
StatsAPI.fitted(::FunctionOnScalarRegression)
StatsAPI.residuals(::FunctionOnScalarRegression)
```
