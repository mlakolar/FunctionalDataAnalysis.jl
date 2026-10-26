# Regression

The package fits Gaussian functional regression models by two routes. FPC
regression has no smoothing parameter: β is expanded in leading FPCs.
[`PenalizedFunctionalRegression`](@ref), penalized
[`FunctionOnScalarRegression`](@ref) and
[`PenalizedFunctionOnFunctionRegression`](@ref) expand β in a spline basis and
select their smoothing parameters by REML (default) or GCV through
[`PenalizedLeastSquares`](@ref). Concurrent regression records the bandwidths it
uses on the fit.

| Response | Covariates | Model | Estimator | Reference |
|:--|:--|:--|:--|:--|
| Scalar | Functional (and scalar) | `y = α + Σ∫β_j X_j + zᵀγ + ε` | [`FPCRegression`](@ref) on FPC scores | fdapace `FLM`, fda.usc `fregre.pc` |
| Scalar | Functional (and scalar) | same | [`PenalizedFunctionalRegression`](@ref), penalized splines with REML | refund `pfr` with `lf()` |
| Functional | Functional (and scalar) | `Y(s) = μ(s) + Σ∫β_j(s,t)X_j(t)dt + zᵀγ(s) + ε(s)` | [`FPCRegression`](@ref) with a functional response | fdapace `FLM` |
| Functional | Functional (and scalar, numeric or factor) | same | [`PenalizedFunctionOnFunctionRegression`](@ref), tensor-product penalized splines with REML | refund `pffr` with `ff()` |
| Functional | Functional at the same time (and scalar) | `Y(t) = β₀(t) + Σβ_j(t)X_j(t) + zᵀγ(t) + ε(t)` | [`ConcurrentRegression`](@ref), kernel smoothing | fdapace `FCReg` (formerly `ConcurReg`) |
| Functional | Scalar only (numeric or factor) | `Y(t) = Σ x_k β_k(t) + ε(t)` | [`FunctionOnScalarRegression`](@ref), penalized splines with REML (fixed-λ basis expansion when λ is supplied) | refund `pffr`, `fosr`; fda `fRegress` |

All estimators follow the StatsAPI interface: `fit`, `predict`, `coef`,
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
| FPCA per covariate and functional response | A fresh `fit(FPCA, x_j)` with that dataset's defaults, except `method=:face` on complete grids (a departure, below) |
| Component selection | `selection=:fpca`, keeping each marginal FPCA's selection, as `FLM` does |
| Regression-selection range | Leading counts from 1 to each FPCA's own selected count, `selection.cap` (a departure, below) |
| Fresh FPCA component rule | `FVE(0.99)`, fdapace 0.6.0 `SetOptions.R` |
| Scalar covariates | None; explicit vector or matrix when included |
| Supplied fit/options/count | `fpca=nothing`, `fpca_options=(;)`, `ncomponents=nothing`: keep the marginal defaults |
| Prediction intervals | `interval=:none`, matching ordinary R prediction without requested intervals |
| Requested interval coverage | `alpha=0.05`, equivalent to fda.usc's `level=0.95` |
| Interval distribution | Student t with residual degrees of freedom, fda.usc 2.2.0 `predict.fregre.fd`/R `lm` |
| Prediction evaluator | `nothing`: native FPCA scoring contract, with exact evaluation of basis curves for raw fits |
| Coefficient interval grid | Each fitted FPCA work grid; a display choice, not a new inner product |
| Coefficient interval method | `method=:conditional`; `:bootstrap` reproduces fdapace `FLMCI` (`B=999`, its `R`) |

### [FPC defaults that depart from fdapace](@id regression-fpc-departures)

fdapace 0.6.0 `FLM` fits each dense covariate with its unsmoothed dense FPCA
(sample covariance). `FPCRegression` instead fits a complete common-grid
covariate (`GridFunctionalData` without missing values) with FACE,
`method=:face`, which smooths the covariance before FVE selection. In the
[regression study](@ref regression-method-choice), noisy curves on a 101-point grid
made the dense route keep 40–63 noise-dominated components under
`FVE(0.99)`; the median relative error of `β̂` was 3.8–11.6 for leading and
smooth coefficients, against 0.15–0.71 for penalized regression.

| Covariate or response | Fresh FPCA route | `settings.fpca_method_source` |
|:--|:--|:--|
| Complete grid, more than `35 + p` points | `method=:face`, 35 knot intervals | `:regression_default` |
| Complete grid, `p + 2` to `35 + p` points | `method=:face`, `length(grid) - p - 1` knot intervals | `:short_grid_knots` |
| Complete grid, fewer than `p + 2` points | throws; name a `method` | |
| Grid with missing values, irregular | PACE (`:auto`) | `:fpca_default` |
| Basis | basis route | `:fpca_default` |
| Any, with `method` in `fpca_options` | as named | `:fpca_options` |
| Supplied `fpca` | the supplied fit | `:supplied` |

FACE's default basis has 35 equal knot intervals of degree `p=3` splines and
needs more than `35 + p` grid points. On a shorter complete grid the
regression default uses `knots = length(grid) - p - 1`, the most intervals
the grid supports, so every complete grid is smoothed. `face` settings in
`fpca_options` win: `(; face=(; knots=10))` fixes the intervals and
`(; face=(; p=2))` changes the degree used in the rule. If the default FACE
fit fails (for example, equal knot intervals on a strongly clustered grid make
the spline design rank deficient), the error names the default and the
overrides. `fpca_options=(; method=:dense)` restores fdapace's route. Dense- or
PACE-only options (such as `presmooth` or bandwidths) without a `method` throw
rather than switching route silently. `settings.fpca_method` records each
covariate's route and `settings.fpca_knots` its FACE knot intervals
(`nothing` for other routes). `fit(FPCA, x)` itself is unchanged.

The FPCA of a functional response follows the same rule, for the same reason
(noise components kept by `FVE(0.99)` on the dense route), through
`response_fpca_options`. It is recorded in `settings.response_fpca_method`,
`settings.response_fpca_method_source` and `settings.response_fpca_knots`
(a one-entry vector). `response_fpca_options=(; method=:dense)` restores
fdapace's response route.

Regression selection (`AIC()`, `BIC()`, `:loocv`) searches leading counts
from 1 up to the count each covariate's FPCA selects with its own selector:
`FVE(0.99)` by default, `fpca_options.ncomponents` when given, or a supplied
fit's retained count. The cap is stored in `selection.cap` and shown. In the
study, searching every retained component (up to `n-2` on the dense route)
let BIC choose about 70 components at `n=100`, with median relative `β̂`
errors of 44.5–106 for leading and smooth coefficients.
`fpca_options=(; ncomponents=FVE(1))` restores a search over all retained
components.

### FPCA options

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
Candidates are capped by each FPCA's own selection, as described above; set
`fpca_options=(; ncomponents=...)` or `max_k` to change the cap. AIC and BIC
minimize `n*log(RSS/n) + c*dof`, with `c=2` or `log(n)`. The additive Gaussian
likelihood constant is omitted. LOOCV minimizes mean squared held-out response
error and refits FPCA in every fold, with the route of the full-data fit, so
it costs `n` FPCA fits per covariate for each candidate count tuple; keep the
cap small with several covariates.
Supplied fits are rejected for LOOCV
because the procedure that produced them is not recoverable in general.
Selection searches every leading-component count tuple up to the caps, jointly
across predictors. It records candidate counts, criteria, failures, the caps
and the winner.
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
| Covariate FPCA | As for a scalar response, including the FACE default on complete grids |
| Response reduction | Fresh `fit(FPCA, y)` with its defaults, including `FVE(0.99)` (fdapace `optnsListY=list()`), except FACE on complete grids ([departures](@ref regression-fpc-departures)) |
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

[`FunctionOnScalarRegression`](@ref) fits

```math
Y_i(t) = \sum_{k=1}^{q} x_{ik}\,\beta_k(t) + \epsilon_i(t),
\qquad \beta_k(t) = \phi(t)^\top b_k,
```

with `x_i = [1, z_i]`. Scalar covariates are a vector or matrix, or a
Tables.jl table whose non-numeric columns are factors. Factor levels are
`sort(unique(column))` and enter through treatment contrasts against the
first level, as in R's `model.matrix`; the design column names are stored in
`settings.design_columns`. Grid (also with missing values), irregular and
basis responses are accepted.

### Penalized fit with REML (default)

Without `lambda`, every coefficient function is a penalized spline and the
smoothing parameters are chosen by REML on the
[penalized least-squares solver](penalized.md). Grid and irregular responses
are fitted at their observed points; basis responses in the L2 norm.

```@example fos
using FunctionalDataAnalysis
import StatsAPI
grid = collect(range(0.0, 1.0; length=31))
u = 1:40
group = ["a", "b", "c"][mod.(7u, 3) .+ 1]
covariates = (; z=sin.(2.3u), group)
values = transpose(1 .+ sin.(2pi .* grid)) .+ covariates.z .* transpose(grid .^ 2) .+
    (group .== "b") .* transpose(cospi.(grid)) .+
    0.3 .* cos.(1.7u) .* transpose(sinpi.(grid)) .+ 0.1 .* sin.(u .* transpose(1:31))
y = GridFunctionalData(values, grid)
fos = StatsAPI.fit(FunctionOnScalarRegression, covariates, y)
(fos.lambda, fos.edf, fos.settings.design_columns)
```

```@example fos
bands = coefficient_bands(fos)       # intercept, z, groupb, groupc
StatsAPI.predict(fos, (; z=[0.5], group=["c"]); interval=:confidence).se
```

The model is the scalar-term model of refund's `pffr`: working-independent
Gaussian errors, a penalized spline for each coefficient function and one
smoothing parameter each. `pffr` is the default reference because it is the
refund fit that selects smoothing parameters by REML (fosr does not; see the
audit below). The coefficient basis follows fosr instead of `pffr`: 15 cubic
B-splines with equally spaced knots on the response domain and the integrated
squared second derivative penalty. This is the package's B-spline basis and
roughness penalty, the same space and penalty as mgcv's `bs` basis with
`m=c(3,2)`. `basis=:pffr` uses `pffr`'s own defaults instead: mgcv P-spline
bases ([`PSplineBasis`](@ref)) with 20 cubic functions for the intercept and 5
for every other coefficient, first-order difference penalties, and knots on
the range of the observation times widened by 0.1%, as mgcv places them (for
a grid response, the range of the whole grid, which is the same unless an end
point is missing for every curve). Any basis, or one basis per coefficient
(sharing one domain), can be supplied with `basis`; for a grid response its
domain must contain the whole grid, since fitted curves and default bands are
returned there, and `penalty_order` changes the penalty; for a `PSplineBasis` it is
the order of the coefficient differences. Every column of `x_i` gets its own
coefficient function: a factor contributes one varying coefficient per
non-reference level.

```@example fos
pffr_style = StatsAPI.fit(FunctionOnScalarRegression, covariates, y; basis=:pffr)
(nbasis.(pffr_style.bases), pffr_style.lambda)
```

| Choice | Default and source |
|:--|:--|
| Intercept | `intercept=true`: `pffr`'s functional intercept, fosr's intercept column |
| Factors | Treatment contrasts against the first sorted level, R `model.matrix` (fosr's formula interface) |
| Smoothing parameters | `lambda=nothing`: one per coefficient chosen by `criterion=:REML`, `pffr`'s `method="REML"` with a smoothing parameter per term |
| Shared smoothing parameter | `shared_lambda=false`; `true` uses one λ for every coefficient, fosr's `multi.sp=FALSE` |
| Criterion | `criterion=:REML`; `:GCV` is mgcv's `GCV.Cp`, which every fosr fit actually uses |
| Coefficient basis (grid and irregular responses) | `basis=nothing`: 15 cubic B-splines on the response domain, fosr's `nbasis=15`, `norder=4`; `basis=:pffr`: `pffr`'s `bs.int = list(bs="ps", k=20, m=c(2,1))` and `bs.yindex = list(bs="ps", k=5, m=c(2,1))` |
| Coefficient basis (basis responses) | The response basis, fosr's `fdobj` route; supplying `basis` throws |
| Penalty | `penalty_order=nothing`: derivative order `norder-2` for B-splines (fosr `pen.order=2`, mgcv `bs` `m=c(3,2)`), the basis's `difference_order` for P-splines (mgcv `ps` `m[2]`), harmonic acceleration for Fourier, order 0 for monomials |
| Observation weights | Unit weights at the observed points, as in `pffr` and fosr (a likelihood for pointwise observations, not a quadrature) |
| Basis-response loss | L2 through the exact Gram matrix `G`: pseudo-observations `G^(1/2)c_i`, fosr's `respmat = C J^(1/2)` |
| Coefficient covariance | `covariance=:cluster`, refund 0.1.40 `pffr`'s default `sandwich="cluster"` (not the posterior covariance; see below) |
| Interval distribution | Normal |
| Interval grid | The response grid for grid responses (fosr's `argvals`); otherwise `max(51, 10*nbasis+1)` equally spaced points |

The smoothing parameters, their statuses, the criterion value, the error
variance `sigma2` (mgcv's scale estimate, the residual sum of squares over
`N - edf`), the effective degrees of freedom `edf` of each coefficient
function, the solver settings and all choices above are stored on the fit and
shown by `show`. `dof` is the total EDF and `dof_residual` the number of
observations minus that EDF. `fitted` returns curves on the full response grid
(grid), at the observation times (irregular) or in the response basis;
`residuals` keeps the observed points of the response, so missing grid values
stay missing. `predict(fos, z)` returns `BasisFunctionalData` in the
coefficient basis when all coefficients share one basis, and curves on a grid
(`GridFunctionalData`) when they do not, as with `basis=:pffr`.

### Residual correlation

Errors of one curve are usually correlated. The point estimate assumes
working independence, as `pffr` and fosr's OLS fit do; `covariance` chooses
how the uncertainty accounts for the correlation:

| `covariance` | Covariance of the coefficients | Reference |
|:--|:--|:--|
| `:cluster` (default) | `N/(N-1) A⁻¹(Σ_i X_iᵀr_i r_iᵀX_i)A⁻¹ + Vp - Ve`, residuals clustered by curve | refund 0.1.40 `pffr(sandwich="cluster")` |
| `:bayesian` | `Vp = A⁻¹σ²`, the posterior covariance under independent errors | `pffr(sandwich="none")`, mgcv `Vp` |
| `:frequentist` | `Ve = A⁻¹XᵀXA⁻¹σ²` under independent errors | mgcv `Ve` |
| `:pooled` | `A⁻¹(Σ_i X_iᵀΣ̂X_i)A⁻¹`, with `Σ̂` the residual covariance | fosr's OLS covariance `GinvXT (I ⊗ Σ̂) GinvXTᵀ` |

Here `A = XᵀX + S_λ` and `N` is the number of curves. `:cluster` needs no
common grid and is robust to any within-curve correlation.

The default departs from the posterior covariance `Vp` (mgcv's default, and
the covariance originally specified for these bands). refund 0.1.40 made the
cluster sandwich `pffr`'s default because `Vp` treats the points of one curve
as independent observations: with correlated residual curves, as functional
data nearly always have, `Vp` understates the uncertainty of the coefficient
functions and the bands undercover. `covariance=:bayesian` gives `Vp`. `:pooled` assumes
one residual covariance for all curves and needs a complete common grid or a
basis response. The field `residual_covariance` holds `Σ̂`, the covariance of
the residual curves divided by `n` (fosr's `(n-1)/n` times `cov`), on the
response grid, or of the residual coefficients for basis responses. For
irregular and incomplete grids it is a `0×0` matrix: estimating it requires
smoothing, for example `fit(FPCA, residuals(fos))`.

Refitting with the residual structure, as fosr's `method="GLS"` (a whitening
refit by the estimated residual covariance) and `method="mix"` (residual
functional principal components as random effects) do, is not offered. refund
0.1.40 removed `pffr`'s GLS variant (`pffr_gls`) because it gave poorly
calibrated inference and recommends the cluster sandwich instead; fosr's GLS
route also inherits the helper defects listed in the audit. Intervals condition
on the selected smoothing parameters. `interval=:prediction` adds the error
variance `sigma2`.

### Fixed smoothing parameters

Supplying `lambda` keeps it fixed. A basis response with supplied `lambda`
follows fda 6.3.0 `fRegress` with scalar covariates and a fixed `fdPar`
penalty (`method=:fRegress`, chosen automatically). Each coefficient function
uses the response basis, and the coefficients solve

```math
\bigl(X^\top X \otimes G + \operatorname{blockdiag}(\lambda_k R_k)\bigr)\, b
= \operatorname{vec}(G\, Y^\top X),
```

with exact Gram matrix `G` and penalty matrices `R_k` from the package's
bases. These are the normal equations of the penalized basis-response fit, so
the coefficients agree with `method=:penalized` at the same λ; the covariance
follows `fRegress.stderr` instead.

```@example fosfixed
using FunctionalDataAnalysis
import StatsAPI
basis = BSplineBasis(4, (0.0, 1.0), 7)
u = 2pi .* (1:30) ./ 30
z = cos.(u)
coefs = 1 .+ z .* transpose(sin.(1:7) ./ 2) .+ 0.1 .* sin.(3u) .* transpose(cos.(1:7))
fixed = StatsAPI.fit(FunctionOnScalarRegression, z, BasisFunctionalData(coefs, basis); lambda=1e-4)
intercept, slope = coefficient_functions(fixed)
bands = coefficient_bands(fixed)        # pointwise intervals, intercept first
StatsAPI.predict(fixed, [0.5, -0.5])    # BasisFunctionalData
```

| Choice | Default and source |
|:--|:--|
| Penalty | `lambda` supplied, one value or one per coefficient; `fdPar`'s default is `lambda=0` |
| Penalty operator | `penalty_order=nothing`: `fdPar`'s default, order `norder-2` for B-splines, harmonic acceleration for Fourier, order 0 for monomials |
| Residual coefficient covariance | `residual_covariance=:residual`: `E'E/(n-q)` from coefficient residuals (`fRegress.stderr` requires it from the user) |
| Interval distribution | Student t with `n-q` degrees of freedom for the estimate; normal for a supplied covariance |
| Interval grid | `max(51, 10*nbasis+1)` equally spaced points, `fRegress.stderr`'s plotting grid |

Standard errors follow `fRegress.stderr`,
`Var(b) = C^(-1) (X'X ⊗ G Σ_c G) C^(-1)`, where `Σ_c` is the covariance of one
response's basis coefficients (`y2cMap*SigmaE*y2cMap'` in fda). A symmetric
positive semidefinite `nbasis × nbasis` matrix may be supplied instead of the
residual estimate; `n-q` ignores the penalty. `predict(fixed, z;
interval=:confidence)` gives `fRegress.stderr`'s fitted-curve standard errors;
`:prediction` adds `φ(t)'Σ_c φ(t)` for a new response curve. `fitted` and
`residuals` are exact `BasisFunctionalData` in the response basis. Grid and
irregular responses with a supplied `lambda` use the penalized fit at that
fixed λ, with its covariance options.

## Reference source audit and validation

The audited sources are [fdapace 0.6.0](https://cran.r-project.org/src/contrib/fdapace_0.6.0.tar.gz)
(`FLM.R`, `FLMCI.R`, `SetOptions.R`, `GetBR2New`, `FCReg` with its `demean`,
`MvCov`, `uniCov`, `GetCrCovYZ`, `GetRawCrCovFuncFunc` and `Lwls2D` helpers),
[fda.usc 2.2.0](https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz)
(`fregre.pc.R`, `fdata2pc.R`, `predict.fregre.fd.r`, `inprod.fdata.R`,
`ops.fda.usc.R`) and [fda 6.3.0](https://cran.r-project.org/src/contrib/fda_6.3.0.tar.gz)
(`fRegress.fd`, `fRegress.double`, `fRegress.formula`, `fRegress.stderr`,
`fRegressArgCheck`, `fdPar`, `inprod`, `bsplinepen`, `times.fd`), together
with [refund 0.1.40](https://cran.r-project.org/src/contrib/refund_0.1.40.tar.gz)
(`pffr`, `pffr_prepare`, `apply_sandwich_correction`, `gam_sandwich_cluster`,
`assemble_cluster_sandwich`, `build_cluster_id`, `pffr_gls`, `fosr`, `amc`,
`lofocv`, `lw.test`) and mgcv 1.9-3 (`smooth.construct.bs.smooth.spec`,
`smooth.construct.ps.smooth.spec` and `gam` with `paraPen`). The reference generators use R 4.5.1.

`FLM` fits marginal FPCA without changing its defaults, concatenates `xiEst`
and scalar columns, then calls `lm`. Its dense predictors therefore use the
unsmoothed dense FPCA, which the package replaces by FACE on complete grids,
for covariates and a functional response
([departures](@ref regression-fpc-departures)); the dense golden tests pass
`method=:dense` explicitly. Sparse predictors use PACE CE
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

`pffr` with scalar terms builds `s(yindex)` for the intercept (a centered smooth
plus the parametric constant, which together span the uncentered spline
space with the same penalty) and `s(yindex, by=x)` for each numeric covariate,
then calls `gam` with `method="REML"`. A factor `g` becomes `s(yindex, by=g)`:
one smooth per level, each under mgcv's centering constraint and without a
factor main effect, so level differences that are constant in `t` are not
modelled. The package codes factors by treatment contrasts instead, and the
fixture passes the dummy columns to `pffr` as numeric covariates. mgcv scales
each penalty by its `S.scale` before applying `sp`, so `sp/S.scale` is the
smoothing parameter of the exact penalty used here. `pffr`'s default bases
are mgcv `ps` smooths with knots built from the range of `yind`; the
package's [`PSplineBasis`](@ref) reproduces mgcv's knot placement exactly.
refund 0.1.40 changed
`pffr`'s default to `sandwich="cluster"`:
`N/(N-1) Vp M Vp + Vp - Ve` with `M` the outer products of the curve-summed
score contributions `X_iᵀr_i/σ²`; it removed `pffr_gls`.

fosr's default `method="OLS"` with `lambda=NULL` chooses one shared λ by
leave-one-function-out CV (`lofocv`, searched by `optimize` with `tol=0.01` up
to a GCV-based `maxlam`). Its fits run through the helper `amc`, which has two
defects that make fosr unsuitable as the default reference: it passes only the
first penalty to `gam` (`paraPen` receives `S.[[1]]`), so with fosr's
`multi.sp=TRUE` (the default for `method="GLS"` and `"mix"`) only the
intercept function is penalized; and it never forwards `gam.method`, so every
λ that fosr estimates with `gam` uses mgcv's default `GCV.Cp`, whatever
`gam.method` says. With `multi.sp=FALSE` and `max.iter=0`, fosr returns the
penalized OLS fit with a shared λ chosen by GCV and the covariance
`GinvXT (I ⊗ Σ̂) GinvXTᵀ`, where `Σ̂ = (n-1)/n cov(residual curves)`; this is the
package's `shared_lambda=true, criterion=:GCV, covariance=:pooled` fit. For
basis responses fosr uses `J^(1/2)` (from `svd` of the Gram matrix) as both
design and response transform, which is the package's L2 loss. Its `lofocv`
weights each held-out curve's error by a trapezoidal weight of the time grid
indexed by the curve number, not by time point.

Recreate the small fixtures from the repository root with
`Rscript test/reference/regression_m7.R` (fdapace 0.6.0 and fda.usc 2.2.0) and
`Rscript test/reference/regression_m7b.R` (fdapace 0.6.0 and fda 6.3.0) and
`Rscript test/reference/regression_m7e.R` (refund 0.1.40, mgcv 1.9-3 and
fda 6.3.0).
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

`test/regression_fosr.jl` compares penalized function-on-scalar fits of 40
dense curves with a numeric covariate and a three-level factor at relative
error below `1e-8`, at the smoothing parameters R selected:

- `pffr` (REML, one λ per coefficient, mgcv `bs` basis with the package's
  knots): coefficient functions, their standard errors under
  `covariance=:bayesian` (`Vp`), `:frequentist` (`Ve`) and `:cluster`, fitted
  curves, predictions, EDF per coefficient, total EDF and the scale;
- `pffr` with its default `ps` bases (`basis=:pffr`): the same quantities, with
  standard errors under `covariance=:bayesian` and `:cluster`;
- fosr with a grid response (GCV, shared λ) and with a cubic B-spline basis
  response: coefficient functions, the standard errors under
  `covariance=:pooled` (fosr's residual-covariance sandwich), fitted
  curves or coefficients, total EDF and the GCV score; and fosr's OLS fit at a
  fixed λ.

Separately, the criterion the package minimizes is no larger at its own
selection than at R's (equal up to rounding of the score when mgcv reaches
the same optimum). mgcv's REML score differs from the package's by a constant
that depends on the parametrization of the intercept, so REML scores are not
compared directly. Irregular and incomplete-grid fits are checked against the
same observations on a grid.

Analytic checks cover noise-free surfaces, bootstrap intervals that collapse
when every resample refits the same model, a replay of the bootstrap draws,
and orthonormal-basis least squares. No R installation is required to use
the package or run its tests.

## Penalized scalar-on-function regression

`PenalizedFunctionalRegression` fits the same Gaussian model with each
coefficient function expanded in a spline basis and penalized for roughness,
as refund's `pfr(y ~ z + lf(X1) + lf(X2), method="REML")`
(Goldsmith, Bobb, Crainiceanu, Caffo and Reich 2011):

```math
y_i = \alpha + z_i^\top\gamma + \sum_j \sum_t L_{jt} X_{ij}(t)\,
B_j(t)^\top b_j + \epsilon_i,
\qquad \text{penalty } \sum_j \lambda_j b_j^\top S_j b_j .
```

`L_jt` are integration weights on the covariate grid, `B_j` a basis and `S_j`
its penalty. Each functional covariate has its own smoothing parameter,
chosen by REML through [`PenalizedLeastSquares`](@ref) (the solver of the
[Penalized fitting](penalized.md) page). The intercept and scalar covariates
are unpenalized.

```@example regression
using Random
rng = Xoshiro(7)
grid = collect(range(0, 1; length=40))
shapes = hcat(ones(40), grid, sinpi.(grid), cospi.(grid), sinpi.(2 .* grid), cospi.(2 .* grid))
X = randn(rng, 80, 6) * transpose(shapes) .+ 0.05 .* randn(rng, 80, 40)
z = randn(rng, 80)
y = 1 .+ 0.5 .* z .+ X * (quadrature_weights(grid) .* sinpi.(2 .* grid)) .+ 0.1 .* randn(rng, 80)
x = GridFunctionalData(X, grid)
penalized = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z)
(penalized.lambda, penalized.edf_terms, penalized.sigma2, StatsAPI.coef(penalized)[1:2])
```

```@example regression
bands = only(coefficient_bands(penalized))   # Vc, as refund's coef.pfr
(bands.settings.covariance, extrema(bands.estimate - sinpi.(2 .* bands.grid)))
```

### Defaults and their sources

| Choice | Default and source |
|:--|:--|
| Basis | `ThinPlateRegressionBasis(grid, 10)`: refund `lf()` passes its `...` to `mgcv::s`, whose default is `bs="tp"` with `k=10` |
| Penalty | `∫β''²` (thin plate order `m=2`), rescaled as mgcv `smoothCon(scale.penalty=TRUE)`; `penalty_scale` is stored |
| Criterion | `criterion=:REML`, refund `pfr(method="REML")`; `:GCV` (Gaussian) or `:UBRE` (binomial, Poisson) is mgcv `"GCV.Cp"` |
| Family | `family=:gaussian`, `link=nothing` (canonical), refund `pfr`'s `gam` default; see [Exponential-family responses](@ref regression-glm) |
| Integration | `integration=:quadrature` (trapezoidal, refund's `"trapezoidal"`); **departs** from refund's `"simpson"`, see below |
| Presmoothing | `presmooth=nothing`, refund `presmooth=NULL` |
| B-spline presmoother | `presmooth_options=(; nbasis=10)`, refund `create.prep.func` |
| Coefficient bands | Normal pointwise, `covariance=nothing` = `Vc` when available, else `Vp` (refund `coef.pfr(useVc=TRUE)`); `alpha=0.05` |
| Band grid | refund `coef.pfr`'s `ndefault`: with fewer than five distinct spacings, equally spaced points at the smallest spacing, their number halved (rounding half to even) until at most 101; else 101 equally spaced points |
| Prediction intervals | `interval=:none`; requested intervals use `Vp` (mgcv `predict.gam`) and normal quantiles |

A [`BSplineBasis`](@ref) or any other basis covering the grid can be supplied
with `basis=...`, penalized by `penalty(basis, penalty_order)` (default
order 2). Pass a tuple of options for several covariates, for example
`k=(10, 6)` or `presmooth=(nothing, :fpca)`. Fixed `lambda` (one value per
covariate, in units of the rescaled penalties like mgcv's `sp`) skips
selection. Every choice is stored in `settings` and the per-term `terms`
records (grid, weights, basis, penalty, scale, presmoother and processed
training curves), and is shown by `show`.

**Integration weights.** refund's `"simpson"` weights are
`(t_J - t_1)/J/3 * [1, 4, 2, 4, …, 2, 1]`. They divide by the number of points
`J` instead of the `J-1` intervals and are not Simpson's rule when `J-1` is
odd, so every `β` is inflated by about `J/(J-1)` (5% on 21 points). The
package therefore integrates with the grid's stored trapezoidal weights, the
same weights it uses for every other inner product and identical to refund's
`integration="trapezoidal"`. `integration=:refund_simpson` and `:riemann`
reproduce refund's two other formulas exactly, and a weight vector may be
supplied.

### Sparse, noisy and basis covariates

Grid covariates with missing values and irregular covariates need a
presmoother, as in refund, where `lf(presmooth=)` reconstructs curves before
fitting. `:bspline` fits cubic B-splines per curve by least squares, with
interior knots at quantiles of the grid as R's
`splines::bs(df=nbasis, intercept=TRUE)`. `:interpolate` interpolates
observed values linearly and holds them constant beyond the first and last
observation (R `approx(rule=2)`). `:fpca` fits the package's FPCA (PACE for
sparse data, with `presmooth_options` passed to `fit(FPCA, ...)`) and uses its
reconstruction `mean + scores × eigenfunctions` on the FPCA work grid or a
supplied `grid`; refund uses its own `fpca.sc`, `fpca.face` or `fpca.ssvd`
here, so this option is not numerically comparable to refund. New subjects are
presmoothed with the stored presmoother: projected onto the training FPCA, or
smoothed per curve. Basis covariates are evaluated exactly on a required
`grid`. Only the presmoothed curves enter the regression: as in refund,
presmoothing and FPCA uncertainty are not propagated to the covariances or
bands.

### Uncertainty

`vcov(fit; type=:bayesian)` is the posterior covariance `(XᵀX + S_λ)⁻¹σ²`
(mgcv `Vp`), `:frequentist` is mgcv `Ve`, and `:corrected` is mgcv's `Vc`.
`Vc` adds a first-order term `J V_ρ Jᵀ`, with `J = ∂b̂/∂log λ` and `V_ρ` the
inverse Hessian of the REML criterion over `(log λ, log σ²)`, and the
second-order Cholesky-derivative correction of Wood, Pya and Säfken (2016),
both as in mgcv 1.9-3 `gam.fit3.post.proc`. It exists only after REML
selection, and `settings.corrected_covariance` says whether it was computed
(`:mgcv_Vc`). When a smoothing parameter is effectively infinite,
`XᵀX + S_λ` is numerically singular and the derivatives with respect to
`log λ` carry no information, so `Vc` is not formed
(`:unavailable_infinite_lambda`); the same holds if the Cholesky
factorization fails (`:unavailable_not_positive_definite`). Bands and
intervals then default to `Vp`. mgcv still reports a `Vc` in these cases. Bands are pointwise normal intervals. refund offers no
simultaneous bands for `pfr`, and none are returned. `dof` is the total
effective degrees of freedom (intercept and scalar coefficients included),
`edf_terms` the effective degrees of freedom of each functional term, and
`sigma2` mgcv's `RSS/(n - dof)`.

### [Choosing a regression method](@id regression-method-choice)

Both fits return a tuple of coefficient functions, `coef`, `vcov`,
`fitted`, `residuals`, `predict` and `coefficient_bands`, so downstream code
can use either. FPC regression truncates to leading components of the
covariate and conditions on the FPCA; it suits strongly smooth covariates
whose leading modes carry the signal, and sparse designs through PACE scores.
Penalized regression lets the response choose the smoothness of `β` through
REML, does not tie `β` to the leading covariate modes, accounts for smoothing
parameter uncertainty in its bands, and is the better choice when `β` has
features in directions of small covariate variance or when several dense
covariates need separate smoothness.

The [regression study](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/master/benchmark/regression)
compares three fits with their default settings:

- `fit(FPCRegression, x, y)`;
- the same fit with `selection=BIC()`;
- `fit(PenalizedFunctionalRegression, x, y)`, with `presmooth=:fpca` for
  sparse curves.

Covariates are simulated with `simulate_kl` from nine Fourier components
with eigenvalues `1/k²`. They are observed with noise either on a dense grid
or at 2–10 random times per curve. The coefficient `β(t)` takes three forms:

- aligned with the two leading eigenfunctions;
- equal to the trailing eigenfunction, which `FVE(0.99)` drops in
  population;
- a smooth bump unrelated to the eigenfunctions (Reiss and Ogden 2007).

The other factors are `n ∈ {100, 500}` and a population R² of 0.8 or 0.5,
with 200 replicates per cell. The study records:

- relative error of `β̂`;
- prediction error on held-out subjects;
- pointwise coverage of the default 95% bands;
- component counts or effective degrees of freedom;
- time, allocations and failures.

Neither method offers simultaneous bands. The study instead reports how
often the pointwise band covers all of `β`.

The results below were obtained with the earlier FPC defaults: dense
covariates used the unsmoothed dense FPCA, and BIC searched every retained
component. They are the reason for the current FACE default and selection
cap ([FPC defaults that depart from fdapace](@ref regression-fpc-departures)).
`fpca_options=(; method=:dense)` reproduces the "FPC default" column, and adding
`ncomponents=FVE(1)` to those options with `selection=BIC()` reproduces the
uncapped "FPC with BIC" column.

Results from USC Discovery (Julia 1.13.2, 14,400 fits, no failures) are in
[`benchmark/regression/results`](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/master/benchmark/regression/results).
Median relative errors of `β̂`:

| Covariates | `β` | n | FPC default | FPC with BIC | Penalized |
|:--|:--|---:|---:|---:|---:|
| Dense | leading | 100 | 6.0–11.6 | 51–106 | 0.49–0.54 |
| Dense | leading | 500 | 4.3–8.5 | 0.06–0.08 | 0.15–0.49 |
| Dense | smooth | 100 | 5.1–10.5 | 44–89 | 0.34–0.71 |
| Dense | smooth | 500 | 3.8–7.6 | 0.20–0.44 | 0.21–0.32 |
| Dense | trailing | 500 | 0.68–1.25 | 0.13–0.15 | 1.0 |
| Sparse | leading | 100–500 | 0.27–0.95 | 0.15–0.44 | 0.55–0.65 |
| Sparse | smooth | 100–500 | 0.42–0.92 | 0.46–0.61 | 0.45–0.84 |
| Sparse | trailing | 100–500 | 1.0 | 1.0 | 1.0 |

Ranges span the two noise levels. On noisy dense curves with the defaults,
penalized regression is the safe choice: the default FPC fit keeps 40–63
noise-dominated components. BIC selection fixes that at n=500, where it is the
most accurate. At n=100, though, BIC chooses around 70 components out of
fewer than 100 and fails badly. Penalized regression shrinks a coefficient
equal to a high-frequency eigenfunction to zero (error 1), and no method
recovers it from sparse curves. Prediction error differs much less than
coefficient error. Pointwise coverage of the 95% bands is close to nominal
only for the default FPC fit, whose bands are very wide. Coverage is 27–95%
for BIC-selected FPC and for penalized fits on smooth or leading
coefficients, and under 8% where the penalty shrinks a trailing
coefficient, because the bands do not include smoothing bias.

### refund and mgcv source audit

The audited sources are refund 0.1-40 (`pfr.R`, `lf.R`, `create.prep.func.R`,
`coefficients.pfr.R`, `predict.pfr.R`) and mgcv 1.9-3
(`smooth.construct.tp.smooth.spec`, `tprs.c`, `smoothCon`, `gam.fit3.post.proc`,
`Vb.corr`, `vcorr`, `dchol`). Findings:

- `pfr` rewrites each `lf(X)` as `s(X.tmat, by=L.X)` with matrix arguments, so
  the term is `Σₜ L_t X(t) f(t)` (mgcv's summation convention) and calls
  `gam(method="REML")`. A centering constraint is added only when the row
  sums of `L*X` are constant; the package throws instead, because the term is
  then confounded with the intercept.
- mgcv computes the penalty scale from the unweighted basis at the grid,
  before the `by` weights are applied, and rescales thin plate columns to unit
  root mean square.
- The thin plate basis uses `η(r)=|r|³/12`, the `k` eigenvalues of largest
  magnitude by Lanczos iteration (tolerance `eps^0.7`) and a Householder
  absorption of `Tᵀδ = 0`. On grids symmetric about their mean (every uniform
  grid) the absorption depends on entries that are zero in exact arithmetic,
  so mgcv's columns there depend on rounding in its eigenvectors. The function
  space, the penalty, `β`, fitted values, `Vp` and `Ve` do not; mgcv's `Vc`
  does, through its Cholesky-based second-order term.
- `lf()` integration weights are as described above; `predict.pfr` reuses
  the training weights and presmoother for new curves on the training grid.
- `coef.pfr` uses `Vc` by default and returns standard errors only;
  `fpca.ssvd` presmoothing cannot predict new curves (refund warns).

### Validation

Recreate the fixture with `Rscript test/reference/refund_m7_pfr.R` (R 4.5.1,
refund 0.1-40, mgcv 1.9-3). `test/pfr.jl` compares, at relative error below
`1e-8`: the thin plate basis and penalty (up to column signs) on three point
sets, including the untruncated case; and, for one functional covariate, two
functional covariates, GCV, and B-spline presmoothing (the package's
trapezoidal default against refund's `integration="trapezoidal"`), and for
refund's default `integration="simpson"` on a uniform grid (through
`integration=:refund_simpson`), each with a scalar covariate, the intercept and
scalar coefficients, fitted values, `σ²`, the criterion, per-term and total
EDF, `β` and its `Vp` standard errors on refund's default grid, and held-out
predictions with standard errors. These comparisons are made at refund's
selected smoothing parameters, and the package's own optimum is checked to be
no worse under the criterion. mgcv's `Vc` standard errors are compared on the
grids that are not symmetric about their mean, where mgcv's basis is
deterministic. No R installation is needed to use the package or run its
tests.

```@docs
PenalizedFunctionalRegression
ThinPlateRegressionBasis
StatsAPI.predict(::PenalizedFunctionalRegression, ::Any)
StatsAPI.vcov(::PenalizedFunctionalRegression)
StatsAPI.coef(::PenalizedFunctionalRegression)
StatsAPI.fitted(::PenalizedFunctionalRegression)
StatsAPI.residuals(::PenalizedFunctionalRegression)
StatsAPI.dof(::PenalizedFunctionalRegression)
StatsAPI.dof_residual(::PenalizedFunctionalRegression)
StatsAPI.r2(::PenalizedFunctionalRegression)
StatsAPI.stderror(::PenalizedFunctionalRegression)
```

## Penalized function-on-function regression

`PenalizedFunctionOnFunctionRegression` fits functional responses on
functional and scalar covariates with penalized splines, as refund's
`pffr(Y ~ z + ff(X1) + ff(X2))` (Ivanescu, Staicu, Scheipl and Greven 2015;
Scheipl, Staicu and Greven 2015):

```math
Y_i(s) = \beta_0(s) + \sum_k z_{ik}\beta_k(s) +
\sum_j \sum_t L_{jt} X_{ij}(t)\,\beta_j(t,s) + \epsilon_i(s),
\qquad \beta_j(t,s) = \sum_{a,b} \theta_{jab} B_a(t) B_b(s).
```

Each surface `β_j` is a tensor product of cubic P-splines with two smoothing
parameters, one per direction. The intercept and scalar-term functions are
those of [`FunctionOnScalarRegression`](@ref) with `basis=:pffr`. Every
smoothing parameter is chosen by REML on the stacked observations
`(i, s)` through [`PenalizedLeastSquares`](@ref), and the default covariance is
`pffr`'s cluster sandwich, which allows for correlated errors within a curve.

```@example pffr
using FunctionalDataAnalysis, LinearAlgebra, Random
import StatsAPI
rng = Xoshiro(11)
tgrid = collect(range(0, 1; length=30))
sgrid = collect(range(0, 1; length=25))
modes = hcat(sinpi.(tgrid), cospi.(tgrid), sinpi.(2 .* tgrid), cospi.(2 .* tgrid))
X = randn(rng, 60, 4) * transpose(modes) .+ 0.05 .* randn(rng, 60, 30)
z = randn(rng, 60)
surface_true(t, s) = cospi(t - s) * (1 + s)
Y = transpose(1 .+ sinpi.(2 .* sgrid)) .+ z .* transpose(1 .- 2 .* sgrid .^ 2) .+
    X * (quadrature_weights(tgrid) .* surface_true.(tgrid, transpose(sgrid))) .+
    0.1 .* randn(rng, 60, 25)
x = GridFunctionalData(X, tgrid)
y = GridFunctionalData(Y, sgrid)
pffr = StatsAPI.fit(PenalizedFunctionOnFunctionRegression, x, y; scalar_covariates=z)
(pffr.lambda, pffr.edf, pffr.settings.sigma2)
```

`lambda` lists the intercept, the scalar coefficient, then the `t` and `s`
directions of the surface; `edf` the effective degrees of freedom of the
intercept, the scalar coefficient and the surface.

```@example pffr
bands = coefficient_bands(pffr)
surface = only(bands.functional)   # (; grid, response_grid, estimate, se, lower, upper, ...)
slope = bands.scalar[2]            # pointwise band for the coefficient function of z
curves = StatsAPI.predict(pffr, x[1:3]; scalar_covariates=z[1:3])
(size(surface.estimate), surface.settings.covariance, size(curves.values))
```

The same data fitted by FPC function-on-function regression
([`FPCRegression`](@ref) with a functional response) give a surface in the
leading covariate and response FPCs. With covariates spanned by four smooth
modes, both recover the surface; the penalized fit does not depend on how
much of the surface lies in the leading covariate modes.

```@example pffr
fpc = StatsAPI.fit(FPCRegression, x, y; scalar_covariates=z)
relative_error(s) = norm(s.values - surface_true.(s.grid, transpose(s.response_grid))) /
    norm(surface_true.(s.grid, transpose(s.response_grid)))
(penalized=relative_error(only(coefficient_functions(pffr).functional)),
    fpc=relative_error(only(coefficient_functions(fpc))), fpc_components=ncomponents(fpc))
```

### [Defaults and their sources](@id pffr-defaults)

| Choice | Default and source |
|:--|:--|
| Surface basis | `k=(5, 5)`: cubic P-splines in `t` and `s` with first-order difference penalties, refund `ff()`'s `splinepars = list(bs="ps", m=list(c(2,1), c(2,1)), k=c(5,5))` with `basistype="te"` |
| Surface penalties | mgcv `te`: each margin reparameterized to function values at `k` equally spaced points of its range and normalized to unit largest eigenvalue; `λ` refers to these unscaled penalties (mgcv's `sp` divided by `S.scale`) |
| Knots | mgcv `ps` knots on the covariate grid range and on the range of the observation times (the whole grid for a grid response) |
| Intercept and scalar terms | `basis=:pffr`: P-splines with 20 functions for the intercept and 5 per scalar term, `pffr`'s `bs.int` and `bs.yindex` |
| Integration | `integration=:quadrature` (refund's `"trapezoidal"`); **departs** from `ff()`'s `"simpson"`, for the reason given for [`PenalizedFunctionalRegression`](@ref) (refund's Simpson weights divide by the number of points instead of intervals); `:refund_simpson` reproduces it |
| Criterion | `criterion=:REML`, `pffr(method="REML")`; `:GCV` is mgcv `"GCV.Cp"` |
| Covariance | `covariance=:cluster`, `pffr`'s default `sandwich="cluster"`; `:bayesian` (`Vp`, `sandwich="none"`) and `:frequentist` (`Ve`) |
| Coefficient bands | Pointwise normal, `alpha=0.05`, on the covariate grid and the response grid (40 equally spaced points for an irregular response, `coef.pffr`'s `n2=40`) |
| Prediction | Curves on the response grid, or on the sorted observation times of an irregular response (`predict.pffr`'s stored `yind`) |
| Covariates | Complete grid data as observed; `presmooth`, `presmooth_options` and `grid` as for [`PenalizedFunctionalRegression`](@ref) |

`k` takes one integer for both directions, a tuple `(k_t, k_s)`, or a vector
with one entry per functional covariate. Fixed `lambda` is a vector with one
value per penalty in the order of `lambda` above. `scalar_covariates` accepts a
vector, a matrix or a Tables.jl table with treatment-coded factors, as
[`FunctionOnScalarRegression`](@ref) does. Every choice is stored in
`settings`, the per-term records `terms` (grid, weights, marginal bases,
penalties, presmoother, processed curves and identifiability diagnostics) and
shown by `show`.

### [Responses on grids and at irregular times](@id pffr-responses)

A grid response may have missing values, and an `IrregularFunctionalData`
response is `pffr`'s `ydata`: the model is fitted at the observed points with
unit weights, so each observed value is one row of the stacked problem.
`fitted` returns curves on the whole grid for a grid response and at the
observation times for an irregular one; `residuals` keeps missing grid values
missing. Basis responses throw (`pffr` takes no `fd` objects); convert them
with `to_grid`. The stacked design is dense, with one row per observed value
and `20 + 5q + Σ_j k_t k_s` columns for `q` scalar covariates. refund switches
from `gam` to `bam` above 10⁵ observations; the package always solves the
same penalized least squares problem.

### [Uncertainty](@id pffr-uncertainty)

`vcov(fit)` is the covariance chosen by `covariance`. The cluster sandwich
`n/(n-1) A⁻¹ (Σ_i D_iᵀr_i r_iᵀD_i) A⁻¹ + Vp - Ve` (with `n` curves,
`A = DᵀD + S_λ`, and the rows `D_i` and residuals `r_i` of curve `i`) accounts for correlation of the errors
within a curve; `Vp` treats the points of one curve as independent. Surface
bands use the variance `(B(t) ⊗ B(s))ᵀ V (B(t) ⊗ B(s))` of each point.
Prediction intervals add the error variance `settings.sigma2` (mgcv's
`sig2`). Bands and intervals are pointwise; smoothing parameters and
presmoothed covariates are treated as fixed. `dof` is the total effective
degrees of freedom and `dof_residual` the number of observed values minus
`dof`.

### [refund and mgcv source audit for pffr](@id pffr-audit)

The audited sources are refund 0.1.40 (`pffr`, `pffr_prepare`,
`pffr_process_ff_terms`, `pffr_configure_algorithm`, `ff`,
`compute_integration_weights`, `trace_lv`, the arguments of `coef.pffr`) and
mgcv 1.9-3 (`smoothCon`, `smooth.construct.tensor.smooth.spec`,
`smooth.construct.ps.smooth.spec`). Findings:

- `pffr` stacks the response in long format (one row per observed value) and
  rewrites `ff(X)` as `te(X.smat, X.tmat, by=L.X)` with matrix arguments:
  `X.smat` repeats the covariate grid, `X.tmat` the response index of the row
  and `L.X` the weighted curve, so mgcv's summation convention gives
  `Σ_t L_t X_i(t) f(t, s)`. It then calls `gam(method="REML")`.
- `ff()` defaults to `integration="simpson"` with refund's formula
  `(t_J - t_1)/J/3 · [1, 4, 2, …, 1]`, switching to `"trapezoidal"` on grids
  that are not equidistant. `limits` (historical models) switch to Riemann
  sums; they are not supported here.
- mgcv `te` reparameterizes each one-dimensional margin to function values
  at equally spaced points of the margin's data range (skipped with a
  warning when the evaluation matrix is near singular, which the package
  mirrors) and divides each marginal penalty by its largest eigenvalue before
  forming `S_t ⊗ I` and `I ⊗ S_s`. `smoothCon` then scales each penalty by
  `‖S‖₁/‖X‖∞²` from the unweighted basis, before the `by` weights.
- With a numeric matrix `by` variable, mgcv adds no centering constraint
  unless the row sums of `L.X` (the integrals `∫X_i`) are constant; the
  package throws in that case, as the surface is then confounded with the
  intercept.
- `ff(check.ident=TRUE)` warns when the covariate's effective rank (99.5% of
  its squared singular values) is at most 4 or below `k_t`, and replaces the
  term by a constrained `ti()` smooth when the null space of `X` overlaps the
  integration-weighted null space of the `t` penalty (overlap above 0.95) and
  the term's design has log10 condition number above 6. The package stores
  the three diagnostics on the term and throws where refund would constrain.
- The functional intercept is a parametric constant plus a centered smooth;
  the package's uncentered P-spline spans the same space with the same
  penalty, so fits agree, but the REML score differs by a constant that does
  not depend on `λ`. The tensor penalties are applied in the original
  B-spline coordinates, `S_t ⊗ G_s` and `G_t ⊗ S_s` with `G = B(u)ᵀB(u)` at
  the reparameterization points `u`, which adds another `λ`-free constant.

### [Validation](@id pffr-validation)

Recreate the fixtures with `Rscript test/reference/regression_m7f.R` (R 4.5.1,
refund 0.1.40, mgcv 1.9-3). `test/regression_pffr.jl` compares, at relative
error below `1e-8` and at refund's selected smoothing parameters:
`pffr(Y ~ ff(X))` with trapezoidal weights (the package default),
`pffr(Y ~ z + ff(X))` with refund's default Simpson weights
(`integration=:refund_simpson`), and the first model with an irregular
response (`ydata`). The quantities are the surface on a 9 × 11 grid, its
standard errors under `Vp`, `Ve` and the cluster sandwich, the intercept and
scalar coefficient functions and their standard errors, per-term and total
EDF, `σ²`, fitted values and predictions for new curves. Because the REML
scores differ by a constant, the change in the score between refund's
smoothing parameters and twice them is compared instead, and the package's
own optimum is checked to be no worse under its criterion. The irregular fit
is also checked against the same points on a grid with missing values. No R
installation is needed to use the package or run its tests.

```@docs
PenalizedFunctionOnFunctionRegression
StatsAPI.predict(::PenalizedFunctionOnFunctionRegression, ::Any)
StatsAPI.coef(::PenalizedFunctionOnFunctionRegression)
StatsAPI.vcov(::PenalizedFunctionOnFunctionRegression)
StatsAPI.stderror(::PenalizedFunctionOnFunctionRegression)
StatsAPI.nobs(::PenalizedFunctionOnFunctionRegression)
StatsAPI.dof(::PenalizedFunctionOnFunctionRegression)
StatsAPI.dof_residual(::PenalizedFunctionOnFunctionRegression)
StatsAPI.fitted(::PenalizedFunctionOnFunctionRegression)
StatsAPI.residuals(::PenalizedFunctionOnFunctionRegression)
```

## [Exponential-family responses](@id regression-glm)

Both scalar-response routes accept `family=:binomial` (0/1 responses),
`family=:poisson` (counts) or `family=:gaussian` with a non-identity `link`,
for the model `g(E yᵢ) = ηᵢ = α + zᵢᵀγ + Σⱼ ∫ Xᵢⱼ(t) βⱼ(t) dt`. `link=nothing`
selects the canonical link (logit, log, identity); the other links are those
listed for [`PenalizedLeastSquares`](@ref). Coefficient functions, their
bands and standard errors are on the link scale.

```@example regression
η = 0.5 .+ 0.3 .* z .+ 2 .* X * (quadrature_weights(grid) .* sinpi.(2 .* grid))
# Poisson counts: arrivals of a unit-rate process before time exp(η).
counts = [count(<(exp(e)), cumsum(randexp(rng, 100))) for e in η]
poisson = StatsAPI.fit(PenalizedFunctionalRegression, x, counts; scalar_covariates=z,
    family=:poisson)
(poisson.lambda, poisson.edf_terms, StatsAPI.r2(poisson))
```

```@example regression
fpc_poisson = StatsAPI.fit(FPCRegression, x, counts; scalar_covariates=z, family=:poisson)
(ncomponents(fpc_poisson), StatsAPI.r2(fpc_poisson),
    StatsAPI.predict(fpc_poisson, x; scalar_covariates=z, type=:link)[1:3])
```

- **FPC route.** [`FPCRegression`](@ref) fits a GLM on the leading FPC scores
  (and scalar covariates) by IRLS to the maximum likelihood estimate, as
  fda.usc 2.2.0 `fregre.glm` does through R's `glm` with a
  `create.pc.basis` basis; fdapace `FLM` is Gaussian only. `vcov` is
  `(XᵀW_F X)⁻¹` times the dispersion (one for binomial and Poisson, the
  Pearson statistic over `n - p` for the Gaussian family), as `summary.glm`.
  Intervals are Wald intervals on the link scale with normal quantiles for
  binomial and Poisson and Student-t quantiles when the Gaussian dispersion
  is estimated. `AIC()` and `BIC()` selection uses R's `AIC()`/`BIC()` for
  `glm`; `:loocv` (held-out squared error) needs the Gaussian identity link.
  Scores that separate binomial classes (fitted probabilities numerically 0
  or 1, where `glm` warns) make the fit throw. The FPCA, scores and component
  choice are conditioned on, as for least squares; `coefficient_bands(...; method=:bootstrap)` refits the FPCA and the
  GLM in every replicate.
- **Penalized route.** [`PenalizedFunctionalRegression`](@ref) fits the
  penalized model of refund 0.1-40 `pfr(y ~ z + lf(X), family=..., method="REML")`
  by penalized IRLS with Laplace-approximate REML (or UBRE / GCV), through
  the extended [`PenalizedLeastSquares`](@ref) solver, which validates
  against mgcv 1.9-3 `gam` (see [Penalized fitting](@ref penalized-glm)).
  `coefficient_bands` defaults to mgcv's `Vc` after REML selection, as
  refund's `coef.pfr`, computed from the Fisher-weighted Hessian, the
  implicit derivatives `dβ/d log λ` and the Laplace REML Hessian.
- **Shared interface.** `predict(f, x; type=:response)` returns means and
  `type=:link` the linear predictor. `interval=:confidence` computes the
  interval for `η` and, with `type=:response`, maps the estimate and both
  endpoints through the inverse link, so the interval stays inside the
  response range (`se` stays on the link scale; `settings.scale` records the
  scale). Prediction intervals for a new response (`interval=:prediction`)
  are defined only for the Gaussian identity link and throw otherwise.
  `fitted_bands(f; type)` does the same for the training subjects. `fitted`
  returns means and `residuals` the response-scale residuals `y - μ`; `r2`
  is the deviance explained `1 - D/D₀`, which reduces to `1 - RSS/TSS` for
  least squares. `sigma2` is the dispersion.

| Choice | Default and source |
|:--|:--|
| Family and link | `family=:gaussian`, `link=nothing` (canonical): the defaults of `glm`, `gam` and `fregre.glm` |
| Penalized criterion | `:REML`, refund `pfr(method="REML")`; `:UBRE` for known scale and `:GCV` for estimated scale are mgcv `"GCV.Cp"` |
| Prediction scale | `type=:response`: fda.usc `predict.fregre.glm` and the declared default of refund `predict.pfr` (see below) |
| FPC GLM dispersion | one (binomial, Poisson); Pearson statistic over `n - p` (Gaussian), `summary.glm` |
| Penalized scale | one (binomial, Poisson); Pearson statistic over `n - τ` (Gaussian), mgcv `scale.est` |
| FPC GLM intervals | link-scale Wald: normal (known scale) or Student-t with `n - p` degrees of freedom (Gaussian), as `summary.glm` |
| Penalized intervals | link-scale normal, `Vc` for coefficient bands, `Vp` for predictions (refund `coef.pfr`, mgcv `predict.gam`) |
| IRLS tolerance | Newton decrement at most `1e-12 (1 + Dₚ)`, at most 200 iterations: a package choice, tighter than `glm`'s `epsilon=1e-8` and mgcv's `1e-7` |

### Source audit

- fda.usc 2.2.0 `fregre.glm` builds the design from `fdata2model.penalty`
  (intercept, scalar covariates, then the PC scores of `create.pc.basis`) and
  calls `glm()` without passing a `control`, so the fit stops at `glm`'s
  default `epsilon=1e-8`; on the validation fixtures its coefficients are
  about `6e-8` (relative) from the converged estimate. Its `r2` is
  `1 - sum(e²)/sum((y - ȳ)²)` with the working residuals `e` of the final
  IRLS iteration, which is not a goodness-of-fit measure for a GLM; the
  package reports the deviance explained instead. `predict.fregre.glm`
  projects new curves with `fdata2basis` and the stored `vs.list` and passes
  `type="response"` (its default) to `predict.glm`.
- R's `glm.fit` returns the weights of the start of its last iteration;
  `summary.glm` uses them for `cov.unscaled`. The package evaluates the
  Fisher information at the converged estimate.
- refund 0.1-40 `pfr` passes `family` and `control` to mgcv's `gam`. Its
  `predict.pfr` declares `type="response"` but forwards only the arguments
  present in `match.call()` to `predict.gam`, whose default is
  `type="link"`; called without `type`, it returns the linear predictor. The
  package follows the declared default, `type=:response`, which also matches
  fda.usc.
- mgcv's weight conventions for non-canonical links (Newton weights in the
  IRLS and the REML log-determinant, Fisher weights in `τ`, `edf`, `Vp`, the
  scale and `Vc`) are recorded on the [Penalized fitting](@ref penalized-glm)
  page; `Vb.corr` ignores the weight derivatives in the second-order `Vc` term.

### Validation

`test/glm.jl` compares, at relative error below `1e-8`:

- the penalized solver with mgcv 1.9-3 for six binomial, Poisson and
  Gaussian-log designs under REML and GCV/UBRE (see
  [Penalized fitting](@ref penalized-glm));
- `PenalizedFunctionalRegression` with refund 0.1-40 `pfr(family=)` for
  binomial (REML and UBRE), Poisson and Gaussian-log responses with a scalar
  covariate and refund's `integration="trapezoidal"`: the intercept and scalar
  coefficients, means, linear predictors, scale, criterion, total EDF,
  deviance (and with it the deviance explained), `β` and its `Vp` and `Vc`
  standard errors on refund's default grid, and held-out predictions on both
  scales with link-scale standard errors, at refund's selected smoothing
  parameters; the package's optimum is no worse under the criterion;
- `FPCRegression` with the converged `glm` estimate on fda.usc 2.2.0
  `fregre.glm`'s own three-component design for binomial, Poisson (with a
  scalar covariate) and Gaussian-log responses: the intercept, means,
  deviance, null deviance, dispersion, `β`, its standard errors, held-out
  predictions on both scales with link-scale standard errors, and R's AIC and
  BIC.

Recreate the references with `Rscript test/reference/mgcv_m7g.R`,
`Rscript test/reference/refund_m7g_pfr.R` and
`Rscript test/reference/fdausc_m7g.R` (R 4.5.1; mgcv 1.9-3, refund 0.1-40,
fda.usc 2.2.0). No R installation is needed to use the package or run its
tests.

## FPC regression reference

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
