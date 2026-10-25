# Regression

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

## Fit and inspect a model

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
coordinates = band_data(pointwise) # bandplot(pointwise) uses the existing ribbon recipe
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

## FPCA and component choices

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

## Prediction and uncertainty

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
to `level=0.95` and `R=999` subject-bootstrap refits of FPCA and regression.
The conditional intervals here follow fda.usc's linear-model covariance and
Student-t prediction formula; they do not reproduce `FLMCI` bootstrap intervals.

## Reference source audit and validation

The audited sources are [fdapace 0.6.0](https://cran.r-project.org/src/contrib/fdapace_0.6.0.tar.gz)
(`FLM.R`, `FLMCI.R`, `SetOptions.R`) and
[fda.usc 2.2.0](https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz)
(`fregre.pc.R`, `fdata2pc.R`, `predict.fregre.fd.r`, `inprod.fdata.R`,
`ops.fda.usc.R`). The reference generator uses R 4.5.1.

`FLM` fits marginal FPCA without changing its defaults, concatenates `xiEst`
and scalar columns, then calls `lm`. Sparse predictors therefore use PACE CE
scores through the ordinary FPCA route. Held-out curves use `predict.FPCA`
with the training component count. `fregre.pc` defaults to `l=1:3`, `lambda=0`
and unit observation weights. The package follows fdapace's FVE default because
it adapts the count to the functional predictor. A fixed `ncomponents=3`
provides the corresponding three-leading-component model.

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

Recreate the small fixtures from the repository root with
`Rscript test/reference/regression_m7.R`, using the package versions above.
`test/regression.jl` compares coefficient functions, scalar coefficients,
fitted values, residuals, residual variance, R², held-out predictions and
conditional intervals at relative error below `1e-8`. It also checks
full-refit LOOCV and AIC/BIC against R, and independent analytic nonuniform-grid,
basis and sparse conditional-expectation examples. No R installation is
required to use the package or run its tests.

```@docs
FPCRegression
coefficient_functions
coefficient_bands
StatsAPI.predict(::FPCRegression, ::Any)
StatsAPI.coef(::FPCRegression)
StatsAPI.vcov(::FPCRegression)
StatsAPI.fitted(::FPCRegression)
StatsAPI.residuals(::FPCRegression)
StatsAPI.dof(::FPCRegression)
StatsAPI.dof_residual(::FPCRegression)
StatsAPI.r2(::FPCRegression)
StatsAPI.stderror(::FPCRegression)
```
