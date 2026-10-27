# Penalized regression

Penalized regression expands each coefficient function in a spline basis and
chooses its smoothing parameters by REML (default) or GCV through
[`PenalizedLeastSquares`](@ref). This page covers penalized scalar-on-function
regression (refund `pfr`), penalized function-on-function regression (refund
`pffr`), and exponential-family responses for both scalar-response routes.
FPC regression, concurrent regression and function-on-scalar regression are
on the [Regression](regression.md) page.

## Penalized scalar-on-function regression

`PenalizedFunctionalRegression` fits the Gaussian scalar-on-function model of
the [Regression](regression.md) page with each
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

```@example penreg
using FunctionalDataAnalysis, Random
import StatsAPI
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

```@example penreg
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

Results from USC Discovery (Julia 1.13.2, 14,400 fits per run, no failures)
are in [`benchmark/regression/results`](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/master/benchmark/regression/results).
The table gives median relative errors of `β̂` with the current defaults, and,
in brackets, with the earlier defaults (unsmoothed dense FPCA, uncapped
criterion search):

| Covariates | `β` | n | FPC default | FPC with BIC | Penalized |
|:--|:--|---:|---:|---:|---:|
| Dense | leading | 100 | 0.58–1.1 (6.0–11.6) | 0.15–0.19 (51–106) | 0.49–0.54 |
| Dense | leading | 500 | 0.23–0.49 (4.3–8.5) | 0.06–0.08 (same) | 0.15–0.49 |
| Dense | smooth | 100 | 0.46–0.99 (5.1–10.5) | 0.42–0.50 (44–89) | 0.33–0.71 |
| Dense | smooth | 500 | 0.21–0.43 (3.8–7.6) | 0.20–0.44 (same) | 0.21–0.31 |
| Dense | trailing | 100–500 | 0.92–0.99 (0.68–1.7) | 1.0 (0.13–15) | 1.0 |
| Sparse | leading | 100–500 | 0.27–0.95 | 0.15–0.44 | 0.55–0.65 |
| Sparse | smooth | 100–500 | 0.42–0.92 | 0.45–0.61 | 0.45–0.84 |
| Sparse | trailing | 100–500 | 1.0 | 1.0 | 1.0 |

Ranges span the two noise levels; sparse covariates use PACE in both runs, so
their numbers are unchanged. With the current defaults, FPC regression
smooths dense curves with FACE (about eight components instead of 40–63
noise-dominated ones) and searches criterion counts only up to the FPCA's own
selection. BIC-selected FPC regression is then the most accurate for
coefficients in the leading modes, and penalized regression is close or
better for smooth coefficients. The cost is the trailing case: a coefficient
equal to a high-frequency eigenfunction is smoothed away by FACE, as it is by
the penalty (error about 1), whereas the earlier unsmoothed BIC fit
recovered it at n=500. Pass `fpca_options=(; method=:dense)` when such
structure is expected and n is large. No method recovers it from sparse
curves. Prediction error differs much less than coefficient error.
Pointwise coverage of the 95% bands is 69–96% for the default FPC fit, 26–86%
for BIC-selected FPC and 43–95% for penalized fits on leading or smooth
coefficients, and at most 32% on trailing coefficients, because the bands do
not include smoothing bias.

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

```@example penreg
η = 0.5 .+ 0.3 .* z .+ 2 .* X * (quadrature_weights(grid) .* sinpi.(2 .* grid))
# Poisson counts: arrivals of a unit-rate process before time exp(η).
counts = [count(<(exp(e)), cumsum(randexp(rng, 100))) for e in η]
poisson = StatsAPI.fit(PenalizedFunctionalRegression, x, counts; scalar_covariates=z,
    family=:poisson)
(poisson.lambda, poisson.edf_terms, StatsAPI.r2(poisson))
```

```@example penreg
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

