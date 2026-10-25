# Penalized fitting

[`PenalizedLeastSquares`](@ref) is the Gaussian engine behind penalized
regression. It minimizes

```math
\|W^{1/2}(y - X\beta)\|^2 + \sum_{j=1}^{M} \lambda_j \beta^\top S_j \beta
```

for a design `X`, known positive observation weights `W`, and symmetric
positive semidefinite penalties `S_j`. Each penalty acts on a block of columns
and may be rank deficient. Several penalties may act on the same block, as for
the two marginal roughness penalties of a tensor-product coefficient. Columns in
no penalty are unpenalized. The smoothing parameters `λ_j` are fixed, or chosen
by REML (the default) or GCV.

```@example penalized
using FunctionalDataAnalysis
import StatsAPI

t = collect(range(0.0, 1.0; length=60))
z = cos.(7 .* t)
y = 0.5 .* z .+ sin.(2pi .* t) .+ 0.2 .* sin.(37 .* t .^ 2)
basis = BSplineBasis(4, (0.0, 1.0), 12)
X = hcat(z, basis_matrix(basis, t))
# One roughness penalty on the spline block; column 1 is unpenalized.
model = StatsAPI.fit(PenalizedLeastSquares, X, y;
    penalties=[penalty(basis, 2) => 2:13])
(model.lambda, model.lambda_status, model.edf_total, model.sigma2)
```

```@docs
PenalizedLeastSquares
```

## Criteria

Write `S_λ = Σ λ_j S_j`, `H = XᵀWX + S_λ`, `β̂ = H⁻¹XᵀWy`,
`D = ‖W^{1/2}(y - Xβ̂)‖²` (weighted residual sum of squares),
`Dₚ = D + β̂ᵀS_λβ̂`, `τ = tr(H⁻¹XᵀWX)` (the trace of the influence matrix, or
effective degrees of freedom), `n` the number of observations and `Mₚ` the
dimension of the null space of `S_λ`. With the scale `φ` unknown:

| `criterion` | Criterion minimized | mgcv |
| --- | --- | --- |
| `:REML` (default) | `Dₚ/(2φ) + (n-Mₚ)/2 log(2πφ) + log|H|/2 - log|S_λ|₊/2 - Σ log wᵢ/2` at `φ = Dₚ/(n-Mₚ)` | `gam(..., method="REML")` |
| `:GCV` | `n D/(n - τ)²` | `gam(..., method="GCV.Cp")` |

`log|S_λ|₊` is the sum of the logarithms of the positive eigenvalues. The REML
default follows refund (`pfr` and `pffr` call `gam` with `method="REML"`); GCV
is the criterion of fda. Profiling out `φ` gives the same minimizer and minimum
as mgcv, which carries `log φ` as an extra Newton parameter.

## Optimization and numerical stability

Smoothing parameters are optimized on `ρ = log λ` by Newton's method with the
exact gradient and Hessian of the criterion. The Hessian is made positive
definite through the absolute values of its eigenvalues, with a floor of
`sqrt(eps)` times the largest; steps are scaled to at most 5 on the log scale
and halved up to 30 times until the criterion decreases (mgcv's `newton`
defaults, `maxNstep=5` and `maxHalf=30`). The starting point is mgcv's
`initial.sp` heuristic: for each penalty, the ratio of the mean diagonal of
`XᵀWX` to the mean diagonal of the penalty over its penalized columns, all
rescaled by one power of ten so that the mean over penalized columns of
`diag(XᵀWX) ./ diag(H)` is at least 0.4 but falls below 0.4 after a further
tenfold increase. Starting values are clamped to the bounds.

Determinants, solves and traces are computed after Wood's (2011, appendix B)
similarity transform. Within every group of penalties that share columns the
coordinates are split into nested ranges: first the range of the penalties of
largest magnitude, then, within its complement, the range of the next largest,
and finally the common null space. Each transformed penalty is set exactly to
zero outside its leading block, and positive definite matrices are factorized
after diagonal scaling. Both the penalty log-determinant and the penalized
Hessian then stay accurate when smoothing parameters differ by many orders of
magnitude. Penalty ranks are judged on penalties scaled by their own norms, so
`Mₚ` does not change with `λ`; `rank_tol` sets the relative eigenvalue
threshold (default `eps^(3/4)`).

When the data do not identify a penalized direction, the criterion keeps
decreasing as its `λ_j` grows (for GCV also as it shrinks towards zero), and
the gradient and curvature vanish. Once the remaining gradient is below
tolerance, such a parameter keeps moving in steps of 5 on the log scale while
the criterion still decreases in working precision, but never beyond `e^60`
times its starting value. Its `lambda_status` is then `:effectively_infinite`
(or `:effectively_zero`): further changes do not alter the fit. This is the
limit `λ_j → ∞`, a fit constrained to the null space of `S_j`, computed with a
large finite `λ_j`.

Bounds: `lower` and `upper` (on `λ`, not `log λ`) constrain each parameter; a
binding bound is reported as `:lower_bound` or `:upper_bound` with `λ_j`
exactly equal to it. Equal bounds, or the `lambda` keyword, fix parameters.
A parameter fixed at zero drops its penalty, and with it its contribution to
the penalty rank.

## What is stored

Every fit stores `β`, the smoothing parameters and their statuses, the
criterion and its value, per-coefficient and total effective degrees of
freedom, the per-penalty sums of effective degrees of freedom (penalties on a
shared tensor block each report that block's total once, so these sums need not
add up to `edf_total`), `sigma2`, the
REML scale, both covariance matrices, fitted values, residuals and weights.
The `settings` named tuple records the bounds, initial smoothing parameters,
`tol`, `maxiter`, `rank_tol`, Newton constants, the number of iterations, how
the search ended (`:gradient`, `:no_decrease`, `:maxiter` or `:fixed`), whether
the criterion is defined, and the final gradient and Hessian on the log scale.
`show` prints all of it.

When every `λ_j` is fixed the criterion is only reported. Where it is undefined
(REML with `n ≤ M_p` or a zero penalized residual sum of squares, GCV with
`n - τ ≈ 0`) the fit still returns `β`, the effective degrees of freedom and
the fitted values, with `criterion_value = NaN` and
`settings.criterion_defined = false`. `sigma2` (and with it both covariances)
is `NaN` when `n - τ ≈ 0`, and the REML scale when `n ≤ M_p`. During selection
such smoothing parameters are inadmissible.

| Quantity | Field | mgcv `gamObject` |
| --- | --- | --- |
| `β` | `coef` | `coefficients` |
| `λ` | `lambda` | `sp` (plus `min.sp` when that is used) |
| criterion value | `criterion_value` | `gcv.ubre` |
| `diag(H⁻¹XᵀWX)` | `edf` | `edf` |
| `τ` | `edf_total` | `sum(edf)` |
| `edf` summed over the columns of `S_j` | `edf_penalty[j]` | `sum(edf[cols_j])` |
| `D/(n - τ)` | `sigma2` | `sig2`, `scale` |
| `Dₚ/(n - Mₚ)` | `reml_scale` | `reml.scale` (REML only) |
| `H⁻¹ sigma2` | `vcov_bayesian`, `vcov(fit)` | `Vp`, `vcov(fit)` |
| `H⁻¹XᵀWX H⁻¹ sigma2` | `vcov_frequentist`, `vcov(fit; type=:frequentist)` | `Ve`, `vcov(fit, freq=TRUE)` |

## Source audit: mgcv 1.9-3

The following was read from the mgcv 1.9-3 sources (R 4.5.1) and confirmed
numerically against `gam` fits.

- `sig2`, the scale used in `Vp` and `Ve`, is the Pearson statistic
  `D/(n - τ)` for both REML and GCV (`gam.fit3.post.proc`), not the REML
  estimate `reml.scale = Dₚ/(n - Mₚ)`. The two coincide at a REML optimum.
- `Vp = H⁻¹ sig2`; `Ve = F Vp` with `F = H⁻¹XᵀWX`; `edf = diag(F)`.
- The Gaussian REML score includes `-Σ log wᵢ/2` for prior weights.
- `paraPen` penalties are used on the supplied scale: `sp` multiplies the
  matrices as given.
- With `min.sp`, the reported `sp` excludes the bound; the penalty multiplier
  is `min.sp + sp`. Bounds are thus approached but never attained, whereas
  [`PenalizedLeastSquares`](@ref) reports a binding bound exactly. mgcv has no
  upper bounds.
- `newton` stops on a gradient tolerance of `conv.tol = 1e-6` relative to the
  score scale. Our default `tol = 1e-8` is tighter; a flat direction moves on
  until the criterion stops decreasing, where mgcv leaves a large finite `sp`.

## Validation

`test/penalized.jl` compares fits with `gam(method="REML")` and
`gam(method="GCV.Cp")` from mgcv 1.9-3 under R 4.5.1. The designs and
penalties are this package's own `BSplineBasis` and `FourierBasis` matrices
(`test/reference/mgcv_m7_inputs.jl`), passed to mgcv through `paraPen`: one
penalty with an unpenalized covariate and a two-dimensional null space,
observation weights, two penalties on disjoint blocks, two penalties on one
tensor-product block, a linear effect whose smoothing parameter diverges, and a
binding lower bound (`min.sp`). At mgcv's smoothing parameters, `β`, fitted
values, `edf`, their sum, `sigma2`, the criterion and the upper triangle of `Vp`
agree to relative error below `1e-8`. Starting from the default, the optimizer
reaches a criterion no larger than mgcv's (within `8eps`). Separately, the
criterion, coefficients and degrees of freedom are checked against textbook
formulas, the derivatives against finite differences, and a tensor fit with
smoothing parameters `1e-8` and `1e12` against 256-bit arithmetic.

To recreate the reference values:

```sh
julia --project=. test/reference/mgcv_m7_inputs.jl
Rscript test/reference/mgcv_m7.R
```

## Relation to other packages

MixedModels.jl fits REML for Gaussian mixed models, but each random-effect
term carries one variance parameter, so it cannot place two penalties with
separate smoothing parameters on one coefficient block. The inner problem here
is a linear solve, and the outer problem a small smooth optimization with
exact Hessians; neither needs proximal or coordinate-descent methods.

## Penalized splines with REML

`PenalizedSpline(basis; lambda=REML())` selects one smoothing parameter for all
curves with this solver. The curves are independent blocks with their own
coefficients, one smoothing parameter and one noise variance, so the
criterion is the REML of the stacked model. Coefficients come from the
solver's stable factorization. `metadata.selection` stores the criterion
value, `λ` and its status, `sigma2`, and for the stacked model the total
effective degrees of freedom and `Mₚ`, as well as the initial value,
tolerances, iterations and convergence. The optimizer uses the defaults above
(`tol=1e-8`, `maxiter=200`, `rank_tol=eps(T)^(3/4)`, no bounds).
[`REML`](@ref) is a selector type parallel to [`GCV`](@ref), passed through the
same `lambda` keyword and shown in the same place. `lambda` remains required,
and the candidate-grid `GCV` selector and its results are unchanged.

```@example penalized
grid = collect(range(0.0, 1.0; length=25))
curves = GridFunctionalData([sin(2pi * s) + i * s + 0.1 * cos(13i * s) for i in 1:4, s in grid], grid)
reml = smooth_curves(curves, PenalizedSpline(BSplineBasis(4, (0.0, 1.0), 8); lambda=REML()))
reml.metadata.selection
```
