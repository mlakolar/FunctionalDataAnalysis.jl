# Functional principal component analysis

`StatsAPI.fit(FPCA, data)` estimates univariate functional principal components.
This milestone implements the dense route for fully observed common-grid data.
With `method=:auto`, regular data take this route. Irregular or masked data
require the PACE route, which is available in task 2. A direct `method=:pace`
request currently gives the same informative error.

```@example dense_fpca
using FunctionalDataAnalysis, Random, LinearAlgebra
import StatsAPI

grid = collect(range(0, 1; length=51))
simulation = simulate_kl(Xoshiro(42), 200;
    mean=t -> 1 + t,
    eigenfunctions=fourier_eigenfunctions((0, 1), 3),
    eigenvalues=[2.0, 0.5, 0.1], noise_sd=0,
    sampling=RegularSampling(grid))
model = StatsAPI.fit(FPCA, simulation.data; ncomponents=FVE(0.99))
(ncomponents(model), eigvals(model), fve(model))
```

The fit retains the sample mean, unprojected sample covariance, training
scores, eigenfunctions, and the component-selection record. Dense covariance
uses the denominator `n - 1`. Scores use the stored trapezoidal quadrature
weights. Their sample covariance has the selected eigenvalues on its diagonal.
The `noise_variance` and `score_covariances` accessors return `nothing` for a
dense fit.

```@example dense_fpca
projected = StatsAPI.predict(model, simulation.data)
restored = reconstruct(model)
(size(projected), size(restored.values), model.selection)
```

New dense curves must use exactly the training grid. Reconstruction evaluates
the training mean and the piecewise-linear eigenfunctions on the requested
`grid`, which must lie inside the training domain. The default reconstruction
uses the training scores and preserves subject IDs. Other score matrices do
not inherit training IDs.

`scores(model)` returns the fit's own score matrix. Mutating it changes the fit
and subsequent reconstruction. Use `copy(scores(model))` for independent storage;
reconstructing from that copy does not inherit training IDs.

Prediction applies the training presmoothing settings. When spline GCV selected
a penalty, prediction uses that selected penalty for new curves.

## Defaults and recorded choices

| Choice | Default and source |
|:--|:--|
| Route | `method=:auto`: fully observed common-grid data use dense FPCA; the PACE route is deferred to task 2. |
| Component selection | `ncomponents=FVE(0.99)`, following fdapace's `FVEthreshold`. An integer requests a fixed number. |
| Maximum components | `max(1, min(length(grid)-2, nobs(data)-2))`, matching fdapace 0.6.0's dense cross-sectional default (`SetOptions.R`). There is no cap of 20 on the dense route. An explicit `maxK` may be as large as `min(length(grid), nobs(data)-1)`, the sample covariance rank bound. |
| Mean and covariance | Cross-sectional sample mean and covariance with denominator `n-1`, matching fdapace's dense cross-sectional route. |
| Quadrature | Stored trapezoidal weights, including nonuniform grids, as used throughout this package. This differs from fdapace's rectangular eigenvalue scaling. |
| Positive spectrum | `eigen_tol=1e-10`: retain eigenvalues strictly larger than this relative tolerance times the largest absolute eigenvalue. This is an explicit numerical convention. |
| Sign | Align each component with the unweighted work-grid mean inner product, following fdapace. If its magnitude is at most `sign_tol * norm(phi) * norm(mean)`, with `sign_tol=1e-12`, make the largest absolute entry positive, following FunctionalGraphicalModels.jl. The relative tolerance is an explicit numerical convention. |
| Presmoothing | `presmooth=nothing`, under the dense noise-free approximation of Ramsay and Silverman (2005), chapter 8. |

All tuning settings are stored on the fit and displayed by `show`. The
`selection` record stores the rule, FVE threshold when applicable, `maxK`,
chosen component count, and rule scores. `all_eigenvalues` and `fve(model)`
refer to every retained positive eigenvalue up to `maxK`, before component
selection. Negative eigenvalues are removed before selection. Their count,
absolute total, and share of the absolute spectrum are recorded in `dropped`.
Values at or below the positive relative tolerance are excluded from the FVE
denominator.

For noisy dense observations, pass a `LocalPolynomial` or `PenalizedSpline`
object as `presmooth`, or use `method=:pace` when that route is available.
Presmoothing calls `smooth_curves` on the data grid. The fit retains the
smoothing settings and any tuning-selection metadata.

The implementation resolves these details left unspecified by the milestone:

- A zero retained positive rank raises `ArgumentError`; `FPCBasis` requires at
  least one function. The default cap is clamped to one for two curves or two
  grid points, following fdapace.
- The dense mean is stored as a one-curve `GridFunctionalData`; the unprojected
  covariance is stored as a matrix. Presmoothing metadata retains the selected
  GCV penalty, which is fixed for prediction.
- Sign ties use the first largest absolute entry. The numerical tolerances
  are the adjustable package choices listed above.
- Reconstruction preserves IDs when the score object is the training score
  matrix. A copied or different score matrix has no inferred IDs.

## Basis and numerical conventions

`eigenfunctions(model)` is an `FPCBasis`. Its columns are orthonormal under
the stored work-grid quadrature weights. Evaluation between work-grid points
uses linear interpolation. Evaluation outside the domain throws.
`gram(basis)` instead integrates the products of those piecewise-linear
interpolants exactly. It is generally close to, but not exactly, the identity.
Derivative evaluation and positive-order roughness penalties are unsupported.

The weighted eigenproblem is
`sqrt(W) * covariance * sqrt(W)`, where `W=Diagonal(weights)`. Dividing its
eigenvectors by `sqrt.(weights)` gives work-grid eigenfunctions. See
[Comparison with fdapace](fdapace.md) for source-level details and the
explicit compatibility convention used by the R golden tests.
Spectrum sums and sign comparisons are evaluated after rescaling to avoid
overflow while preserving the stated FVE and relative sign rules.

The dense covariance denominator differs from the `n` denominator currently
used by FunctionalGraphicalModels.jl. Milestone 4 must account for this change
when migrating that package's FPCA implementation.

PACE conditional-expectation scores and AIC/BIC selection arrive in task 2.
Multivariate and pooled FPCA and cross-covariances arrive in milestone 4.
Smoothed FPCA with a roughness penalty, FACE, and fdapace's conditional-score
`rho` regularization are not yet implemented. Presmoothing individual curves
does not implement a roughness penalty on the FPCA eigenfunctions.

```@docs
FPCA
FPCBasis
FVE
AIC
BIC
eigenfunctions
LinearAlgebra.eigvals(::FPCA)
scores
score_covariances
fve
noise_variance
StatsAPI.predict(::FPCA, ::GridFunctionalData)
reconstruct
```
