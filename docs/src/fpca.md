# Functional principal component analysis

`StatsAPI.fit(FPCA, data)` estimates univariate functional principal components.
With `method=:auto`, fully observed common-grid data use the dense route.
Irregular and masked grid data use PACE, including designs classified as
`:dense_irregular`. `method=:pace` also permits conditional-expectation scores
for complete grid data. `method=:dense` requires complete common-grid data.

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
| Route | `method=:auto`: complete common-grid data use dense FPCA; irregular or masked data use PACE. This explicit design dispatch follows the milestone specification. |
| Component selection | `ncomponents=FVE(0.99)`, following fdapace's `FVEthreshold`. An integer requests a fixed number. PACE also supports `AIC()` and `BIC()`, following Yao, Müller, and Wang (2005), section 2.5. |
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
object as `presmooth`, or use `method=:pace`.
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

## PACE for sparse observations

PACE estimates a pooled mean and latent covariance, then predicts each
subject's principal component scores by conditional expectation. It reuses
[`SmoothedMean`](@ref) and [`SmoothedCovariance`](@ref). At this stage, supply
a work `grid` and the bandwidth for each smoother that is estimated. Automatic
work-grid and bandwidth choices are deferred to milestone 3's next task.

```@example pace_fpca
using FunctionalDataAnalysis, Random, LinearAlgebra
import StatsAPI

grid = collect(range(0, 1; length=31))
simulation = simulate_kl(Xoshiro(43), 120;
    mean=t -> 1 + t,
    eigenfunctions=legendre_eigenfunctions((0, 1), 2),
    eigenvalues=[1.5, 0.4], noise_sd=0.3,
    sampling=SparseSampling((0, 1), 3:8))
model = StatsAPI.fit(FPCA, simulation.data;
    grid, mean_bandwidth=0.5, covariance_bandwidth=0.6,
    noise_variance=0.09, ncomponents=FVE(0.99), maxK=5)
(ncomponents(model), size(scores(model)), size(score_covariances(model)))
```

The explicit noise variance in this example is the simulation's known value.
Omit it to use the nonnegative `SmoothedCovariance` estimate. Smoother settings,
selection metadata, the origin of every supplied or estimated input, and the
noise truncation details are retained on the fit. Inputs are copied.

Let ``\widetilde G`` denote the positive-semidefinite projection of the work-grid
surface over **all** positive components retained up to `maxK`. Negative
eigenvalues are dropped and reported in `model.dropped`. The selected component
count ``K`` can be smaller than this retained rank. At subject ``i``'s observed
times, let ``\Phi_i`` contain the selected eigenfunctions and
``\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_K)``. The scores and
their conditional covariance are

```math
\begin{aligned}
\Sigma_i &= \widetilde G(t_i,t_i)+\sigma^2 I,\\
\widehat\xi_i &= \Lambda\Phi_i^\mathsf T\Sigma_i^{-1}(y_i-\mu_i),\\
\Omega_i &= \Lambda-\Lambda\Phi_i^\mathsf T\Sigma_i^{-1}\Phi_i\Lambda.
\end{aligned}
```

The denominator ``\Sigma_i`` uses the full retained projected covariance,
including components beyond the selected ``K``. This is the observed-curve
covariance in Yao, Müller, and Wang (2005) and fdapace. The numerator and
``\Omega_i`` use the selected components. Mean and eigenfunction evaluation
are linear; covariance evaluation is bilinear. The systems are solved by
Cholesky factorization. A non-positive-definite system raises an error naming
the curve and suggesting an explicit `noise_variance`. No ridge, noise floor,
or pseudoinverse is added.

`score_covariances(model)` is a `K × K × n` array: slice `[:, :, i]` is
``\Omega_i``. Under the Gaussian model, it measures remaining uncertainty
about the latent scores after observing curve ``i``. It is not the sample
covariance of the predicted scores. Use it when propagating score uncertainty
to reconstructions or downstream inference; sparse-curve scores can have
substantial shrinkage and uncertainty. These covariances condition on the
fitted mean, covariance, and noise variance, so they do not include uncertainty
from estimating those quantities.

```@example pace_fpca
new_scores, new_covariances = StatsAPI.predict(model, simulation.data[1:3])
restored = reconstruct(model)
(size(new_scores), size(new_covariances), size(restored.values))
```

PACE prediction accepts grid or irregular data and returns `(scores,
covariances)`. It uses the training mean, eigenfunctions, projected covariance,
and noise variance. Every observation time must lie inside the work-grid
domain. Prediction on the training data agrees with the stored scores and
conditional covariances. Reconstruction uses the same interface as dense
FPCA and preserves training IDs.

### Supplied means, surfaces, and noise variance

Supply `mean=smoothed_mean` to reuse a `SmoothedMean`, or
`mean=(; grid, values)` for an explicit mean curve. The mean grid must cover
the entire work grid and every observation time. Supply
`covariance=smoothed_covariance` to reuse a `SmoothedCovariance`, or
`covariance=(; grid, surface)` for a finite symmetric covariance surface. Surfaces that are asymmetric only by rounding (relative tolerance `sqrt(eps)`, as R's `isSymmetric` allows) are accepted and replaced by `(S + S')/2`.
The surface grid must cover the work grid. A supplied surface requires an
explicit nonnegative `noise_variance`. Supplied objects are copied with their
settings; they are not refitted.

The fit stores its work-grid mean as one-curve `GridFunctionalData` and its
unprojected work-grid covariance as a matrix. `settings.mean_input` and
`settings.covariance_input` retain copied supplied or estimated inputs;
`settings.full_projected_covariance` retains the surface used for scores and
prediction. Mean, covariance, and noise provenance are recorded separately.
Noise inherited from a `SmoothedCovariance` retains `:estimated` provenance,
including when that covariance object was supplied to the fit.

If the covariance must be estimated, the mean must be estimated in the same
fit or supplied as a `SmoothedMean`. A named-tuple mean cannot be passed to
the existing pooled covariance smoother; supply a covariance surface with
that mean instead. `mean_bandwidth` and `covariance_bandwidth` are required
only for the respective estimated inputs.

### PACE defaults and selection

| Choice | Default and source |
|:--|:--|
| Work grid and bandwidths | Required when needed in this task; no automatic choices are made. |
| Maximum components | `max(1, min(20, length(grid)-2, nobs(data)-2))`, following fdapace 0.6.0's smoothed-route cap in `SetOptions.R`. An explicit PACE `maxK` may be as large as the work-grid length. |
| Mean and covariance inputs | `nothing` estimates the pooled smoothers, following the PACE construction of Yao, Müller, and Wang (2005). |
| Noise variance | `nothing` uses `SmoothedCovariance`'s nonnegative estimate. A supplied surface requires a number. A supplied number overrides the estimate. |
| Kernel | `KernelSmoothing.EpanechnikovKernel()`, the milestone's explicit choice and the existing pooled smoother default. This differs from fdapace's Gaussian sparse default. |
| Weighting | `:obs`, the existing pooled smoother convention; `:subj` gives equal weight to curves in the pooled estimates. |
| Binning | `nothing`, retaining the existing covariance smoother's explicit no-binning default. |
| Eigenanalysis and signs | The same trapezoidal quadrature, relative eigenvalue cutoff, and sign convention as dense FPCA. |
| Score solve | Unregularized conditional expectation, as in Yao, Müller, and Wang (2005); Cholesky factorization must succeed. |

For `AIC()` and `BIC()`, each candidate ``k`` uses its own ``k``-component
covariance plus ``\sigma^2 I`` in the Gaussian likelihood. The criterion is
minus twice the log likelihood plus ``2k`` or ``k\log(n)``, respectively.
`selection.scores` stores the full criterion curve for `k=1:maxK`;
`selection.candidates` records those counts. When `maxK` exceeds the retained
positive rank, the additional components have zero variance, leave the
likelihood unchanged, and increase only the penalty.
Selection stops at the first increase, following fdapace. The chosen count
and `selection.global_minimizer` are both recorded, so a later lower value
remains visible. The Gaussian normalization constant is included; fdapace
omits it, which does not affect either selected count.

The analytic tests check the conditional-expectation formulas, including the
full-covariance denominator. A separate 3000-curve sparse Gaussian simulation
uses known mean, covariance, and noise with 3 to 8 points per curve. Mean
squared score error is compared with the mean trace of ``\Omega_i`` within
10%, and each standardized component error variance is checked within 10%
of one. These are Monte Carlo checks, not coverage guarantees after smoothing.

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
