# Functional principal component analysis

Prefer `project_scores(model, data)` for score matrices and
`predict_scores(model, data)` for a named-field `ScorePrediction` result.
Legacy `StatsAPI.predict` retains the route-dependent shapes described below
with deprecation warnings; see the [prediction contract](interface.md#Explicit-score-prediction).

`StatsAPI.fit(FPCA, data)` estimates univariate functional principal components.
With `method=:auto`, fully observed common-grid data use the dense route.
Irregular and masked grid data use PACE, including designs classified as
`:dense_irregular`. `method=:pace` also permits conditional-expectation scores
for complete grid data. `method=:dense` requires complete common-grid data.
Use `method=:face` explicitly for fast covariance smoothing on long common
grids, including grids with missing observations, or for sparse curve-specific
times with the sparse FACE estimator.

For `GridFunctionalData`, automatic routing depends on its observed-mask
field at runtime, so inference includes both dense and PACE result types.
Compiled callers that request a fixed `method=:dense`, `:pace`, or `:face` can
infer that route's concrete result type. No-keyword irregular-data fits and
the bandwidth candidate helper retain their inference checks.

```@example dense_fpca
using FunctionalDataAnalysis, Random, LinearAlgebra
import StatsAPI

grid = collect(range(0, 1; length=51))
simulation = simulate_kl(Xoshiro(42), 200;
    mean=t -> 1 + t,
    eigenfunctions=fourier_eigenfunctions((0, 1), 3),
    eigenvalues=[2.0, 0.5, 0.1], noise_sd=0,
    sampling=RegularSampling(grid))
model = StatsAPI.fit(FPCA, simulation.data)
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

## Choosing a route

The [scaling benchmark](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/main/benchmark/scaling)
compares dense FPCA with dense FACE on long grids, and PACE with sparse FACE
on sparsely observed curves: three Fourier components, two noise levels and
200 replicates per cell, one CPU per fit on USC Discovery (AMD EPYC 7513,
Julia 1.13.2). Results are in
[`benchmark/scaling/results`](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/main/benchmark/scaling/results).
Mean time per fit:

| n | m | Dense FPCA | Dense FACE |
|---:|---:|---:|---:|
| 100 | 100 | 0.0014 s | 0.0043 s |
| 100 | 500 | 0.024 s | 0.0084 s |
| 1000 | 500 | 0.054 s | 0.044 s |
| 100 | 2000 | 0.97 s | 0.042 s |
| 1000 | 2000 | 1.26 s | 0.35 s |
| 100 | 5000 | 25 s | 0.12 s |
| 1000 | 5000 | 27 s | 1.2 s |

Dense FACE is faster from about 500 grid points and 20–200 times faster at
5000; dense FPCA is faster at 100 points. At m=5000 dense FPCA also needs
about twice the memory (2.7 GB against 1.4 GB peak per process). The two
routes estimate the leading eigenvalues equally well. FACE's eigenfunctions
are as accurate or slightly more accurate, and its reconstructions are far
closer to the noise-free curves, because dense FPCA does not smooth and
`FVE(0.99)` then keeps tens to hundreds of noise components. Neither route
failed.

On sparse curves with 2–10 points each, sparse FACE took 0.007 s (n=100)
and 0.07 s (n=1000) per fit, against 0.63 s and 6.3 s for PACE. At n=1000
its eigenvalue error was less than half of PACE's (0.064 against 0.148,
relative) with similar eigenfunction error, and it overestimated σ²
slightly more (bias 0.045 against 0.035, true σ² 0.0875 or 0.35). At n=100
the two routes had similar eigenvalue and eigenfunction errors, but sparse
FACE's σ² bias was larger (0.16 against 0.085 at the low noise level).
Without a floor on the estimated noise variance, both routes sometimes
failed with low noise and n=100, because the estimate was truncated to zero
and the observation covariance became singular (sparse FACE 9 of 200
replicates, PACE 4 of 200), and PACE gave a few extreme reconstructions.
With the estimated-noise floor described below, every replicate of every
cell fits, the largest reconstruction error in the sparse cells is 0.94
(against 1.8e5 before), and eigenvalue and eigenfunction errors are
unchanged; dense FACE results are identical, because the floor never binds
there.

The implementation has the following covariance-estimation and
eigenanalysis costs for fixed tuning-search sizes:

| Route | Main time cost | Covariance storage |
|:--|:--|:--|
| Dense FPCA | `O(n*m^2 + m^3)` | `O(m^2)` |
| Dense FACE | `O(m*c^2 + n*m*c + n*c^2 + c^3)` | `O(m*c)` factor |
| PACE, without binning | `O(g*M + g^2*P + g^3)` | `O(g^2)`, plus `O(P)` raw products |
| Sparse FACE | `O(Pplus*q^2 + q^3 + g*c^2)` | `O(g*c)` factor, plus tensor-design working storage |

Here `n` is curve count, `m` is common-grid length, `g` is work-grid length,
`c` is spline-basis size, `N_i` is curve `i`'s observed count,
`M=sum(N_i)`, `P=sum(N_i*(N_i-1))`,
`Pplus=sum(N_i*(N_i+1)/2)`, and `q=c*(c+1)/2+1`.
Sparse FACE's tensor design and subject-CV moments need
`O(Pplus*q + n*q^2)` working storage. Conditional scores add
`O(n*m*K^2 + n*K^3)` time for dense FACE with `K` selected components,
and `O(sum(N_i^3))` for PACE and sparse FACE. The FACE mean smoother uses a
bounded basis; missing-data iterations and optional bandwidth searches add
the costs described below. Automatic PACE binning reduces the covariance
pool when enabled.

## Defaults and where they come from

Let `n = nobs(data)`, `g = length(model.grid)`, and `r` be the range of the
observed times. The defaults below are resolved and recorded when fitting.
The source audit uses fdapace **0.6.0**; its executable defaults differ from
some older help text. See [Comparison with fdapace](fdapace.md) for details.

| Keyword or choice | Default | Source and override |
|:--|:--|:--|
| `method` | `:auto`: complete common-grid data use dense FPCA; irregular or masked data use PACE. | Milestone design dispatch. Use `method=:dense` or `method=:pace`. |
| `grid`, PACE | 51 equally spaced points over the observed time range. | fdapace [`nRegGrid=51` and work-grid construction](https://github.com/cran/fdapace/blob/0.6.0/R/FPCA.R), with the count set in [`SetOptions`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass a strictly increasing `grid` covering all observations. |
| `grid`, dense | The data grid and its stored quadrature weights. | Preserve the data representation. A new grid requires an explicit data conversion before fitting. |
| `ncomponents` | `FVE(0.99)`. | fdapace [`FVEthreshold`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass `FVE(threshold)` or a fixed integer; PACE also accepts `AIC()` and `BIC()` from [Yao, Müller, and Wang (2005), section 2.5](https://doi.org/10.1198/016214504000001745). |
| `max_k`, PACE | `max(1, min(20, g-2, n-2))`. | fdapace [`SetOptions`, smoothing route](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass an integer no larger than `g`. |
| `max_k`, dense | `max(1, min(g-2, n-2))`, without a cap of 20. | fdapace [`SetOptions`, cross-sectional route](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). An explicit cap may be as large as `min(g, n-1)`, the sample covariance rank bound. |
| `kernel`, PACE | `KernelSmoothing.GaussianKernel()`. | fdapace [`SetOptions`, non-dense data](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass another KernelSmoothing kernel. Unlike compact kernels, the Gaussian has no empty windows from finite support. Direct `SmoothedMean` and `SmoothedCovariance` calls retain their Epanechnikov default. |
| `weighting`, PACE | `:obs`, equal weight per observation. | fdapace [default smoothing weighting](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass `:subj` for equal curve weights. Neither weighting scheme dominates in all designs; see [Zhang and Wang (2016)](https://doi.org/10.1214/16-AOS1446). |
| `mean_bandwidth`, PACE | `0.05r`. | fdapace [`methodBwMu="Default"`, `userBwMu`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass a number, `CurveCV`, or a supported KernelSmoothing regression bandwidth selector. |
| `covariance_bandwidth`, PACE | `0.10r`. | fdapace [`methodBwCov="Default"`, `userBwCov`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass a number or `CurveCV` with explicit candidates. |
| `diagonal_bandwidth`, PACE | The selected covariance bandwidth. | The milestone 2 smoother default, following fdapace's diagonal-variance smoothing. Pass a number, `CurveCV`, or a supported univariate selector. |
| Noise estimation, PACE | Rotated diagonal over the full observed domain, inherited from `SmoothedCovariance`. | Issue #17 Discovery benchmark and the 25 October 2026 default decision. The rotated fit follows fdapace 0.6.0 `PC_CovE`; the full interval matches its internal formal default, while `SetOptions` uses the middle half. To change either choice, supply a `SmoothedCovariance` fitted with explicit `noise_method` / `noise_interval`, or supply `noise_variance`. |
| Estimated-noise floor, PACE and both FACE routes | `max(raw_noise_variance, 0.025V)`, where `V` is the maximum fitted pointwise variance. | Package stability rule, with the scale and constant justified under [Estimated noise and its positive floor](@ref fpca-noise-floor). Deliberately replaces face 0.1-8's absolute `1e-6` floor, fdapace 0.6.0's negative-only replacement, and refund 0.1-40's zero truncation. Explicit `noise_variance`, including zero, is unchanged. |
| `binning`, PACE | `:auto`, the audited fdapace `useBinnedData="AUTO"` bin-count rule. | fdapace [`GetBinNum`](https://github.com/cran/fdapace/blob/0.6.0/R/GetBinNum.R). Pass `nothing` to disable snapping or an integer of at least two to set its grid size. The package retains milestone 2's weighted covariance-pair aggregation. |
| Dense mean and covariance | Cross-sectional sample mean and covariance with denominator `n-1`. | fdapace [`GetCovDense`](https://github.com/cran/fdapace/blob/0.6.0/R/GetCovDense.R). |
| Quadrature | Stored trapezoidal weights, including nonuniform grids. | Package inner-product convention. This differs from fdapace's rectangular eigenvalue scaling. |
| `eigen_tol` | `1e-10`: retain eigenvalues strictly above this relative tolerance times the largest absolute eigenvalue. | Adjustable package numerical convention. |
| `sign_tol` | `1e-12` for a negligible mean overlap. | Adjustable package numerical convention. Signs first align with the unweighted work-grid mean inner product, following fdapace. With negligible overlap, the largest absolute entry is positive, following FunctionalGraphicalModels.jl; the first entry breaks ties. |
| `presmooth`, dense | `nothing`. | Dense noise-free approximation in [Ramsay and Silverman (2005), chapter 8](https://doi.org/10.1007/b98888). Pass a `LocalPolynomial` or `PenalizedSpline` object, or use `method=:pace`. |

All tuning settings are stored on the fit and displayed by `show`. The
`selection` record stores the rule, FVE threshold when applicable, `max_k`,
chosen component count, and rule scores. `all_eigenvalues` and `fve(model)`
refer to every retained positive eigenvalue up to `max_k`, before component
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

## FACE for long common grids

FACE (Xiao, Zipunnikov, Ruppert and Crainiceanu, 2016,
[doi:10.1007/s11222-014-9485-x](https://doi.org/10.1007/s11222-014-9485-x))
smooths covariance through a small P-spline basis. It suits densely sampled
EEG, imaging and wearable trajectories when the ordinary grid-square
covariance is too large. It is an explicit route; `:auto` retains its
dense/PACE dispatch. Sparse curve-specific times can opt into sparse FACE,
described below.

```julia
model = StatsAPI.fit(FPCA, data; method=:face)
fixed = StatsAPI.fit(FPCA, data; method=:face, ncomponents=4,
    face=(; knots=35, p=3, m=2, lambda=2.5))
prediction = predict(model, new_grid_data)  # (scores, covariances)
recovered = reconstruct(model)
intervals = fitted_bands(model; curves=[1, 3])
```

The `face` named tuple groups settings specific to this route. Marginal
multivariate FPCA accepts the same keywords. For joint MFPCA, pass
`univariate=(; method=:face, face=(; lambda=2.5))`. Each node retains its own
fit. Pooled multivariate FACE is unsupported.

### Defaults and refund source audit

The audited sources are CRAN **refund 0.1-40**:
[`fpca.face.R`](https://github.com/cran/refund/blob/0.1-40/R/fpca.face.R) and
[`pspline.setting.R`](https://github.com/cran/refund/blob/0.1-40/R/pspline.setting.R).
Here `J` is grid length, `n` is subject count, and `c=knots+p`.

| Setting | Default and source |
|:--|:--|
| `face.knots` | `35`, from `fpca.face`. The source counts **equal intervals**, with 34 strictly interior knots, despite calling this the number of knots. |
| `face.p`, `face.m` | Cubic degree `p=3`, second coefficient differences `m=2`, from `fpca.face` / `pspline.setting`. |
| `face.lambda` | `nothing`: closed-form pooled GCV, from `fpca.face`. A supplied finite nonnegative value fixes smoothing. |
| `face.loglambda` | 100 equally spaced values in `[-20,20]`; candidate lambda is `exp(loglambda)`, from `search.length`, `lower`, and `upper` with `search.grid=TRUE`. First minimum wins. |
| `face.alpha` | `1`, the GCV trace multiplier in `fpca.face`. |
| `face.center` | `true`, from `fpca.face`. False uses a zero mean. |
| `face.mean_smoother` | `nothing`: the existing `PenalizedSpline` with a clamped cubic basis of `min(38,J)` functions, integrated squared second derivative, and `GCV(exp.(range(-20,20; length=100))*domain_width^3)`. Package choice described below. |
| `ncomponents` | `FVE(0.99)`, from refund's `pve=.99`. Fixed K also works; excess requests throw instead of refund's warning and truncation. AIC/BIC throw. |
| `maxK` | `min(n,c)`, the source's maximum spectrum size. The shared package relative `eigen_tol=1e-10` removes numerical zeros. |
| Missing data | At most 100 passes; stop when the Frobenius relative change of smoothed coordinates is at most 0.02, from `fpca.face`. |
| Signs and quadrature | Shared package mean-alignment sign rule (`sign_tol=1e-12`) and stored trapezoidal weights, including nonuniform grids. |

For interval count `k`, the simple knot sequence is
`a + (b-a)*(-p:k+p)/k`. It extends `p` intervals beyond both endpoints;
endpoints are not clamped. The design uses the existing `BSplineBasis`
wrapper and BSplineKit backend with this unaugmented sequence. The internal
P-spline building block penalizes `D_m' D_m`, where `D_m` takes coefficient
differences. This is distinct from `PenalizedSpline`'s integrated derivative
penalty. It is not exported. Require `knots+p < J`. Refund adds `1e-6` to all
design Gram eigenvalues if the minimum is at most `1e-7`. FDA throws in that
case, asking for fewer knots or a smaller degree, so no ridge is added silently.

Write `A=B(B'B)^(-1/2)U`, where `U` diagonalizes the transformed difference
penalty with eigenvalues `s`. For centered completed observations `Y`, let
`Z=A'Y'` and `h_j=1/(1+lambda*s_j)`. The GCV curve is

```math
\frac{\sum_j \|Z_{j\cdot}\|^2(1-h_j)^2-\|Z\|_F^2+\|Y\|_F^2}
     {(1-\alpha\sum_j h_j/[J(1-f_{\rm miss})])^2}.
```

The curve is unnormalized, matching refund. FDA retains all candidates and
scores even when lambda is fixed. With `H=Diagonal(h)Z`, the covariance is
`C=A*H*H'*A'/n`, with denominator **n**, rather than dense FPCA's `n-1`.
Fitting and scoring do not form `C`. For fixed basis size the main work is
`O(J*c^2 + n*J*c + n*c^2 + c^3)`, plus the bounded-dimension mean smoother
and any missing-curve initialization. For selected rank `K`, complete matching
grids share one posterior factorization per fit or prediction batch, adding
`O(J*K^2 + K^3 + n*J*K + n*K^2)`. Masked predictions retain a separate
`O(J_i*K^2 + K^3)` factorization per curve with observed count `J_i`.
Each missing-data pass repeats the
data transformation and small eigenanalysis. The fit stores `C` as a
`J` by `min(n,c)` factor obtained by QR of `H'`, retaining its full surface
independently of component selection. The projected covariance factor has
one column per retained positive component up to `maxK`.
`covariance_data` materializes a copied matrix only when requested.
These are algorithmic costs, not measured benchmark results.

### Mean, missing observations, and recorded choices

Refund smooths column means with `stats::smooth.spline(all.knots=TRUE)`.
FDA uses the existing cubic `PenalizedSpline` smoother with a bounded basis
to preserve scaling on long grids and share the smoothing implementation.
The basis dimension, penalty, selected lambda and full GCV curve are stored
in `model.settings.mean`. To choose another mean smoother, set
`face=(; mean_smoother=PenalizedSpline(...))`. A supplied PACE-style
`SmoothedMean` or `(; grid, values)` is copied, retained in
`model.settings.mean.input`, and evaluated on the data grid.
Do not combine a supplied mean with a mean smoother or `center=false`.

Missing values initialize curve by curve with the default reduced knot count
and spar GCV search of R **stats 4.5.1** `smooth.spline`. The implementation
uses the same integrated-penalty fitting and GCV algebra as `smooth_curves`,
preparing each curve's design and matrix products once for the scalar search
and final fit. Its internal initialization basis penalty
preserves `stats/src/sgram.f`'s legacy `0.3330` coefficient in place of exact
`1/3`; ordinary spline fits retain exact integrated penalties. Outside a
curve's observed time range, refund initializes with that curve's observed
centered mean. Incomplete curves need at least four observations; fully
missing curves throw. Estimating a column mean needs an observed value at
every grid point; a supplied mean can cover an entirely missing column.
Strictly increasing times are used as supplied; FDA does not apply R's
tolerance-based grouping of nearby times. Singular spline fits throw rather
than replacing a curve with a constant after a failed smoothing solve.

Only the first imputation pass selects lambda. Each pass updates missing
entries using refund's Euclidean component selection and shrinkage, keeping
observed entries unchanged. As in the source, the stopping check compares
smoothed coordinates, and missing entries are updated once more before
returning. The final covariance comes from that pass's smoothed coordinates;
final scores use the updated data. This ordering is retained for compatibility.
The shared FVE selector uses `>=` throughout FDA, whereas refund uses `>`;
`FVE(1)` is therefore supported. Numerical zeros are removed using `eigen_tol`.

`model.settings` records knots, degree, penalty order, candidates, GCV scores,
lambda and its source, alpha, centering, mean tuning, quadrature and score
rules. `settings.imputation` records iterations, convergence, relative
changes, tolerance, cap, initial curve smoothing choices, missing fraction,
and completed centered observations. A fit reaching 100 passes remains
available with `converged=false`. `show` displays these settings.
Original observed data and masks remain in `training_data` for plotting.

### L² conversion, noise and shrinkage

Refund returns Euclidean-unit eigenvectors `Phi_R`, eigenvalues `E_R` of
`C`, and scores `S_R` in those coordinates. Its internal `Sigma=E_R/J`;
returned `evalues=J*Sigma`, `efunctions=Phi_R`, and `scores=S_R`.
With rectangular weights `Delta`, the L² conversion is

```math
\lambda=\Delta E_R,\qquad
\phi=\Phi_R/\sqrt\Delta,\qquad \xi=\sqrt\Delta S_R.
```

FDA's stored trapezoidal weights are not constant, even on an equally spaced
grid. They require the weighted operator `sqrt(W)*C*sqrt(W)`, rather than
separate column rescaling. A thin QR decomposition of `sqrt(W)*A` transfers
this calculation to a `c` by `c` matrix. Nonuniform grids use the same rule.
Component selection, eigenvalues, functions and scores use this L² system.
Thus selected fitted curves can differ from refund's after weighting changes
the eigenspaces. The reference script independently converts the covariance
and solves for the corresponding scores, and also retains refund's original
outputs for a rectangular compatibility calculation.

Noise remains on the **observation** scale:

```math
\widehat\sigma^2_{\rm raw}=
\operatorname{mean}_{\rm observed}(Y^2)-\operatorname{tr}(C)/J,\qquad
\widehat\sigma^2=\max\{\widehat\sigma^2_{\rm raw},0.025V\},\qquad
V=\max_j C_{jj}.
```

The trace and variance scale include the full smoothed covariance before
truncation and are recomputed at each completion iteration. The raw
estimate matches refund's `mean(Y[observed]^2)-sum(Sigma)`; summing the L²
eigenvalues without converting the integration scale would give the wrong noise
variance. refund 0.1-40 [`fpca.face`](https://github.com/cran/refund/blob/0.1-40/R/fpca.face.R)
truncates at zero. The package instead uses the same positive floor as sparse
FACE and PACE, so prediction from a rank-deficient subset of grid observations
retains positive observation covariance. The floor and its application are
recorded as described under [Estimated noise and its positive floor](@ref fpca-noise-floor).
For selected functions `Phi` and diagonal L² eigenvalues `Lambda`, scores are

```math
\widehat\xi_i=(\Phi^\mathsf{T}\Phi+\widehat\sigma^2\Lambda^{-1})^{-1}
\Phi^\mathsf{T}(y_i-\mu).
```

The low-rank extension of the existing conditional-expectation solver computes
this expression without a grid-square solve. With Euclidean-unit functions
it reduces to refund's shrinkage by `Sigma/(Sigma+sigma2/J)`. This is the
Gaussian conditional expectation used by PACE, specialized to the selected
FACE model; PACE additionally includes all components up to `maxK` in its
observation-covariance denominator. New masked grid curves use their observed
points in this posterior solve. Training predictions reproduce stored scores
from completed data. Complete curves on matching grids reuse one precision
factorization and posterior covariance per batch; at zero noise they reuse
one pivoted QR with the same rank checks. Prediction requires exactly the
training grid.

### Pointwise variance: correction to refund

Under `xi ~ N(0,Lambda)` and independent white noise, posterior curve covariance
for complete observations is

```math
V=\widehat\sigma^2\Phi
(\Phi^\mathsf{T}\Phi+\widehat\sigma^2\Lambda^{-1})^{-1}\Phi^\mathsf{T}.
```

`fitted_bands` uses `diag(V)`, consistent with shrinkage scores and the
package's conditional-variance contract. In refund 0.1-40, the `var=TRUE`
code instead inserts its internal `Diagonal(Sigma)` in place of the inverse
score covariance. In its returned coordinates `Sigma=E_R/J`; this also
introduces a scale inconsistency. For the default reference fixture's first
grid point, the weighted posterior variance is approximately **0.01874790**.
Replacing `Lambda^{-1}` by `Lambda` in the same weighted formula gives
**0.01971711**. The test also preserves and verifies refund's actual unconverted
`diag.var` (approximately **0.01906664** at this point), and shows that it
differs from the corresponding posterior.

The mean, covariance, selected rank and noise are treated as fixed. Training
missing values are treated as completed observations, matching refund's
design; imputation uncertainty is excluded. `coverage=:interval` supports
custom evaluation grids and curve subsets through the existing band layer.
Simultaneous FACE bands are unsupported; `coverage=:band` throws.

Reference values are recreated with `Rscript test/reference/refund_m6.R`,
using refund 0.1-40 and R 4.5.1. The committed fixtures have 40 curves on
50 points, with default lambda selection, fixed lambda 2.5, missing values,
and an independent calculation of FDA's mean-smoother departure. Tests compare
means, spectra, eigenfunctions, scores, noise, lambda, GCV curves, completed
data and five fitted curves at relative error below `1e-8`. The fixtures retain
all eigenfunctions and scores. Completed data are recovered from the shared
input, the mean and the reference imputed entries; fitted curves cover all four
missing-data patterns and one complete curve. Variance is checked
against an independent posterior calculation, with the refund discrepancy
kept visible. CI reads the CSVs and does not require R.

### Sparse FACE

For irregular observations, `method=:face` implements Xiao, Li, Checkley and
Crainiceanu (2018), [doi:10.1007/s11222-017-9744-8](https://doi.org/10.1007/s11222-017-9744-8).
`:auto` continues to choose PACE. The reference implementation is **face 0.1-8**:
[`face.sparse.inner.r`](https://github.com/cran/face/blob/0.1-8/R/face.sparse.inner.r),
[`pspline.r`](https://github.com/cran/face/blob/0.1-8/R/pspline.r),
`pspline.setting`, `raw.construct`, `select.knots`, and `predict.face.sparse`.
Sparse FACE fits a symmetric tensor-product P-spline covariance and a
constant measurement-error variance jointly from within-subject residual
products, including the diagonal. This is a different estimator from the
sandwich smoother for common grids.

```julia
cd4 = load_dataset(:cd4).data
sparse_face = StatsAPI.fit(FPCA, cd4; method=:face, ncomponents=2)
pace = StatsAPI.fit(FPCA, cd4; method=:pace, grid=sparse_face.grid,
    ncomponents=2)
# Both use the same grid and quadrature; align signs when comparing functions.
face_phi = basismatrix(eigenfunctions(sparse_face), sparse_face.grid)
pace_phi = basismatrix(eigenfunctions(pace), sparse_face.grid)
for k in axes(pace_phi, 2)
    if sum(sparse_face.weights .* face_phi[:, k] .* pace_phi[:, k]) < 0
        pace_phi[:, k] .*= -1
    end
end
# With Plots.jl loaded, compare the two aligned eigenfunction pairs.
plot(sparse_face.grid, face_phi; label=["FACE 1" "FACE 2"])
plot!(pace.grid, pace_phi; label=["PACE 1" "PACE 2"], linestyle=:dash)
prediction = predict(sparse_face, new_irregular_data) # scores, covariances
intervals = fitted_bands(sparse_face; curves=[1, 3])
```

| Setting | Sparse default and source |
|:--|:--|
| `face.knots` | `7`, from `face.sparse`. The source counts equal intervals, so there are six strictly interior knots and `c=knots+p=10` basis functions. |
| Knot range | Observed minimum and maximum, expanded at each end by `0.001` times the observed width; simple knots extend `p` intervals further, from `select.knots`. |
| `face.p`, `face.m` | Cubic degree `3`, second coefficient differences `2`. The requested difference order is honored for other `m`; the reference hardcodes second differences in its covariance penalty. |
| `face.lambda` | `nothing`: minimize the subject-deletion criterion below; a finite nonnegative number fixes smoothing. |
| `face.loglambda` | 14 equally spaced values from `-3` to `10`, from `lower`, `upper`, `search.length`. Candidates are `exp.(loglambda)`; the first minimum wins. The full criterion is recorded even at fixed lambda (R returns zero placeholders then). |
| `face.center` | `true`, from `face.sparse`; false uses a zero mean. A supplied mean requires true. |
| `face.lambda_mean` | `nothing`, from `face.sparse` / `pspline`; a finite nonnegative number fixes mean smoothing. It cannot be combined with a supplied mean or `center=false`. |
| `face.mean_loglambda` | 100 equally spaced values from `-20` to `20`, from `pspline`; independent of the covariance grid. |
| Mean basis and weights | Same interval count and expanded range as covariance; always cubic with second differences, as `face.sparse` calls `pspline` without passing `p,m`. Each observation receives weight `1/N_i`. |
| `grid` | 100 equally spaced points across the observed range, from `argvals.new`. An explicit finite increasing grid must span all observation times, remain in the observed range, and have full spline rank. |
| `ncomponents`, `maxK` | `FVE(0.99)`; cap `c`, the spline dimension. Shared fixed-K and numerical cutoff rules apply. The package selects from functional eigenvalues; R's source uses coefficient-space eigenvalues for its component count. AIC/BIC throw. |
| Ridges | Covariance always adds `1e-6*maximum(eigvals(X'X))` to the Gram spectrum. Mean adds `1e-6` if its smallest weighted Gram eigenvalue is at most `1e-7`. Both reference rules and realized values are stored. |
| Noise | Joint fitted diagonal coefficient, used as `max(raw_noise_variance, 0.025V)`. `V` is the full positive spline covariance's maximum variance over the work grid and all training observation times. This deliberately replaces face 0.1-8's absolute `1e-6` floor; see [Estimated noise and its positive floor](@ref fpca-noise-floor). The raw estimate, threshold, scale, relative factor and application flag are stored and shown. |
| Signs and quadrature | The same mean-alignment sign rule and stored trapezoidal weights as other routes; `eigen_tol=1e-10`, `sign_tol=1e-12`. |

A supplied `mean=(; grid, values)` or value-valued `SmoothedMean` uses the
existing PACE validation and evaluation rules. Inputs and fitted settings are
copied. A custom `mean_loglambda` requires an estimated mean. The mean's
selected lambda, candidate grid, criterion, coefficients,
knots, weighting, and ridge are stored under `settings.mean`. The covariance
has its own criterion, knots, lambda, realized ridge, and raw noise estimate.
`show` reports these choices. `two_step` is unsupported and throws: its second
fit needs subject-specific covariance-of-products weights, beyond the
identity-weighted default implemented here.

#### What the reference calls `gcv`

Write `y_i` for a subject's vector of upper-triangular residual products,
including its squared observations. Let `X` be the symmetric tensor-product
design augmented by the diagonal indicator. The covariance penalty is

```math
Q=\operatorname{blockdiag}\{G^\top(I_c\otimes D_m^\top D_m)G,0\},
```

where `G` duplicates the lower triangle into a symmetric coefficient matrix.
The reference diagonalizes the ridge-adjusted Gram matrix and penalty to
obtain coordinates `A=X A_0` and penalty spectrum `s`. For
`d_j=(1+lambda*s_j)^(-1)`, its smoothing matrix is
`S=A*Diagonal(d)*A'`. With `r=y-S*y`, its actual criterion is

```math
\|r\|^2-\|y\|^2+2\sum_i r_i^\top S_{ii}r_i.
```

This is the first-order approximation to leave-one-subject-out squared
prediction error: replace `(I-S_ii)^(-1)` by `I+S_ii` in the deleted-subject
residual and retain the first-order term. It is neither exact deletion nor
the usual trace-denominator GCV. The omitted `y'y` is constant across lambda.
The mean uses the same criterion with `S=A*Diagonal(d)*A'*W`, where each
subject's weights are `1/N_i`; squared prediction errors remain unweighted.
The covariance uses identity weights on the unique within-subject pairs.

The fitted symmetric coefficient matrix is projected to its positive
semidefinite part before weighted functional eigenanalysis. The known
`m*(m+1)/2+1` null penalty directions are set to zero numerically; their count
is stored as `penalty_nullity`. Normal matrices and their decompositions use
at least Float64 precision before converting coordinates to the input floating
type; `normal_matrix_type` records that precision. Only spline-sized matrices
are diagonalized. The covariance is stored as a grid-by-rank factor;
plotting or requesting `Matrix(model.covariance)` materializes the surface.
For `P=sum_i N_i*(N_i+1)/2` and `q=c*(c+1)/2+1`, the tensor design uses `O(P*q)`
storage; normal-matrix formation takes `O(P*q^2)` and decomposition takes
`O(q^3)`. This avoids a grid-square covariance during fitting, but subjects
with many observations still contribute quadratically many raw products.

#### Scale, scores, prediction, and uncertainty

R reports functional eigenvalues using the average-grid metric `B'B/J`.
With constant quadrature weight `w`, convert its eigenvalues by `J*w`,
eigenfunctions by `1/sqrt(J*w)`, and scores by `sqrt(J*w)`. With trapezoidal
or nonuniform weights, diagonalize the small weighted operator instead; a
single scalar conversion is insufficient. Noise stays on the measurement
scale. The tests independently perform this conversion from R's fitted
coefficient covariance.

At arbitrary training or prediction observation times, the fitted spline
mean, eigenfunctions, and full positive coefficient covariance are evaluated
exactly. For selected eigenfunctions `Phi_i`, selected eigenvalues `Lambda`,
and full spline covariance `C_i`, the shared conditional-expectation solver
computes

```math
\widehat\xi_i=\Lambda\Phi_i^\top(C_i+\sigma^2 I)^{-1}(y_i-\mu_i),\qquad
\Omega_i=\Lambda-\Lambda\Phi_i^\top(C_i+\sigma^2 I)^{-1}\Phi_i\Lambda.
```

This is the coefficient BLUP in `predict.face.sparse`, expressed in the
weighted functional eigenbasis. All positive spline directions enter `C_i`,
including directions beyond `maxK` or the selected K. New grid or irregular
data must lie within the work grid. The score solve uses the fitted positive
noise variance unchanged; prediction applies no additional ridge or floor.
`predict` returns the usual `(scores,covariances)` tuple. Reconstruction and
pointwise `fitted_bands` use the selected components, with variance
`diag(Phi*Omega_i*Phi')`; omitted components and estimation uncertainty are
excluded. R's direct curve predictions and standard errors use its full
coefficient covariance, irrespective of `pve`; they coincide with this
projection when all positive components are retained. Reconstruction on a
new grid uses the shared piecewise-linear `FPCBasis` interface.

Reference values are recreated with `Rscript test/reference/face_m6.R` using
face 0.1-8, refund 0.1-40, mgcv 1.9-3, Matrix 1.7-3, and R 4.5.1. Fixtures
cover the milestone-3 sparse inputs and all 366 CD4 subjects (1,888 observed
counts), exported with the same native months, positional IDs, and missing-cell
convention as `load_dataset(:cd4)`. Default selections and fixed covariance
lambda `2.5` / mean lambda `0.75` are checked. Mean curves, weighted spectra,
eigenfunctions, noise, tuning, both criterion curves, scores, their
covariances, and selected fitted values and variances are compared at relative
error below `1e-8`. R's full coefficient predictions and standard errors are
also checked after retaining all positive components. The CD4 fixture keeps
score and posterior outputs for a small set of subjects, including a singleton;
it does not store full covariance surfaces. CI consumes CSVs without R.

## PACE for sparse observations

PACE estimates a pooled mean and latent covariance, then predicts each
subject's principal component scores by conditional expectation. It reuses
[`SmoothedMean`](@ref) and [`SmoothedCovariance`](@ref). A no-keyword fit uses
the work grid, Gaussian kernel, and fractional bandwidths listed above.
The Gaussian avoids empty windows caused by compact support, but it cannot
prevent ill-conditioned local fits on extremely coarse or clustered data.
Such fits raise an informative error suggesting a larger bandwidth. Increase
`mean_bandwidth` or `covariance_bandwidth` explicitly when needed; the fit
does not enlarge them automatically.

```@example pace_fpca
using FunctionalDataAnalysis, Random, LinearAlgebra
import StatsAPI

simulation = simulate_kl(Xoshiro(43), 120;
    mean=t -> 1 + t,
    eigenfunctions=legendre_eigenfunctions((0, 1), 2),
    eigenvalues=[1.5, 0.4], noise_sd=0.3,
    sampling=SparseSampling((0, 1), 3:8))
model = StatsAPI.fit(FPCA, simulation.data)
(ncomponents(model), size(scores(model)), size(score_covariances(model)))
```

This fit estimates the noise variance with `SmoothedCovariance`'s rotated,
full-interval default. Singular rotated windows throw an informative error;
increase the covariance bandwidth or supply a `SmoothedCovariance` fitted
with `noise_method=:surface_diagonal` for a small design. Pass
`noise_variance=0.09` to use the simulation's known value. Smoother settings,
selection metadata, the origin of every supplied or estimated input, and the
noise truncation details are retained on the fit. Inputs are copied.

### [Estimated noise and its positive floor](@id fpca-noise-floor)

PACE and both FACE routes use
`noise_variance(model) = max(raw_noise_variance, 0.025V)` when noise is estimated.
The scale `V` is a maximum fitted pointwise signal variance:

- PACE uses the diagonal of the full positive projected covariance retained
  up to `maxK`, evaluated on the work grid. Its piecewise-linear representation
  also bounds the fitted variance between grid points.
- Sparse FACE uses the full positive spline covariance on the work grid and
  at every training observation time, including components beyond the selected K.
- Dense FACE uses the full smoothed covariance on the common grid at each
  completion iteration, before selecting components.

This scale has the same squared observation units as the raw estimate.
Multiplying observations and their fitted mean by `a` multiplies both the
raw variance and the floor by `a²`, preserving whether the floor binds.
Changing time units leaves the pointwise covariance scale unchanged. Using
functional eigenvalues without their integration scale would not have this
time-unit invariance.
Explicit `noise_variance` values on the PACE route, including zero, are used
unchanged. Noise inherited from a supplied `SmoothedCovariance` is still
estimated and receives the floor. Standalone `SmoothedCovariance` continues
to truncate negative estimates to zero; the FPCA fit retains its raw estimate
before that truncation.

The factor `0.025` controls statistical amplification as well as numerical
conditioning. Let `K_i` be the positive-semidefinite fitted observation
covariance for a curve with `m_i` observations and diagonal entries at most
`V`. Its largest eigenvalue is at most `m_i*V`, so

```math
\kappa_2(K_i+\sigma^2 I)
\le 1+\frac{m_iV}{\sigma^2}
\le 1+40m_i.
```

For at most ten observations this bound is 401. At a target with fitted
variance at most `V`, the map from the observation residual vector to the
conditional reconstructed deviation has Euclidean gain at most `sqrt(10)`.
To see this, factor the full fitted covariance as `BB'`. The singular values
of `B'*(BB'+sigma2*I)^(-1)` are `s/(s²+sigma2)`, bounded by
`1/(2sqrt(sigma2))`; multiplying by a target factor row with norm at most
`sqrt(V)` gives the bound. Restricting the reconstructed numerator to selected
components preserves it. These bounds motivate the constant as a safeguard
against amplification of noisy observations; they do not bound reconstruction
error against an unknown true curve. Sparse FACE evaluates splines exactly,
so an unseen prediction time can have fitted variance above the stored `V`.
The same stored noise is used there, but the uniform `sqrt(10)` gain bound
then does not apply. Prediction does not silently change the fit's floor.

The source audit found that face 0.1-8
[`face.sparse.inner`](https://github.com/cran/face/blob/0.1-8/R/face.sparse.inner.r#L185-L189)
resets estimates at or below `1e-6` to `1e-6`, and
[`predict.face.sparse`](https://github.com/cran/face/blob/0.1-8/R/predict.face.sparse.r#L33-L36)
uses `max(sigma2, 1e-6)`.
fdapace 0.6.0
[`GetSmoothedCovarSurface`](https://github.com/cran/fdapace/blob/0.6.0/R/GetSmoothedCovarSurface.R#L81-L109)
replaces strictly negative estimates by `1e-6` but leaves zero and tiny
positive estimates unchanged. Dense FACE's reference, refund 0.1-40
[`fpca.face`](https://github.com/cran/refund/blob/0.1-40/R/fpca.face.R), truncates
at zero. The relative floor deliberately departs from these safeguards:
an absolute threshold changes its relative strength with the observation units
and can leave conditional estimates highly sensitive to noise.
fdapace's optional [`rho` path](https://github.com/cran/fdapace/blob/0.6.0/R/FPCA.R#L219-L233)
is prior art for replacing the estimated variance when scoring needs more
regularization. Its [`GetRho`](https://github.com/cran/fdapace/blob/0.6.0/R/GetRho.R)
selects a value using reconstructed total variance. The package uses the fixed
relative rule above and does not reproduce that selector or its default
`methodRho="vanilla"` behavior.

The fit records `settings.raw_noise_variance`, `settings.noise_floor`,
`settings.noise_floor_relative`, `settings.noise_floor_scale`,
`settings.noise_floor_scale_source`, `settings.noise_floor_applied`,
`settings.noise_variance_source`, and `settings.noise_truncation`; `show`
displays them. For estimated noise, `noise_floor=0.025V`,
`noise_floor_relative=0.025`, `noise_floor_scale=V`, and
`noise_floor_scale_source=:maximum_fitted_pointwise_variance`.
`noise_floor_applied` tests whether the raw estimate is strictly below the
threshold, and `noise_truncation=:positive_floor`. Equality leaves the estimate
unchanged and the flag false. For supplied noise, the threshold and three scale
fields are `nothing`, the flag is false, and `noise_truncation=:none`;
`raw_noise_variance` records the supplied value.
The existing [`noise_variance`](@ref) accessor
returns the variance actually used in component selection, scores,
uncertainties, reconstruction, and prediction. Prediction and derivative
scores reuse this stored value without adding a second floor.

Let ``\widetilde G`` denote the positive-semidefinite projection of the work-grid
surface over **all** positive components retained up to `max_k`. Negative
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
Cholesky factorization, using the recorded positive floor when noise is
estimated. No additional ridge or pseudoinverse is added. A
non-positive-definite system, including one from explicit zero noise, raises
an error naming the curve and suggesting an explicit `noise_variance`.

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
that mean instead. Bandwidth defaults are used only for the respective
estimated inputs. Supplying `mean_bandwidth` together with a supplied `mean`
raises `ArgumentError`. Supplying `covariance_bandwidth` or
`diagonal_bandwidth` together with a supplied `covariance` also raises
`ArgumentError`. Reused inputs retain their fitted tuning values; unused
bandwidth overrides are rejected.

Binning only affects covariance estimation. With a supplied covariance,
`binning` must remain `:auto`, and the fit records `settings.binning=nothing`
and `settings.binning_rule=:not_applicable`. When both mean and covariance
are supplied, `kernel` must remain `nothing` and `weighting` must remain
`:obs`; overrides raise `ArgumentError`. Their active settings are recorded
as `nothing` because no smoother runs. If either input is estimated,
`kernel` and `weighting` still tune that smoother. Supplied fitted inputs
keep their original settings under `settings.mean_input` and
`settings.covariance_input`.

### Overriding defaults and selecting bandwidths

Every automatic choice can be replaced explicitly. For example:

```julia
import KernelSmoothing

model = StatsAPI.fit(FPCA, simulation.data;
    method=:pace,
    grid=collect(range(extrema(simulation.data.times)...; length=61)),
    ncomponents=2, max_k=6,
    kernel=KernelSmoothing.GaussianKernel(), weighting=:subj,
    mean_bandwidth=0.15, covariance_bandwidth=0.25,
    diagonal_bandwidth=0.20, binning=nothing)
```

For an irregular object, use `extrema(simulation.data.times)` to obtain the
observed endpoints: concatenated times are ordered within curves, rather than
across curves. A work grid may cover a wider interval, but all observations
must remain inside it. Fractional bandwidth defaults always use the observed
time range, not the work-grid range.

[`pace_bandwidth_candidates`](@ref) returns the ten geometric mean and
covariance bandwidths from fdapace's `GCVLwls1D1` and `GCVLwls2DV2` candidate
grids. The candidate values use the Gaussian convention of the PACE default.
Pass `kernel=KernelSmoothing.EpanechnikovKernel()` to generate the compact-kernel
variant, and use the same kernel for `fit`. The helper does not run fdapace's
GCV, choose a bandwidth, or fit a smoother. Use whole-curve CV to select from
these candidates:

```julia
candidates = pace_bandwidth_candidates(simulation.data)
model = StatsAPI.fit(FPCA, simulation.data;
    mean_bandwidth=CurveCV(candidates.mean; folds=10, rng=Xoshiro(44)),
    covariance_bandwidth=CurveCV(candidates.covariance; folds=10, rng=Xoshiro(45)))
```

Pass `grid=work_grid` to both the helper and `fit` when using a custom work
grid. `CurveCV` retains the candidate grid, errors, and fold assignments.
Mean selectors and covariance selectors pass to the existing pooled
smoothers unchanged. The 2D covariance smoother accepts a number or explicit
`CurveCV` candidates; KernelSmoothing bandwidth selectors are univariate.
Covariance CV holds the supplied fitted mean fixed across folds. It holds out
whole curves, whereas fdapace's covariance CV holds out raw covariance pairs.
These procedures therefore need not select the same bandwidth. The helper
requires enough unique observation times and time pairs for the audited
nearest-neighbor starts. It throws if they are unavailable. Equal candidate
endpoints can yield repeated values; use `unique(candidates.mean)` or
`unique(candidates.covariance)` before `CurveCV` in that case.

### Cost and automatic binning

Let `N_i` be the number of observations on curve `i`, `M=sum(N_i)`,
`P=sum(N_i*(N_i-1))`, and `g` the work-grid size. Fixed-bandwidth pooled mean
smoothing costs `O(g*M)`; unbinned covariance smoothing costs `O(g^2*P)` after
forming `P` off-diagonal products. The weighted eigenstep costs `O(g^3)` and
the per-curve Cholesky score solves cost `O(sum(N_i^3))`. The defaults perform
no bandwidth search.

For `binning=:auto`, take `m=median(N_i)` for sparse designs and `m=max(N_i)`
otherwise, as in fdapace's audited rule:

1. With `m<=20`, do not bin.
2. With `m>400`, initially choose 400 bins.
3. With `n>5000`, compute `mstar=max(20,400+19*(5000-n)/2250)` and choose
   `ceil(mstar)` bins if `mstar<m`; otherwise do not bin.
4. If none of these rules selects a count, do not bin.

The candidate helper and bin-count decision use fdapace's `IsRegular`
classification. Let `u` be the number of unique pooled observation times and
`coverage=M/(n*u)`. Full coverage is `Dense` with at least eight unique times
and `Sparse` with fewer than eight. Incomplete coverage above `0.8` is
`DenseWithMV`; other designs are `Sparse`. This classifier controls only the
mean-candidate start and bin-count statistic. It does not change FPCA's
`:auto` route: complete `GridFunctionalData` use dense FPCA, and irregular or
masked data use PACE.

The resolved bin count and the decision's design, count statistic, summary
count, and number of curves are stored and shown. Binning snaps covariance-pool
times and combines within-curve cell products with their original weights;
it does not discard observations. It uses the existing covariance smoother,
rather than fdapace's separate preprocessing of each curve. The pooled mean
and work-grid length are unchanged by this aggregation. With `b` bins,
`Q=sum(min(N_i*(N_i-1),b^2))` bounds its occupied cells. Exact duplicate pair
coordinates are also combined across subjects for a fixed-bandwidth fit,
without moving times or combining the original diagonal observations. Thus
`O(g^2*Q)` smoothing and `O(Q)` pair storage remain upper bounds, after `O(P)` work to form
the products. Most sparse curves have at most 20 observations, so automatic
binning does not turn on, even when `n` is large.

The candidate helper's covariance nearest-neighbor calculation itself takes
`O(U^2)` time and `O(U)` storage for `U` unique time pairs, including diagonal
pairs. It is optional and is not run by a default fit.

A 2D `CurveCV` with `H` candidate bandwidths and `F` folds splits subjects
before combining exact duplicate coordinates separately in training and held-out
pools. Per candidate and fold, smoothing costs the product of those pools'
distinct-coordinate counts. Without duplicates, the bounds remain
`O(H*P^2 + H*F*P)` without binning, or `O(H*Q^2 + H*F*Q)` with binning. The held-out
products are evaluated at their pool coordinates (snapped when binning is
enabled), rather than only at the work grid. Ten candidate bandwidths can therefore cost much more than a
fixed-bandwidth fit. Reduce the candidate set or set an explicit bin count
when that expense is inappropriate. See [Smoothing](smoothing.md) for the
weight-preserving binning and CV contracts.

### PACE component selection

Missing `mean` and `covariance` inputs estimate the pooled smoothers, following
[Yao, Müller, and Wang (2005)](https://doi.org/10.1198/016214504000001745).
Missing `noise_variance` uses `SmoothedCovariance`'s raw estimate with the
[positive floor](@ref fpca-noise-floor);
a supplied surface requires an explicit number. The score solve uses
this stored noise variance and must admit Cholesky factorization.
Eigenanalysis, signs, and numerical tolerances are shared with the dense route.

For `AIC()` and `BIC()`, each candidate ``k`` uses its own ``k``-component
covariance plus ``\sigma^2 I`` in the Gaussian likelihood. The criterion is
minus twice the log likelihood plus ``2k`` or ``k\log(n)``, respectively.
`selection.scores` stores the full criterion curve for `k=1:max_k`;
`selection.candidates` records those counts. When `max_k` exceeds the retained
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

### Default recovery checks and their limits

A fixed-seed `simulate_kl` check uses 400 sparse curves, 2–6 observations per
curve, Fourier constant/sine components with eigenvalues `[2.0,0.7]`, and
noise variance `1.0`. A no-keyword fit has weighted mean L2 error below
`0.30`, sign-invariant weighted L2 error below `0.35` for each of the first
two eigenfunctions, and noise relative error below `0.40`. An 80-curve,
25-point regular-grid check compares dense and PACE eigenvalues within `15%`
and absolute weighted eigenfunction correlations above `0.98`.

These thresholds describe finite simulations. They are not universal
recovery guarantees. A historical lower-noise diagnostic with the same sparse
signal and true variance `0.25` estimated `0.56615` under the former
surface-diagonal, middle-half default. That value does not describe the current
rotated, full-interval default, chosen using the [Discovery results](@ref sigma2-discovery-results).
The fixed `0.10r` covariance bandwidth can leave appreciable smoothing bias.
Inspect the fitted noise estimate and use an explicit bandwidth selector or
known `noise_variance` when appropriate.

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

The dense covariance denominator differs from the `n` denominator in the
FunctionalGraphicalModels.jl reference. The multivariate comparison below
accounts for this convention explicitly.

## Multivariate FPCA

`StatsAPI.fit(FPCA, mfd::MultivariateFunctionalData)` returns a
[`MultivariateFPCA`](@ref). The default `basis=:marginal` fits each node with
the univariate method above. Nodes may use different grids or mix complete
grid data with sparse irregular data. Each node retains its own route,
defaults, tuning record, and conditional score covariances, when applicable.
This operation performs separate marginal decompositions. The joint
multivariate decomposition of Happ and Greven is a later milestone 4 task.

```julia
model = StatsAPI.fit(FPCA, mfd; basis=:marginal, ncomponents=2)
node_model = component(model, 1)
node_scores = scores(model, 1)
counts = ncomponents(model) # one selected count per node
```

For marginal fitting, every univariate keyword accepts either a shared value
or a length-`p` vector containing one value per node. A numeric work `grid`
is one shared grid; an outer vector of grids or `nothing` entries specifies
one choice per node. To supply
different means or covariance surfaces, use a vector of the same fitted
objects or named tuples accepted by univariate FPCA. A vector with the wrong
node count throws. For example:

```julia
model = StatsAPI.fit(FPCA, mfd;
    basis=:marginal, ncomponents=[2, 3], max_k=[4, 5],
    grid=[work_grid_node_1, work_grid_node_2])
```

The `basis=:pooled` route estimates one common basis from the average of the
node sample covariance matrices. It follows the partial-separability
construction of [Zapata, Oh, and Petersen
(2022)](https://doi.org/10.1093/biomet/asab046). Every node must be complete
`GridFunctionalData` on exactly the same grid. Masked grids, irregular nodes,
and mismatched grids throw. Pooled PACE is not implemented.

For centered node matrices ``Z_j=X_j-\mathbf 1\widehat\mu_j^\mathsf T``, the
pooled surface and node scores are

```math
\widehat G_{\mathrm{pool}}
    =\frac1p\sum_{j=1}^p\frac{Z_j^\mathsf T Z_j}{n-1},
\qquad
\widehat\xi_{ijm}
    =\sum_g w_g\{X_{ij}(t_g)-\widehat\mu_j(t_g)\}\widehat\phi_m(t_g).
```

The weighted eigenstep and positive-eigenvalue cutoff are the same as for
univariate FPCA. `ncomponents` is an integer or `FVE(threshold)`, defaulting
to `FVE(0.99)`; `max_k` follows
the dense route. Selection uses the pooled spectrum. The sign rule uses
the average of the node mean curves as its common reference and records
`sign_reference=:average_node_mean`. Near-zero overlap uses the same
largest-absolute-entry fallback as univariate FPCA. The pooled tuning
keywords are `ncomponents`, `max_k`, `eigen_tol`, and `sign_tol`;
`method=:auto` or `:dense` is accepted. Pooled presmoothing, custom work
grids, and PACE smoothing options throw because no corresponding pooled
method is implemented. Each node keeps its own sample mean, and
`eigvals(model)[j]` reports that node's score sample variances, with
denominator `n-1`. These node variances need not be
ordered within a node because the shared basis is ordered by the pooled
spectrum. `eigenfunctions(model)` returns the shared `FPCBasis` for every
node. `component(model, j)` throws for a pooled fit because there is no
separately fitted univariate FPCA at that node.

```julia
pooled = StatsAPI.fit(FPCA, mfd; basis=:pooled, ncomponents=2)
array = score_array(pooled) # subjects × nodes × principal components
```

[`score_array`](@ref) gives FunctionalGraphicalModels.jl's `n × p × M`
convention. Its node order is exactly the order of `component(mfd, j)` and
the second axis of `to_array(mfd)`. It requires equal selected counts across
nodes. Unequal counts throw and suggest an integer `ncomponents`; scores are
never padded or truncated. `scores(model)` always returns a vector of node
score matrices. These matrices are the fit's own storage; copy them before
independent mutation. `score_covariances(model)` returns a vector whose entries
are conditional covariance arrays for PACE nodes and `nothing` for dense or
pooled nodes.

`StatsAPI.predict(model, newmfd)` returns one prediction per node using the
training means and bases. Dense predictions are score matrices; PACE
predictions are the univariate `(scores, covariances)` named tuples.
`reconstruct(model)` returns `MultivariateFunctionalData` and preserves
training subject IDs. Its `grid` keyword accepts one shared grid or one grid
per node; without an override, each training grid is used. Reconstruction
from other score matrices follows the univariate ID convention above.

The FunctionalGraphicalModels.jl reference computes dense covariance and
score variances with denominator `n`. FDA uses `n-1`, so matching eigenvalues
are the reference values multiplied by `n/(n-1)`. Dense scores retain their
scale. Reference dense eigenfunction signs must be aligned before comparing
scores. See [Milestone 4 comparison report](m4_report.md) for the frozen
reference, sparse-route differences and defaults.

Cross-covariances are described on the cross-structure page. FACE and
roughness-penalized basis FPCA are described on this page. fdapace's
conditional-score `rho` selector remains unimplemented. Presmoothing
individual curves does not implement a roughness penalty on FPCA eigenfunctions.

```@docs
FPCA
FPCBasis
pace_bandwidth_candidates
FVE
AIC
BIC
eigenfunctions
LinearAlgebra.eigvals(::FPCA)
LinearAlgebra.eigvals(::MultivariateFPCA)
scores
score_covariances
fve
noise_variance
StatsAPI.predict(::FPCA, ::GridFunctionalData)
StatsAPI.predict(::MultivariateFPCA{T}, ::MultivariateFunctionalData{S}) where {T,S}
reconstruct
MultivariateFPCA
score_array
```

## Derivatives

`derivative(model)` estimates velocity curves from a fitted FPCA. It returns
an `FPCADerivative` with a mean derivative, derivative functions on the stored
work grid, scores, bandwidths, kernel, and all choices. `reconstruct(d)` returns
functional data; `curve_data(d)` and `plot(d)` draw the reconstructed derivative
curves. `mean_data(d)` returns the mean derivative. These operations do not
refit the model.

```julia
velocity_fit = derivative(model; mean_bandwidth=0.08,
    covariance_bandwidth=0.10)
velocity_curves = reconstruct(velocity_fit)
velocity_mean = mean_data(velocity_fit)
# With Plots loaded:
# plot(velocity_fit; curves=1:3)
```

The default `method=:FPC` differentiates the stored mean and each selected
eigenfunction by local polynomial fitting, then combines them with the original
scores. Orders one and two are supported. Derivative functions need not be
orthogonal, and the retained eigenvalues describe the original curves. The
default degree equals the derivative order, matching fdapace 0.6.0 `FPCAder`;
this differs from the `ν+1` default for standalone polynomial derivatives.
fdapace labels second derivatives experimental. Accuracy still depends on the
work-grid resolution, bandwidth, and smoothness of the fitted functions.

`method=:DPC` (Dai, Müller and Tao, 2018) instead estimates principal components
of the derivative process, with its own eigenvalues and scores. It supports
first derivatives and defaults to degree two and `FVE(0.9999)`. Pass an integer
or `FVE` via `ncomponents` to change that choice. It applies the polynomial
derivative operator in both arguments of the covariance and reuses the usual
weighted eigenstep. Dense fits use the sample covariance and project the
correspondingly differentiated centered training curves. PACE and FACE use
the full projected covariance and conditional-expectation scores from the
original observations. The curve/derivative-score cross-covariance supplies the CE
numerator. The parent fit's stored noise variance, including any estimated-noise
floor, is reused without an additional ridge. Conditional score covariances omit
uncertainty from fitting mean, covariance, and noise.

For FACE, DPC uses all retained components up to `maxK` in this denominator,
including when the original fit selects fewer components. Masked training
curves use their observed entries. This observation-space calculation forms
grid-square derivative covariance matrices; it does not retain FACE's fitting
and prediction costs on long grids.

DPC here uses stored quadrature weights, including nonuniform grids, and the
same differentiated covariance for its derivative eigenvalues and score prior.
fdapace's native DPC mixes finite differences, polynomial smoothing,
rectangular quadrature, and an unsmoothed score-prior spectrum. Its DPC mean is
also re-estimated from raw observations, whereas this package differentiates
the fitted mean supplied by the user. The
[fdapace audit](fdapace.md#Derivative-source-audit) describes the resulting
reference quantities and their validation. `FPC1` is not implemented.

For either method the default kernel is Gaussian. With derivative order `ν`
and domain width `r`, both default bandwidths are `ν*0.10r` when the fitted
noise variance divided by the sum of selected original eigenvalues is at least
0.01, and `ν*0.05r` otherwise (including dense fits without a noise estimate).
These rules follow fdapace 0.6.0 `SetDerOptions`. Supply a common `bandwidth`,
or both `mean_bandwidth` and `covariance_bandwidth`. A partial pair or mixed
forms throw. There is no bandwidth search. `show` reports resolved bandwidths,
rule, degree, kernel, component selection and numerical tolerances.

For `MultivariateFPCA`, differentiation runs in node order and returns a
`MultivariateFPCADerivative`. Shared or per-node keyword settings work as for
marginal FPCA; `component(d,j)` retrieves the node result. Reconstruction,
plotting, and mean data work nodewise, with shared or per-node grids. Both marginal
and pooled fits support the two methods. Pooled FPC differentiates the shared
basis and retains node scores. Pooled DPC uses each node's sample covariance
and copied original training observations; its derivative components are
node-specific.

Fixed-seed Fourier checks use `simulate_kl` with dense or sparse sampling. They
check relative weighted derivative-function error and relative derivative-curve
error below 0.12. Sparse checks supply the known mean, covariance and noise to
isolate differentiation and CE reconstruction from covariance-smoothing bias.
These are finite simulation checks, not general recovery guarantees.

```@docs
FPCADerivative
MultivariateFPCADerivative
```

## Smoothed FPCA

Basis data use `fit(FPCA, fd::BasisFunctionalData; lambda=0, penalty=nothing)`.
This is the roughness-penalized method of
[Silverman (1996)](https://doi.org/10.1214/aos/1033066196), implemented in
fda **6.3.0** `pca.fd`. Eigenfunctions remain exact linear combinations of the
input basis. Prediction requires that same basis. Reconstruction evaluates
the original basis at the requested points; it does not interpolate a sampled
harmonic. The usual `scores`, `eigvals`, `fve`, mean/mode/scree/covariance
coordinates, fitted paths, and plot recipes accept this FPCA result.

```@example smoothed_fpca
using FunctionalDataAnalysis, LinearAlgebra
import StatsAPI
height = load_dataset(:growth; sample=true).data
b = BSplineBasis(4, domain(height), 5)
height_basis = to_basis(height, b)
smoothed = StatsAPI.fit(FPCA, height_basis; lambda=0.1, ncomponents=2)
(size(scores(smoothed)), eigvals(smoothed), smoothed.settings.lambda)
```

`lambda=LambdaCV([0.0, 0.1, 1.0])` opts into leave-one-curve-out selection.
For example, `fit(FPCA, height_basis; lambda=LambdaCV([0.0, 0.1, 1.0]),
ncomponents=2)` compares two leading subspaces. fda does not provide an
automatic selector for this function, so the default remains zero.
FPCA with `LambdaCV` requires a positive integer `ncomponents`; the default
`FVE(0.99)` remains available with a fixed lambda.

### Reference, scale, and defaults

The reference sources are fda 6.3.0
[`pca.fd.R`](https://github.com/cran/fda/blob/6.3.0/R/pca.fd.R) and
[`fdPar.R`](https://github.com/cran/fda/blob/6.3.0/R/fdPar.R).
For centered coefficient rows C, Gram matrix G, and roughness matrix P, they
solve

```math
G(C^\top C/n)G h_j = \nu_j (G+\lambda P)h_j,
\qquad h_j^\top(G+\lambda P)h_k=\delta_{jk}.
```

The source's `inprod.R` uses Romberg integration with a hard-coded relative
stopping rule `EPS=1e-4`. Its B-spline fast path checks `is.integer` on the
default Lfd objects and does not trigger for these calls. Even Fourier
cross-products and harmonic-acceleration penalties can therefore have errors
larger than `1e-8`. The package uses its exact basis Gram/penalty matrices.
The R generator keeps the complete `pca.fd` body and supplies exact
matching-basis inner products in a private function environment. For the
Fourier differential operator it supplies independently integrated products
with `rel.tol=1e-12`, `abs.tol=1e-14`. No installed R namespace is modified.
These are the documented exact-inner-product reference quantities compared
at `1e-8`; the CSVs also retain native `pca.fd` eigenvalues as
`native_eigenvalues`. Native `cca.fd` uses `eval.penalty` for its Gram
matrices and needs no such change for the second-derivative comparisons.

The denominator is **n**, including when centering is disabled. This differs
from dense-grid FPCA's `n-1`. We keep R's eigenvalues ν and their variance
proportions. Write q_j² = h_jᵀ G h_j. The package harmonic is h_j/q_j, and
its score is R's score divided by q_j. Consequently the score second moment
is ν_j/q_j², rather than ν_j, for a positive penalty. With centering on, this
is the score variance using denominator n. Harmonics have unit exact L² norm;
positive-lambda harmonics generally are **not L²-orthogonal**. Reconstruction
uses these L²-normalized harmonics and their inner-product scores, as for the
other FPCA routes. It is not an orthogonal subspace projection at positive
lambda. `settings.original_harmonic_norms` records all q_j.

| Choice | Default and source |
|:--|:--|
| `lambda` | `0`, fda `fdPar`; a fixed finite nonnegative value or explicit `LambdaCV` candidates |
| `penalty` | `nothing` resolves to derivative order `max(0, b.order-2)` for B-splines, including order 2 for cubic splines; harmonic acceleration for Fourier; order 0 for monomials. These follow `fdPar`. A derivative order or a symmetric positive-semidefinite matrix overrides this. |
| Fourier operator | `D³ + (2π/domain_width)² D`; `penalty(b, :harmonic_acceleration)` exposes its matrix. It leaves the constant and first sine/cosine pair unpenalized. |
| `center` | `true`, fda `pca.fd`'s `centerfns`; false stores a zero mean for consistent prediction and reconstruction. R still returns the empirical mean when centering is off. |
| `ncomponents` | `FVE(0.99)`, retained for consistency with the package. R uses `nharm=2`; a fixed integer reproduces that count. AIC/BIC require PACE. |
| `maxK` | `min(nbasis, n-1)` with centering, `min(nbasis,n)` without it; the available rank bound, with no grid-route cap |
| `eigen_tol`, `sign_tol` | `1e-10`, `1e-12`, the shared FPCA cutoff and sign conventions |
| `grid` | 201 equally spaced display points, as in `curve_data`; a supplied grid must be strictly increasing inside the basis domain |
| Signs | Shared mean alignment on the display grid, with largest-absolute-entry fallback; R instead orients the whitened eigenvectors by their coefficient sums |
| Noise and score uncertainty | `nothing`, since the basis-data method specifies no measurement-noise model |

Exact basis Gram matrices determine inner products and normalization; stored
trapezoidal weights accompany the display grid only. `eigenfunctions(f)`
implements `basismatrix`, including exact derivatives, `gram`, `penalty`,
`domain`, and `nbasis`. The original basis and eigenfunction coefficients are
also available as `.basis` and `.coefs`. The plot-data layer samples those
functions. `fitted_curve_data` reports input basis curves evaluated on the
requested grid. `covariance_data(f)` samples the original coefficient
covariance; `surface=:fitted` samples the retained sum
Σ_j ν_j φ_j(s)φ_j(t), in the documented normalized scale.
`fitted_bands` is unavailable without a noise model.
The grid-based `derivative(f::FPCA)` fit is unsupported for this route;
`DerivativeBasis(eigenfunctions(f), order)` instead gives the exact derivative
basis, and `evaluate(f.mean, points; derivative=order)` differentiates the mean.

`LambdaCV` follows Ramsay and Silverman (2005), §9.3.3, equation (9.3).
It centers once using the overall mean, then fits each leave-one-curve-out
sample without further centering. For each held-out curve it computes exact
L² **orthogonal** projection residuals for subspaces m=1,…,K and sums their
squared norms. Projection solves the subspace Gram system, since smoothed
harmonics need not be L²-orthogonal. A positive integer `ncomponents=K` is
required so that every fold and candidate uses the same number of subspaces.
An FVE rule would compare sums with different numbers of terms and is rejected.
The smallest total wins. Candidates remain in user order, and ties select the
first. All scores, the criterion, centering rule,
fold count, and selected index are in `settings.lambda_selection`. An invalid
candidate/fold throws with its context rather than being silently dropped.

### Reference values

Run `Rscript test/reference/fda_m6.R` with **fda 6.3.0** to recreate the small
CSV fixtures. Headers record the R and fda versions. The inputs are five-basis
coefficient matrices for the 54 girls in `growth` (height and interval growth
rate) and the 35 Canadian weather stations (temperature and log10
precipitation). Both datasets use cubic B-splines and Fourier bases. Lambdas
are `(0, 0.1, 10)` for growth and `(0, 1e4, 1e6)` for weather. They are fixed
values, with no optimizer comparison. Julia tests compare eigenvalues,
harmonics on 101 points, scores, variance proportions, canonical correlations,
weight functions, and variates at relative tolerance `1e-8`.

For a nonzero lower domain endpoint, R's Fourier basis uses sin(ωt), while the
package uses sin(ω(t-a)). The generator rotates coefficient pairs by ωa
before exporting inputs. This orthogonal transformation preserves norms and
penalties. FPCA comparisons apply the q_j normalization and package sign rule.
CCA comparisons keep R's coefficient norm and apply paired signs in the
rotated basis. The generator also evaluates both CV criteria independently
through R fits. CI consumes these fixtures without running R.

```@docs
LambdaCV
StatsAPI.fit(::Type{FPCA}, ::BasisFunctionalData)
```
