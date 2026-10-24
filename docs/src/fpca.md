# Functional principal component analysis

`StatsAPI.fit(FPCA, data)` estimates univariate functional principal components.
With `method=:auto`, fully observed common-grid data use the dense route.
Irregular and masked grid data use PACE, including designs classified as
`:dense_irregular`. `method=:pace` also permits conditional-expectation scores
for complete grid data. `method=:dense` requires complete common-grid data.

For `GridFunctionalData`, automatic routing depends on its observed-mask
field at runtime, so inference includes both dense and PACE result types.
Compiled callers that request a fixed `method=:dense` or `method=:pace` can
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
| `maxK`, PACE | `max(1, min(20, g-2, n-2))`. | fdapace [`SetOptions`, smoothing route](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass an integer no larger than `g`. |
| `maxK`, dense | `max(1, min(g-2, n-2))`, without a cap of 20. | fdapace [`SetOptions`, cross-sectional route](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). An explicit cap may be as large as `min(g, n-1)`, the sample covariance rank bound. |
| `kernel`, PACE | `KernelSmoothing.GaussianKernel()`. | fdapace [`SetOptions`, non-dense data](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass another KernelSmoothing kernel. Unlike compact kernels, the Gaussian has no empty windows from finite support. Direct `SmoothedMean` and `SmoothedCovariance` calls retain their Epanechnikov default. |
| `weighting`, PACE | `:obs`, equal weight per observation. | fdapace [default smoothing weighting](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass `:subj` for equal curve weights. Neither weighting scheme dominates in all designs; see [Zhang and Wang (2016)](https://doi.org/10.1214/16-AOS1446). |
| `mean_bandwidth`, PACE | `0.05r`. | fdapace [`methodBwMu="Default"`, `userBwMu`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass a number, `CurveCV`, or a supported KernelSmoothing regression bandwidth selector. |
| `covariance_bandwidth`, PACE | `0.10r`. | fdapace [`methodBwCov="Default"`, `userBwCov`](https://github.com/cran/fdapace/blob/0.6.0/R/SetOptions.R). Pass a number or `CurveCV` with explicit candidates. |
| `diagonal_bandwidth`, PACE | The selected covariance bandwidth. | The milestone 2 smoother default, following fdapace's diagonal-variance smoothing. Pass a number, `CurveCV`, or a supported univariate selector. |
| `binning`, PACE | `:auto`, the audited fdapace `useBinnedData="AUTO"` bin-count rule. | fdapace [`GetBinNum`](https://github.com/cran/fdapace/blob/0.6.0/R/GetBinNum.R). Pass `nothing` to disable snapping or an integer of at least two to set its grid size. The package retains milestone 2's weighted covariance-pair aggregation. |
| Dense mean and covariance | Cross-sectional sample mean and covariance with denominator `n-1`. | fdapace [`GetCovDense`](https://github.com/cran/fdapace/blob/0.6.0/R/GetCovDense.R). |
| Quadrature | Stored trapezoidal weights, including nonuniform grids. | Package inner-product convention. This differs from fdapace's rectangular eigenvalue scaling. |
| `eigen_tol` | `1e-10`: retain eigenvalues strictly above this relative tolerance times the largest absolute eigenvalue. | Adjustable package numerical convention. |
| `sign_tol` | `1e-12` for a negligible mean overlap. | Adjustable package numerical convention. Signs first align with the unweighted work-grid mean inner product, following fdapace. With negligible overlap, the largest absolute entry is positive, following FunctionalGraphicalModels.jl; the first entry breaks ties. |
| `presmooth`, dense | `nothing`. | Dense noise-free approximation in [Ramsay and Silverman (2005), chapter 8](https://doi.org/10.1007/b98888). Pass a `LocalPolynomial` or `PenalizedSpline` object, or use `method=:pace`. |

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

This fit estimates the noise variance with `SmoothedCovariance`. Pass
`noise_variance=0.09` to use the simulation's known value. Smoother settings,
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
    ncomponents=2, maxK=6,
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
`Q=sum(min(N_i*(N_i-1),b^2))` bounds its occupied cells. Covariance smoothing
then costs `O(g^2*Q)`, and pair storage costs `O(Q)`, after `O(P)` work to form
the products. Most sparse curves have at most 20 observations, so automatic
binning does not turn on, even when `n` is large.

The candidate helper's covariance nearest-neighbor calculation itself takes
`O(U^2)` time and `O(U)` storage for `U` unique time pairs, including diagonal
pairs. It is optional and is not run by a default fit.

A 2D `CurveCV` with `H` candidate bandwidths and `F` folds costs
`O(H*P^2 + H*F*P)` without binning, or `O(H*Q^2 + H*F*Q)` with binning. The held-out
products are evaluated at their pool coordinates (snapped when binning is
enabled), rather than only at the work grid. Ten candidate bandwidths can therefore cost much more than a
fixed-bandwidth fit. Reduce the candidate set or set an explicit bin count
when that expense is inappropriate. See [Smoothing](smoothing.md) for the
weight-preserving binning and CV contracts.

### PACE component selection

Missing `mean` and `covariance` inputs estimate the pooled smoothers, following
[Yao, Müller, and Wang (2005)](https://doi.org/10.1198/016214504000001745).
Missing `noise_variance` uses `SmoothedCovariance`'s nonnegative estimate;
a supplied surface requires an explicit number. The score solve uses
unregularized conditional expectation and must admit Cholesky factorization.
Eigenanalysis, signs, and numerical tolerances are shared with the dense route.

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

### Default recovery checks and their limits

A fixed-seed `simulate_kl` check uses 400 sparse curves, 2–6 observations per
curve, Fourier constant/sine components with eigenvalues `[2.0,0.7]`, and
noise variance `1.0`. A no-keyword fit has weighted mean L2 error below
`0.30`, sign-invariant weighted L2 error below `0.35` for each of the first
two eigenfunctions, and noise relative error below `0.40`. An 80-curve,
25-point regular-grid check compares dense and PACE eigenvalues within `15%`
and absolute weighted eigenfunction correlations above `0.98`.

These thresholds describe finite simulations. They are not universal
recovery guarantees. In a lower-noise diagnostic with the same sparse signal
and true variance `0.25`, the default fit estimated `0.56615`. The fixed
`0.10r` covariance bandwidth can leave appreciable diagonal smoothing bias.
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
    basis=:marginal, ncomponents=[2, 3], maxK=[4, 5],
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
to `FVE(0.99)`; `maxK` follows
the dense route. Selection uses the pooled spectrum. The sign rule uses
the average of the node mean curves as its common reference and records
`sign_reference=:average_node_mean`. Near-zero overlap uses the same
largest-absolute-entry fallback as univariate FPCA. The pooled tuning
keywords are `ncomponents`, `maxK`, `eigen_tol`, and `sign_tol`;
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
reference, sparse-route differences, defaults, and development checks.

Cross-covariances arrive in a later milestone 4 task.
Smoothed FPCA with a roughness penalty, FACE, and fdapace's conditional-score
`rho` regularization are not yet implemented. Presmoothing individual curves
does not implement a roughness penalty on the FPCA eigenfunctions.

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
