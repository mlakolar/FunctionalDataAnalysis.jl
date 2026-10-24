# Multivariate FPCA (MFPCA)

`StatsAPI.fit(MFPCA, data; ncomponents=M)` estimates joint principal components
by the two-step construction of [Happ and Greven (2018), Proposition 5](https://doi.org/10.1080/01621459.2016.1273115).
Nodes may have different domains, grids, observation designs, and univariate
component counts. Each node is a univariate functional data object; higher
dimensional domains supported by R MFPCA are outside this package's current API.

```julia
import StatsAPI
using FunctionalDataAnalysis

# data is a MultivariateFunctionalData object with two nodes
model = StatsAPI.fit(MFPCA, data; ncomponents=3,
    univariate=(; ncomponents=[2, 3]), weights=[1.0, 0.5])
joint_scores = scores(model)             # subjects × 3
node_functions = eigenfunctions(model)  # two FPCBasis objects, each with 3 columns
new_scores = StatsAPI.predict(model, newdata)
fitted_curves = reconstruct(model)
```

The univariate first step calls the existing **marginal** FPCA implementation.
Keywords in `univariate` follow its shared/per-node convention. For example,
`univariate=(; method=[:dense, :pace], grid=[nothing, workgrid],
noise_variance=[nothing, 0.05])` requests dense FPCA for the first node and
PACE for the second. Its means, preprocessing, covariance smoothing, rank
choices, and conditional score solver are reused and retained in
`model.univariate`; `component(model, j)` returns the original univariate fit.

## Construction and weights

Let ``\Xi_j`` contain the ``M_j`` selected univariate scores and let ``\Phi_j``
denote their eigenfunctions, orthonormal under the node's stored quadrature.
For positive node weights ``w_j``, form

```math
A = [\sqrt{w_1}\Xi_1\ \cdots\ \sqrt{w_p}\Xi_p],\qquad
A_c = A-\mathbf{1}\bar A,\qquad
Z_w = \frac{A_c^\top A_c}{n-1},\qquad Z_w c_m = \nu_m c_m.
```

The unit eigenvectors are ordered by decreasing eigenvalue. To match R MFPCA
1.3-11's exact route, normalization uses the **uncentered** matrix
``Q=A/\sqrt{n-1}``. Define ``h_m=\|Qc_m\|^{-1}`` and
``b_m=Q^\top Qc_m\,h_m/\sqrt{\nu_m}``. The functions and scores are

```math
\psi_m^{(j)} = w_j^{-1/2}\Phi_j b_m^{(j)},\qquad
\rho_m = Ac_m\sqrt{\nu_m}h_m.
```

When the univariate scores have sample mean zero, ``b_m=c_m`` and
``\sqrt{\nu_m}h_m=1``. These formulas then reduce to Proposition 5:

```math
\psi_m^{(j)} = w_j^{-1/2}\Phi_j c_m^{(j)},\qquad
\rho_{im} = \sum_j \sqrt{w_j}\,\Xi_j[i,:]c_m^{(j)}.
```

In this mean-zero case, the weighted sum of node quadrature inner products
is the identity:
``\sum_j w_j \langle\psi_m^{(j)},\psi_\ell^{(j)}\rangle_j=\delta_{m\ell}``.
Individual node components need not be orthonormal or even nonzero.
Signs make the largest absolute entry of ``c_m`` positive; the first entry
breaks ties. Repeated eigenvalues define an eigenspace and can rotate within it.

The default weights are all one, following R. The paper also discusses
``w_j=1/\int_{\mathcal T_j}\operatorname{Var}(X^{(j)}(t))\,dt`` to balance
nodes of different scales. Supply such weights explicitly. Multiplying all
weights by ``a>0`` multiplies eigenvalues by ``a``, scores by ``\sqrt a``, and
functions by ``1/\sqrt a``; fitted curves are unchanged. Changing a node's
units by ``b`` is canceled by dividing its weight by ``b^2``.

PACE conditional scores need not have sample mean zero. The covariance
eigenproblem centers these scores, matching R's `cov(allScores)`. Its
eigenvalues describe the estimated scores' sample covariance, not the
univariate latent PACE covariance spectrum. R's normalization and final score
projection still use uncentered scores. Consequently, returned scores can
have nonzero means, and the returned functions need not be orthonormal when
these means are nonzero. This reproduces R's exact route rather than imposing
another normalization. The normalization factors and score scales are stored
in `model.settings` and shown with the fit.

Prediction calls univariate `predict` per node and applies the same weights
and coefficients, followed by the stored R score scales. It does not subtract
training or new-data score means. It returns a matrix, including when some
nodes use PACE.
Nodewise conditional covariances remain available from the marginal fit;
joint conditional uncertainty would require cross-node assumptions and is
not estimated here. Reconstruction adds each node's training mean to its
multivariate expansion. It accepts a shared grid or one grid per node and
keeps training IDs only when using the stored training score object.

## Choosing a basis

| Method | Use |
|:--|:--|
| `fit(FPCA, data; basis=:marginal)` | Separate node modes and truncation; scores remain node-specific. Suitable for nodewise analysis and FGM estimators. |
| `fit(FPCA, data; basis=:pooled)` | A common basis on the same complete grid under a shared covariance structure. Scores remain node-specific. |
| `fit(MFPCA, data; ncomponents=M)` | Joint modes that combine cross-node score dependence, with different node domains and designs allowed. Scores have one joint component index. |

## Comparison with MFPCA

The source audit uses CRAN **MFPCA 1.3-11** (published 2025-08-27), available
as a [versioned source archive](https://cran.r-project.org/src/contrib/MFPCA_1.3-11.tar.gz).
Its SHA-256 is
`aef336bd23edd8e9df2741f70d358066410b3217882054771d2cd002f2e65125`.
The audited files are `R/MFPCA_calculation.R` (`MFPCA`, `calcMFPCA`),
`R/univDecomp.R` (`givenBasis`, `fpcaBasis`), `R/univPCA.R` (`PACE`, `.PACE`),
and `R/univariateExpansions.R`. See also the
[maintainer's manual](https://clarahapp.r-universe.dev/MFPCA/doc/manual.html).

| R argument | Default and behavior in 1.3-11 | FDA |
|:--|:--|:--|
| `M` | Required number of multivariate components. Excess above the total univariate count is reduced with a warning. | Required `ncomponents`; excess count or positive-rank requests throw. |
| `weights` | `rep(1, length(mFData))`. Weighted sum of node inner products. | Same all-one default, with finite, strictly positive validation. |
| `uniExpansions` | Required list, one expansion specification per node. | `univariate=(;)` delegates to existing marginal FPCA instead of duplicating R's expansion library. |
| `fit` | `FALSE`; if true, returns mean plus the truncated joint expansion. | Explicit `reconstruct`; fits are not computed eagerly. |
| `approx.eigen` | `FALSE`; `TRUE` uses `irlba` SVD, reverting to exact if `M > min(n, sum(M_j))/2`. | Exact existing eigenstep only. |
| `bootstrap` | `FALSE`; if true, resamples subjects, refits data-dependent bases and forms percentile bands. | No bootstrap implementation. |
| `nBootstrap` | `NULL`, an error with `bootstrap=TRUE`. | Not applicable. |
| `bootstrapAlpha`, `bootstrapStrat` | `0.05`, `NULL`; optional within-stratum resampling. | Not applicable. |
| `verbose` | `options()$verbose`. | No progress output from the estimator. |

R's `uFPCA` is **not fdapace**. It calls `MFPCA::PACE`, adapted from refund's
`fpca.sc`, with `nbasis=10`, `pve=0.99`, `npc=NULL`, `makePD=FALSE`, and
`cov.weight.type="none"`. `npc` overrides `pve`. It fits the mean with an
`mgcv` spline, forms pairwise empirical residual products using observation
counts, removes the diagonal, and smooths the covariance with a tensor spline.
It solves the trapezoidally weighted covariance eigenproblem, estimates noise
from the middle half of the domain, and calculates conditional scores via
``(\Phi_i^\top\Phi_i+\sigma^2\Lambda^{-1})^{-1}\Phi_i^\top(Y_i-\mu_i)``.
FDA instead reuses its dense sample covariance or fdapace-style local linear
PACE. These smoothing choices and default rank caps can change the result.

For orthonormal expansions, R's exact multivariate eigenproblem is
`cov(allScores) * outer(allWeights, allWeights)`, with `allWeights` repeating
``\sqrt{w_j}`` for each univariate component. Nonorthogonal expansions use
the block basis Gram matrix through its Cholesky factor; they are outside
this API's FPCA-only first step. R has no explicit deterministic sign rule
for its original fit; bootstrap functions are aligned to that fit.

To describe R's normalization, set ``Q=A/\sqrt{n-1}``. R stores eigenvectors
``v_m`` and ``h_m=\|Qv_m\|^{-1}``, returns scores
``A v_m\sqrt{\nu_m}h_m``, and builds node functions from coefficients
``(Q^\top Qv_m)^{(j)}h_m/\sqrt{w_j\nu_m}``.
For centered orthonormal expansions, ``h_m=1/\sqrt{\nu_m}`` and these reduce
to Proposition 5. R fitted values add its mean functions to the resulting
score/function expansion. FDA reproduces this centered eigenproblem and
uncentered normalization, including for noncentered conditional scores.

The supported strict validation route is `type="given"`, supplying the FDA
univariate functions and scores, with **explicit** `ortho=TRUE`. R otherwise
treats given functions as nonorthogonal. FDA's exporter fixes the observations,
different node grids, and first-step arrays. R then independently computes
the multivariate eigenvalues, coefficient vectors, node eigenfunctions,
scores, and fitted curves. It applies the stated coefficient sign rule.
The strict test compares both unit and unequal weights at relative error
below `1e-8`. A mixed dense/PACE fixture also supplies FDA's nonzero-mean
conditional scores via `type="given"`, checking the centered eigenvalues,
eigenvectors, normalization factors, scores, functions, and reconstructed
curves. Since R's `given` route uses empirical mean functions while FDA retains
the supplied PACE mean, the reference reconstructions subtract R's mean and
add FDA's first-step mean. This isolates the multivariate step without changing
its scores or functions. CI consumes CSVs and does not run R.

A separate **cross-method** fixture runs R's own `uFPCA` end to end (`npc=2`,
other defaults unchanged), on the same three-node dense data. FDA uses two
dense components per node. The deterministic curves have linear means and
rank-two constant/linear residual spaces; the comparison checks this shared
regime rather than general smoother equivalence. The linear mean and
constant/linear residuals lie in both smoothers' unpenalized spaces, so this
fixture is exact up to numerical error. Joint function signs are
aligned once across all nodes. Eigenvalues, functions, scores, and fitted
curves use relative tolerance `1e-8` for this cross-method check, as do the
strict given-expansion comparisons.

## Defaults, implementation choices, and validation

| FDA keyword or choice | Value and source |
|:--|:--|
| Multivariate `ncomponents` | Required; follows R `M`. No silent cap or FVE selection. |
| `weights` | All ones, R MFPCA 1.3-11. Represented in the data floating type; explicit higher precision weights promote the result. |
| `univariate` | Empty named tuple; task requirement to reuse marginal FPCA. All inherited defaults and their individual sources are listed in [FPCA defaults](fpca.md#Defaults-and-where-they-come-from). Node-specific overrides are copied and stored. |
| Multivariate `eigen_tol` | `1e-10`, inherited package numerical cutoff; exposed separately from `univariate.eigen_tol`. |
| Eigenanalysis | Exact, R's `approx.eigen=FALSE` default, reusing `_fpca_eigen`. |
| Signs | Largest coefficient positive; first entry on ties, as permitted by the task. |
| Score centering and divisor | Center for the covariance eigenproblem with divisor `n-1`; R MFPCA 1.3-11 exact route. |
| Normalization and score projection | Uncentered scores with R's `normFactors`, function coefficients, and score scales, also stored for prediction. |
| Rank-deficient requests | Error rather than silently adding null components. Inherited numerical cutoff is recorded. |
| Reconstruction grid | Node training grids; shared or per-node overrides use existing piecewise-linear evaluation and domain checks. |
| IDs | Existing FPCA rule: preserve for stored training scores, no inferred IDs for other matrices. |
| PACE prediction result | Joint score matrix; no assumption of independent nodes to manufacture joint conditional uncertainties. |
| Eager fitting and bootstrap | Disabled, consistent with R defaults; reconstruction is an explicit operation. No bootstrap API. |

The inherited first-step defaults include routing, work grid, quadrature,
`FVE(0.99)`, `maxK`, no dense presmoothing, sample dense mean/covariance,
estimated PACE mean/covariance/noise, Gaussian kernel, observation weighting,
5%/10% range bandwidths, diagonal bandwidth inheritance, automatic binning,
`eigen_tol=1e-10`, and `sign_tol=1e-12`. The linked univariate docstring/audit
specifies their sources, exceptions for supplied inputs, and resolved settings.
All are visible in `show(model)` through the retained univariate fits.

Recreation commands (R is only needed for regeneration):

```sh
julia --project=. test/reference/mfpca_m4.jl
Rscript test/reference/mfpca_m4.R
julia --project=. -e 'include("test/mfpca.jl")'
```

Reference generation uses R **4.5.1**, MFPCA **1.3-11**, funData **1.3-9**,
and mgcv **1.9-3**. The test prints strict and cross-method maximum relative
errors. Its analytic checks include the single-node reduction, independent
nodes' union spectrum, weighted orthonormality, global weight scaling, unit
invariance, and training reconstruction. Further checks cover unequal grids
and counts, mixed dense/PACE prediction, nonzero-mean conditional scores,
inference, Float32/promotion, IDs, copied inputs, and invalid arguments.

```@docs
MFPCA
ncomponents(::MFPCA)
component(::MFPCA, ::Integer)
eigvals(::MFPCA)
eigenfunctions(::MFPCA)
scores(::MFPCA)
predict(::MFPCA, ::MultivariateFunctionalData)
reconstruct(::MFPCA)
```
