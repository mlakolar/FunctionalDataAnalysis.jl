# Clustering and classification

Three clustering methods partition a sample of curves:

- [`FunctionalKMeans`](@ref): k-means of the curves in L², or of their FPC
  scores.
- [`KCFC`](@ref): k-centres functional clustering (Chiou and Li, 2007), which
  assigns each curve to the cluster whose own FPCA reconstructs it best.
- [`FunctionalGaussianMixture`](@ref): a Gaussian mixture with an unrestricted
  covariance per cluster, fitted to FPC scores.

All three results are `Clustering.ClusteringResult`s from
[Clustering.jl](https://github.com/JuliaStats/Clustering.jl), so `assignments`,
`nclusters`, `Clustering.counts`, `randindex` and Clustering.jl's other
comparison and quality measures apply directly. Every setting is stored on
the result and shown with it. Classification methods are added in a later
release.

## Example

Three groups of thirty curves differ in their mean and in their dominant mode
of variation: group `g` has standard deviation 1 along mode `g` and 0.2 along
the other two.

```@example clustering
using FunctionalDataAnalysis, Random
import StatsAPI
rng = Xoshiro(2026)
grid = collect(range(0, 1; length=41))
group = repeat(1:3; inner=30)
modes = sqrt(2) .* [sinpi(j * t) for t in grid, j in 1:3]
sd(g) = [j == g ? 1.0 : 0.2 for j in 1:3]
curve(g) = 1 .+ grid .+ 1.2 * (g - 2) .* sinpi.(grid) .+ modes * (sd(g) .* randn(rng, 3))
fd = GridFunctionalData(reduce(vcat, [transpose(curve(g)) for g in group]), grid)
nothing # hide
```

k-means of the curves uses the stored quadrature weights for the L² distance.
k-means of FPC scores clusters a fitted [`FPCA`](@ref).

```@example clustering
km = StatsAPI.fit(FunctionalKMeans, fd; k=3, nstart=5, rng)
fpca = StatsAPI.fit(FPCA, fd; ncomponents=FVE(0.9))
scores_km = StatsAPI.fit(FunctionalKMeans, fpca; k=3, nstart=5, rng)
(randindex(km, group)[1], randindex(scores_km, group)[1])
```

The groups overlap in L², so both k-means partitions agree poorly with them.
kCFC starts from k-means of the FPC scores of all curves (here the best of ten
starts) and then reclassifies curves by their reconstruction error under each
cluster's FPCA (one component per cluster here). Like k-means, it finds a
local optimum that depends on its start; `initial` supplies another one.

```@example clustering
kcfc = StatsAPI.fit(KCFC, fd; k=3, cluster_fpca_options=(; ncomponents=1),
    kmeans_options=(; nstart=10), rng)
(kcfc.status, kcfc.iterations, randindex(kcfc, group)[1])
```

The Gaussian mixture is fitted to the FPC scores. A range of `k` selects the
number of clusters by BIC.

```@example clustering
gm = StatsAPI.fit(FunctionalGaussianMixture, fpca; k=2:4, rng)
(gm.k, gm.selection.values, randindex(gm, group)[1])
```

Cluster centres are functional data: `km.centers`, `scores_km.centers`,
`kcfc.centers` (the cluster FPCA means) and `gm.centers` (the curves of the
cluster mean scores) can be evaluated and plotted like any other data, for
example with `curve_data`. `predict` assigns new curves; for the score-based
methods it projects them with [`project_scores`](@ref) of the stored FPCA.

## Defaults and their sources

| Choice | Default and source |
|:--|:--|
| k-means distance | L² with stored quadrature weights (grid) or the exact Gram matrix (basis); trapezoidal like fda.usc 2.2.0 `metric.lp` |
| k-means clusters | `k=2`, fda.usc `kmeans.fd` `ncl=2` |
| k-means seeding | `init=:kmpp` (k-means++, Clustering.jl's default); fda.usc's `method="sample"` is `init=:spread`; R's random rows are `init=:rand` |
| k-means starts | `nstart=1`, as R `kmeans` and fda.usc |
| k-means iterations | `maxiter=100`, fda.usc `max.iter` |
| k-means tolerance | `tol=1e-6` relative to the total sum of squares (package choice: Clustering.jl's `tol` is absolute and so depends on units) |
| kCFC clusters and iterations | `k=3`, `maxiter=125`, fdapace 0.6.0 `kCFC` |
| kCFC FPCAs | FVE 0.9 for all curves and 0.7 per cluster (fdapace `kCFC`), otherwise the package's `fit(FPCA, ...)` defaults |
| kCFC initial partition | Lloyd k-means with k-means++ seeding on the FPC scores, `maxiter=125` |
| Mixture clusters | `k=3`, fdapace `FClust`; a range selects by `criterion=BIC()` |
| Mixture FPCA | FVE 0.9, fdapace `FClust` |
| Mixture covariance | unrestricted per cluster (EMCluster's model) |
| Mixture initialization | `init=:emem`, EMCluster 0.2-18 `em.EM` with `short_iter=200`, `short_tol=0.01`, `restarts=10` |
| Mixture EM | `tol=1e-6` relative log-likelihood change, `maxiter=1000` (EMCluster `em.eps`, `em.iter`); `parameter_tol` optional |

## Departures from the references

- **k-means of curves.** fda.usc 2.2.0 `kmeans.fd` builds its distance matrix
  once, from every curve to the initial centres (`kmeans.center.ini`), and the
  loop in `kmeans.fd` never recomputes it after `kmeans_centers_update`.
  Every pass therefore reassigns curves to the nearest *initial* centre, the
  centres stop changing after one update, and the result is the
  nearest-initial-centre partition with depth-trimmed centres
  (`func.trim.FM`, `trim=0.05`). The package runs Lloyd's algorithm to
  convergence with plain means, as R's `kmeans`. With `trim=0`, fda.usc's
  centres are plain means, and when one more Lloyd step changes nothing both
  agree; the validation uses such a case.
- **Seeding.** k-means++ replaces fda.usc's best-of-100-random-subsets rule
  (still available as `:spread`) because its expected cost is within
  `O(log k)` of the optimum (Arthur and Vassilvitskii, 2007).
- **kCFC FPCAs.** fdapace's default `kCFC` options smooth the mean and
  covariance with GCV bandwidths. The package uses its own FPCA defaults:
  dense cross-sectional estimates for complete grids (no smoothing) and PACE
  otherwise. Pass `fpca_options` and `cluster_fpca_options` to change them.
- **kCFC initial partition.** fdapace calls R
  `kmeans(algorithm="MacQueen")` after `set.seed(kSeed=123)`. That stream
  cannot be reproduced outside R; the package uses Clustering.jl's Lloyd
  iteration with k-means++ seeding, records the initial labels and accepts an
  R partition as `initial`.
- **kCFC on sparse data.** fdapace 0.6.0 `kCFC` stops unless the data are
  dense, and its `GetISEfromFPCA` always projects curves by numerical
  integration (`GetINScores`), whatever the cluster FPCA's score method. The
  package projects with each cluster FPCA's own [`project_scores`](@ref): the
  same trapezoidal projection for dense FPCA fits, and PACE conditional
  expectations for irregular or masked data, which makes kCFC usable for
  sparse curves. Cluster FPCAs then share the initial FPCA's work grid.
- **Mixture EM.** EMCluster stops when the relative log-likelihood change is
  at most `em.eps`; near the optimum the log likelihood is flat, so the
  parameters remain about `sqrt(em.eps)` from the EM fixed point. On the
  validation data EMCluster with `em.eps=0` stopped after 40 iterations while
  one more EM step still moved the means by `2e-8`. A positive
  `parameter_tol` stops on parameter changes instead (zero is rejected,
  because rounding can keep the iteration from reaching an exact fixed
  point). Log-likelihood decreases within `100 eps |loglik|` are treated as
  rounding rather than as EMCluster's decrease stop, so a run that starts at
  the optimum reports `:converged`. FClust calls `emcluster` a second time
  after `em.EM` with the same criterion; the package runs one EM. EMCluster
  can count a short-EM start that immediately decreases the likelihood as
  minus one iteration; the package counts at least one, so initialization
  always ends. When the final EM degenerates (a component with posterior
  weight sum at most the number of scores, or a singular covariance), the
  package redraws the `em.EM` initialization within `restarts`, as EMCluster
  does for a small final class, and throws after the last attempt instead of
  producing a singular covariance. Initialization fails, and counts as an
  attempt, when 1000 random partitions all have a class of at most that many
  curves or every short run degenerates. A `k` with more classes than the
  curves can fill fails at once. In a range of `k`, a `k` that fails in every
  attempt is skipped with criterion `Inf` and listed in
  `selection.degenerate`. The fit throws only if every `k` fails.
- **Selecting the number of mixture components.** FClust fits a fixed `k`.
  A range of `k` uses EMCluster's `em.bic` (or `em.aic`), the criterion of its
  `runemEMcluster`, with BIC as the default because it is consistent for the
  number of components (Keribin, 2000).

## Implementation

k-means uses Clustering.jl `kmeans` on the curves scaled by the square roots
of the quadrature weights (or multiplied by the Cholesky factor of the basis
Gram matrix), so that Euclidean distances are L² distances. The Gaussian
mixture's E and M steps are GaussianMixtures.jl `em!` on a full-covariance
`GMM`; the EMCluster initialization, stopping rules and decrease guard wrap
single `em!` steps. kCFC is implemented in the package and reuses
[`fit(FPCA, ...)`](@ref FPCA), [`project_scores`](@ref) and the FPCA
reconstruction at observation times used by `residuals`.

## Reference source audit

Versions audited: fdapace 0.6.0 (`kCFC`, `FClust`, `GetISEfromFPCA`,
`GetINScores`, `List2Mat`), fda.usc 2.2.0 (`kmeans.fd`, `kmeans.center.ini`,
`kmeans_assig_groups`, `kmeans_centers_update`, `metric.lp`) and EMCluster
0.2-18 (`em.EM`, `init.EM`, `emcluster`, `assign.class` and the C routines
`shortems`, `randomEMinit`, `initials`, `shortemcluster`, `emcluster`,
`mstep`).

- fdapace `FClust` defaults to `cmethod="EMCluster"` with `k=3` and FPCA
  options `methodMuCovEst="smooth"`, `FVEthreshold=0.9` and GCV bandwidths.
  It clusters `FPCA(...)$xiEst` with
  `emcluster(xi, em.EM(xi, nclass=k), assign.class=TRUE)`.
- EMCluster fits one unrestricted covariance per component. `em.EM` draws
  `k` distinct observations as centres, assigns every observation to the
  nearest one and starts from class means, covariances with denominator
  `n_c - 1` and proportions `n_c/n`, redrawing while a class has at most `p`
  members; short EM runs (gain relative to the run's total gain at most
  `short.eps=0.01`) share a budget of `short.iter=200` iterations, and the
  best start (`>=`, so the last on ties) is run by `emcluster` until the
  relative log-likelihood change is at most `em.eps=1e-6` or `em.iter=1000`
  iterations; a decrease reverts the last step. The R wrapper repeats the
  whole initialization up to `min.n.iter=10` times while a final class has
  fewer than `p+1` members, and warns if it fails. Labels maximize the
  posterior probability.
- fdapace `kCFC` has `k=3`, `kSeed=123`, `maxIter=125`, FVE thresholds 0.9 and
  0.7 and refuses sparse data. Its initial step is
  `kmeans(xiEst, k, algorithm="MacQueen", iter.max=maxIter)`; it stops if a
  cluster has three or fewer curves. Each iteration computes the trapezoidal
  ISE of every curve under every cluster FPCA, takes the minimum, stops for a
  lost cluster, for a cluster of at most `0.01n` or three curves, or when the
  partition equals an earlier iteration's (not the initial one), and
  otherwise refits the cluster FPCAs. `iterToConv` is the iteration count
  minus one; the returned FPCAs are those fitted to the previous partition.
- fda.usc `kmeans.fd` is described under the departures above; its default
  distance is `metric.lp` (trapezoidal L²) and its default centre is the FM
  depth-trimmed mean with `trim=0.05`.

## Validation

`test/clustering.jl` compares, at relative error below `1e-8`, on three
groups of twelve noise-free curves whose deviations vanish at both endpoints
(where fdapace's rectangular and the package's trapezoidal eigenproblems
coincide):

- `FunctionalKMeans` with seed curves against fda.usc 2.2.0 `kmeans.fd`
  (`trim=0`): labels, centres and `metric.lp` distances to the seeds;
- `FunctionalKMeans` of FPC scores against `stats::kmeans(algorithm="Lloyd")`
  from the same seed rows: labels, centres and within-cluster sums of squares,
  together with the FPC scores and eigenvalues against fdapace's dense
  cross-sectional FPCA;
- `KCFC` against fdapace 0.6.0 `kCFC` with cross-sectional FPCAs, starting
  from kCFC's recorded initial labels: final labels, iteration count, every
  intermediate partition, the final `n × k` ISE matrix and each cluster
  FPCA's mean and eigenvalues;
- `FunctionalGaussianMixture` from supplied labels against the fixed point of
  EMCluster 0.2-18's own E and M steps (`parameter_tol=1e-12`): proportions,
  means, covariances, posterior probabilities, labels, log likelihood, AIC
  and BIC. With the default `tol` the labels agree and the log likelihood is
  within the relative tolerance.

Random initializations (k-means++, `em.EM`, fdapace's seeded MacQueen
k-means) are not compared, because they depend on each language's random
number generator; their outputs are recorded so that a partition can be
supplied. Sparse kCFC is checked by recovering a known two-cluster partition.
Mixture initialization and BIC selection are checked on three groups of
one-dimensional scores six standard deviations apart, where the true
partition and `k = 3` are recovered whatever the random starts.

Recreate the references with `Rscript test/reference/clustering_m8.R` (R 4.5.1;
fdapace 0.6.0, fda.usc 2.2.0, EMCluster 0.2-18). No R installation is needed
to use the package or run its tests.

## Reference

```@docs
FunctionalKMeans
KCFC
FunctionalGaussianMixture
posterior_probabilities
StatsAPI.predict(::FunctionalKMeans, ::AbstractFunctionalData)
StatsAPI.predict(::FunctionalGaussianMixture, ::AbstractFunctionalData)
```
