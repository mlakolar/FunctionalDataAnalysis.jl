# Milestone 4 task 3: cross-covariance validation

Implemented `CrossCovariance`, `SmoothedCrossCovariance`, and
`cross_covariances`. The dense estimator returns sample cross-covariance;
the smoother reuses milestone 2's local linear regression, subject-level CV,
and weight-preserving pair aggregation. It supports separate axis bandwidths.
The block helper uses existing diagonal covariance fits and fits unordered
cross-node pairs once. The reverse block is an independent transposed copy.

The [fdapace source audit](@ref fdapace-cross-covariance)
documents the 0.6.0 implementation and its differences from this API.
The [Smoothing page](@ref smoothing-cross-covariance) gives the public contract.

## Defaults and their sources

| Setting | Default and source |
|:--|:--|
| Dense centering and divisor | Separate node sample means; `n-1`, matching fdapace 0.6.0 `GetRawCrCovFuncFunc`'s dense `cov` route. Missing values throw by package convention. |
| Direct smoother work grids | Required `grid1`; `grid2=grid1`, as specified by this task. Each grid must cover its own node's times. fdapace instead constructs two 51-point observed-range grids. |
| Kernel | Gaussian, from `GetCrCovYX(kern="gauss")`, also used by the block helper's diagonals and estimated means unless overridden. |
| Bandwidth | Required scalar, `(h1,h2)`, or explicit-candidate `CurveCV`. This deliberate departure from fdapace's automatic Gaussian GCV follows the task's bandwidth interface and the existing covariance smoother's explicit tuning / subject-level CV contract. |
| Same-time pairs | Keep (`exclude_same_time=false`), matching `rmDiag=FALSE`. Assumes independent measurement errors across nodes. |
| Pair weighting | `:obs`, one per pair, matching `GetCrCovYX` smoothing weights. Optional `:subj` is the task's `1/(N1_i*N2_i)` weighting. |
| Binning | `nothing`, as requested. Explicit integers reuse milestone 2 snapping separately on each observed axis range. fdapace's automatic duplicate grouping is algebraically equivalent to our unsnapped observation weighting for a fixed bandwidth. |
| Polynomial degree | One in each axis, using the three-column local linear fit from milestone 2 and `Lwls2D`. No interaction column is added. |
| CV folds, RNG, ties | Inherited `CurveCV`: ten folds; `rng=nothing` only for deterministic leave-one-subject-out; otherwise a copied RNG is required. First minimum wins; undefined candidates throw. Both means stay fixed. |
| Mean inputs in block helper | `fpca=nothing,means=nothing`. Estimate when neither is provided. These are optional explicit inputs required to support task 1 means. |
| Estimated block means | Sample means for complete grid nodes; otherwise local linear smoothing with bandwidth 5% of the observed range, Gaussian kernel, and observation weighting. These are task 1 marginal FPCA / audited fdapace PACE defaults. Explicit `kernel` and `weighting` apply to estimation. |
| Block work grids | `grid=nothing`: supplied mean/FPCA grids, or complete data grids, or 51 equally spaced observed-range points for irregular/masked nodes. Follows the corresponding task 1 / univariate FPCA defaults. |
| Mean smoothing override | `mean_bandwidth=nothing`, inheriting the previous row's mean-estimation contract. An explicit shared or per-node choice requests smoothing, including for complete grids. |
| Diagonal noise bandwidth | `diagonal_bandwidth=nothing`: selected covariance bandwidth, inherited from `SmoothedCovariance`. Explicit overrides pass through unchanged. |
| Interpolation and quadrature | Bilinear surface evaluation inside the grids; trapezoidal weights stored separately for each axis. Dense fits retain the input's stored weights. These follow the task and package conventions. |

## Details resolved by the implementation

- A node's common grid need not equal the other node's common grid. Dense
  covariance is a rectangular sample cross-covariance without resampling.
- Direct cross fits accept `SmoothedMean` or one complete grid curve as each
  mean. Existing `SmoothedCovariance` now also accepts the latter, so the
  helper reuses task 1 means without fitting or inventing another mean.
- `fpca=` accepts marginal and pooled multivariate fits. Node count, subject
  count, and available IDs are checked. `means=` accepts a vector per node.
  Conflicting mean sources and unused mean-bandwidth overrides throw.
- The helper takes one shared scalar or scalar-candidate `CurveCV` for
  covariance smoothing. Direct cross fits support anisotropic fixed widths.
  No unrequested mapping from arbitrary pairwise widths to diagonal widths
  is inferred. Mean bandwidths and grids can be supplied per node.
- The helper always smooths, including complete-grid inputs, to return the
  specified existing diagonal estimators. The separate dense estimator is
  the explicit unsmoothed route.
- Equal-time exclusions use exact original times before snapping. Subject
  weights retain denominator `N1_i*N2_i` after exclusions. A subject with no
  surviving pairs throws. Single observations per node are otherwise allowed
  in a cross fit, provided the pooled pair cloud identifies the local fits.
- Binned CV uses snapped pair coordinates, matching the existing covariance
  smoother. Within-cell scatter preserves the full raw-product squared loss.
  Means are evaluated before snapping and remain fixed across CV folds.
- Off-diagonal fits store independent copies of the means. Transposed blocks
  copy grids, weights, means, selection records, and surfaces, and reverse
  the axis-specific settings. Mutating one block does not mutate its reverse.

The local roadmap records these API additions. The externally linked plan
was inaccessible during development and still needs these additions,
including the existing covariance estimator's grid-mean input form.

## Reproduction and checks

Reference generation uses **R 4.5.1 and fdapace 0.6.0**:

```sh
Rscript test/reference/fdapace_m4.R
```

The committed generator writes fixed two-node sparse inputs and both
same-time variants of `GetCrCovYX`'s full 51 by 51 surface. Bandwidths are
`(.55,.8)`, the kernel is Epanechnikov, supplied means are linear, and
weighting is one per observation pair. The script asserts that fdapace's
exact-duplicate aggregation is exercised. CSV headers retain versions.
CI reads these files and does not run R.

To rerun these tests against KernelSmoothing's GitHub `master` in a
temporary environment without a Manifest:

```sh
testenv=$(mktemp -d)
julia +release --project="$testenv" -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/mlakolar/KernelSmoothing.jl", rev="master")); Pkg.develop(PackageSpec(path=pwd())); Pkg.add(["Test", "DelimitedFiles", "StatsAPI"])'
FDA_M4_CROSS_REPORT=true julia +release --project="$testenv" -e 'using FunctionalDataAnalysis, Test, LinearAlgebra, Random; include("test/cross_covariance.jl"); include("test/covariance.jl")'
git diff --check
```

Observed errors at the stated tolerances:

| Reference / recovery quantity | Observed error or bias |
|:--|--:|
| `GetCrCovYX`, same-time pairs kept: relative surface error | `2.1489083171866735e-15` |
| `GetCrCovYX`, same-time pairs excluded: relative surface error | `2.1809984872497296e-15` |
| Independent-noise recovery, pairs kept: maximum absolute error | `0.053766546992685016` |
| Independent-noise recovery, pairs excluded: maximum absolute error | `0.0640970231699981` |
| Independent-noise surface change after exclusion: maximum absolute difference | `0.020234531887112994` |
| Balanced correlated noise, pairs kept: mean diagonal bias | `1.357528666109249` |
| Balanced correlated noise, pairs excluded: maximum absolute error | `0.00494651811319613` |

## Recovery interpretation

The seeded Gaussian recovery test uses 1,000 subjects, five times sampled
from an eleven-point grid per subject, a shared standard Gaussian score,
node functions `1+.5t` and `.7-.2t`, and independent Gaussian measurement
noise with standard deviation `.4`. The analytic cross-covariance is their
outer product. Maximum absolute error must be below `.2` with either
same-time setting; changing that setting must change the surface by less
than `.08`. These are finite-sample diagnostic tolerances, not a statistical
guarantee.

The correlated-noise check isolates systematic bias using 125 sparse
designs, each repeated for eight balanced sign patterns (1,000 subjects).
Six orthogonal nonconstant Walsh columns represent one score and five
time-specific errors, each with unit second moment and zero cross moments.
Both nodes share errors with standard deviation two. This makes all
off-time error products cancel exactly while retaining same-time noise
covariance four. The mean diagonal bias with same-time pairs must exceed
`.5`; excluding them must recover the analytic surface within `.03` and
reduce Frobenius error by at least a factor of three. This checks the
exclusion rule and smoothing bias separately from high-noise sampling
variation. It does not assert that every Gaussian sample of 1,000 subjects
has such a small maximum error.
