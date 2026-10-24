# Milestone 4: multivariate FPCA comparison

This report covers milestone 4 task 1: separate marginal FPCA fits and a
shared pooled basis. Functional graphical estimators, cross-covariance
surfaces, and Happ--Greven multivariate FPCA are later tasks.

The pooled route follows the partial-separability model of [Zapata, Oh, and
Petersen (2022), *Biometrika* 109(3), 665--681](https://doi.org/10.1093/biomet/asab046).

## Reference and reproduction

The FunctionalGraphicalModels.jl reference is branch `bsplinekit`, commit
`6f0eac3`, provided in `~/projects/juliaPkg/fgm-ref-bsplinekit`. Reference
execution uses a temporary copy of that checkout so the original worktree
is unchanged. `test/reference/fgm_m4.jl` is the committed generating script.
The CSV headers record the reference branch, commit, Julia version, and
relevant dependency versions. CI reads the committed CSVs and does not need
to instantiate the reference package.

Generation ran on Julia **1.13.2** with FunctionalGraphicalModels.jl 0.1.0,
BSplineKit 0.19.3, CovarianceSelection 0.1.0, ProximalBase 0.5.3, and
ProximalSolvers 0.1.1. The original FGM worktree remained clean and had no
`Manifest.toml` after generation.

The dense comparison aligns each reference eigenfunction's sign before
comparing scores. Reference eigenvalues and per-node pooled score variances
are multiplied by `n/(n-1)` because FDA uses sample covariance denominator
`n-1`; the reference uses `n`. Scores themselves have the same normalization.
The score array is ordered by subjects, nodes, then principal components.

## Defaults and sources

For marginal fits, every univariate keyword delegates to the corresponding
node without changing its resolved defaults. The full table and primary
sources are in [Defaults and where they come from](fpca.md). In particular,
nodes retain their own data route, work grid, bandwidths, selection rule,
noise estimate, and smoothing provenance.

| Setting | Default | Source |
|:--|:--|:--|
| Multivariate `basis` | `:marginal` | Milestone 4 task API. |
| Marginal `method` | `:auto` per node | Milestone 3: complete grid data use dense FPCA; irregular or masked data use PACE. |
| Pooled `method` | `:auto`, resolving to dense. | Pooled route requires complete common-grid data; `:dense` is also accepted. |
| Marginal `grid` | Data grid for dense; 51 equally spaced observed-range points for PACE. | Dense data representation; fdapace 0.6.0 `nRegGrid` and `FPCA.R`. |
| `ncomponents` | `FVE(0.99)` per node or on the pooled spectrum. | Task API and fdapace 0.6.0 `FVEthreshold`. |
| `maxK`, dense and pooled | `max(1,min(g-2,n-2))`; explicit caps no larger than `min(g,n-1)`. | Task requires the dense cap; fdapace 0.6.0 cross-sectional `SetOptions.R`. |
| `maxK`, marginal PACE | `max(1,min(20,g-2,n-2))`. | fdapace 0.6.0 smoothing-route `SetOptions.R`. |
| Marginal PACE mean/covariance | Estimate missing inputs with existing local-linear smoothers. | Yao, Müller, and Wang (2005), inherited milestone 3 route. |
| Marginal PACE `noise_variance` | Covariance smoother's nonnegative estimate; supplied surfaces require an explicit value. | Inherited milestone 3 contract and fdapace covariance smoothing. |
| Marginal PACE `kernel` | Gaussian when a smoother runs. | fdapace 0.6.0 non-dense `SetOptions.R`. |
| Marginal PACE `weighting` | `:obs`. | fdapace 0.6.0 smoothing default. |
| Marginal PACE `mean_bandwidth` | `0.05` times the node's observed time range. | fdapace 0.6.0 `methodBwMu="Default"`. |
| Marginal PACE `covariance_bandwidth` | `0.10` times the node's observed time range. | fdapace 0.6.0 `methodBwCov="Default"`. |
| Marginal PACE `diagonal_bandwidth` | Selected covariance bandwidth. | Existing milestone 2 smoother and fdapace diagonal smoothing. |
| Marginal PACE `binning` | `:auto`, using the existing audited fdapace bin-count rule. | fdapace 0.6.0 `GetBinNum.R`; [full rule](fpca.md). |
| Marginal dense `presmooth` | `nothing`. | Existing dense noise-free approximation; Ramsay and Silverman (2005), chapter 8. |
| Pooled mean and covariance | One sample mean and covariance per node; equal-node average covariance, denominator `n-1`. | Task specification; pooled partial-separability construction. |
| Quadrature | Stored trapezoidal weights, also on nonuniform grids. | Package inner-product convention and milestone 4 task. |
| `eigen_tol` | `1e-10`, relative to the largest absolute eigenvalue. | Existing adjustable package numerical convention. |
| `sign_tol` | `1e-12`, relative mean-overlap tolerance. | Existing adjustable package numerical convention. |
| Pooled sign reference | Average of node mean curves. | Explicit extension of the univariate mean-alignment rule to one shared basis; stored as `sign_reference=:average_node_mean`. |
| Reconstruction `scores` and `grid` | Training score matrices and node training grids. | Univariate reconstruction contract extended node by node. |

The fit stores and displays resolved settings. Marginal settings remain on
the per-node `FPCA` objects. Pooled selection and the shared eigensystem
remain separate from per-node score variances.

## Details resolved by the implementation

- A numeric `grid` vector denotes one shared grid, even when its length
  equals the number of nodes. A vector of grids denotes per-node grids.
  Other vector-valued keyword containers must have one entry per node.
  Per-node `nothing` leaves that node's univariate default active.
- Pooled `maxK` retains the task's dense-route bound `min(g,n-1)`. The
  frozen FGM implementation allows `min(g,p*(n-1))` because averaging node
  covariances can increase rank. FDA follows the requested dense cap;
  this restriction is explicit and does not imply that the pooled
  covariance has rank at most `n-1`.
- Per-node pooled score variances preserve pooled component order. They
  are not sorted separately, since sorting would disconnect node scores
  from the common basis.
- The common sign-reference curve is the equal-node average of the
  training mean curves. Near-zero overlap uses the existing largest-entry
  fallback, with the first entry breaking ties. This choice affects only
  signs, not the pooled covariance or score variances.
- Pooled tuning accepts shared `ncomponents`, `maxK`, `eigen_tol`, and
  `sign_tol`, with `method=:auto` or `:dense`. Presmoothing, custom work
  grids, and PACE-specific options are rejected explicitly. The task did
  not define pooled presmoothing or a new pooled work-grid method.
- `score_array` allocates an independent array. It copies node scores in
  the original node order and rejects unequal component counts.
- Reconstruction preserves subject IDs only when every supplied node
  score matrix is the corresponding stored training matrix. Copied or
  new score matrices carry no inferred training IDs.
- Pooled fits expose a shared basis and node score variances; they do not
  manufacture per-node `FPCA` objects. `component` therefore throws for
  pooled fits, as required by the API.
- Marginal node fits retain their original numerical types. The vectors
  returned by multivariate numerical accessors use the promoted type across
  nodes; widening a node's arrays does not discard precision. With a common
  type, these arrays and basis objects reuse the stored univariate objects.
  Prediction uses the original node fits. Reconstruction promotes the fit,
  supplied score, and supplied grid types instead of narrowing new inputs.

## Validation and limits

Development checks are restricted to touched test files on one Julia
version. The controller runs the complete suite on Julia 1.10 and the
current release, Aqua, and the documentation build after development.
Those controller checks are not claimed as completed here.

The final focused run used Julia **1.13.2**:

```sh
julia +release --project=/private/tmp/fda-m4-env.oOt4gn -e 'using FunctionalDataAnalysis, Test, LinearAlgebra, Random; include("test/multivariate_fpca.jl"); include("test/multivariate_fpca_golden.jl")'
```

```text
Test Summary:     | Pass  Total   Time
Multivariate FPCA |  257    257  34.9s
Test Summary:                          | Pass  Total  Time
Multivariate FPCA FGM reference values |   47     47  1.8s
```

All **304 checks passed**, including fit/accessor/prediction/reconstruction
inference, exact marginal delegation, nonuniform-grid analytic pooled
results, node order, IDs, mixed precision, keyword dispatch, and error paths.
`git diff --check` also passed. No Manifest was created in the FDA worktree
or preserved FGM reference worktree. No commits, branches, or pushes were
made. The local roadmap records the new API; the inaccessible external
plan still needs the `MultivariateFPCA` and `score_array` additions.

To recreate the temporary test environment from the FDA worktree:

```sh
testenv=$(mktemp -d)
julia +release --project="$testenv" -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(path=expanduser("~/projects/juliaPkg/KernelSmoothing.jl"))]); Pkg.add(["Test", "DelimitedFiles", "StatsAPI"])'
julia +release --project="$testenv" -e 'using FunctionalDataAnalysis, Test, LinearAlgebra, Random; include("test/multivariate_fpca.jl"); include("test/multivariate_fpca_golden.jl")'
```

The reference comparison ran on Julia **1.13.2** with this command:

```sh
FDA_M4_REFERENCE_REPORT=true julia +release --project=/private/tmp/fda-m4-env.oOt4gn -e 'using FunctionalDataAnalysis, Test, LinearAlgebra; include("test/multivariate_fpca_golden.jl")'
```

The temporary development environment resolves FDA from the current
worktree and KernelSmoothing from its local checkout. The golden test
result was **47 passed, 47 total**, in 5.6 seconds. The maximum relative
errors were:

| Reference comparison | Maximum relative error |
|:--|--:|
| Dense marginal and pooled | `1.6554831126432828e-15` |
| Sparse marginal PACE | `5.638229917901376e-14` |

Both are below `1e-8`. The sparse comparison uses FGM's explicit Gaussian
kernel, subject weighting, bandwidths, work grid, supplied noise variances,
relative eigenvalue cutoff, and full retained covariance rank. These are
compatibility settings, rather than FDA's unmodified defaults. See
[Comparison with FunctionalGraphicalModels.jl](m4_fgm.md) for every setting
and the sign, cutoff, and singular-covariance differences.

Pooled PACE, masked pooled fitting, and mismatched pooled grids are
unsupported and raise errors. The common-basis recovery check is a finite
Monte Carlo diagnostic, not a general statistical guarantee. It uses 3,000
Gaussian curves per node, two nodes with the same weighted-orthonormal
constant/linear basis, node score variances `[4,1]` and `[2,0.5]`, and the
fixed seed `20261024`. Each absolute weighted correlation with the true
basis must exceed `0.99`; each pooled score variance must be within
`6sqrt(2/(n-1))` times its target. These checks use inequalities rather than
exact random-number output across Julia versions.
