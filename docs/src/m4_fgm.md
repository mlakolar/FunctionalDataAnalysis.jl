# Comparison with FunctionalGraphicalModels.jl

The milestone-4 fixtures compare FDA with FunctionalGraphicalModels.jl (FGM)
version 0.1.0, branch `bsplinekit`, commit
`6f0eac3d5cbe0fad93e88dcad28f389d89b61495`. The reference implementations are
[`src/fpca.jl`](https://github.com/mlakolar/FunctionalGraphicalModels.jl/blob/6f0eac3d5cbe0fad93e88dcad28f389d89b61495/src/fpca.jl)
and
[`src/doubly.jl`](https://github.com/mlakolar/FunctionalGraphicalModels.jl/blob/6f0eac3d5cbe0fad93e88dcad28f389d89b61495/src/doubly.jl).
The input data are deterministic. CI reads the committed CSVs and does not
load FGM or regenerate its fits.

`test/data/m4_fgm_dense.csv` contains the nonuniform common grid, all
12-subject, two-node observations, and the marginal and pooled fits with two
components. The test compares eigenfunctions, scores, and node-specific score
variances at relative error below `1e-8`. Eigenfunction columns and scores
are sign aligned before comparison because FGM's dense eigenvectors have no
sign normalization. FDA uses sample covariance and score variances with
denominator `n - 1`; the stored FGM score variances use denominator `n`.
The expected FDA variances are therefore the FGM values times `n/(n - 1)`.
This change does not change the eigenfunctions or scores.

`test/data/m4_fgm_sparse.csv` contains 18 subjects and two nodes with four
to six observations per curve. Interior observation times differ across
subjects and generally lie between the 21 work-grid points. The comparison
uses these explicit settings:

| Choice | FDA setting | FGM setting |
|:--|:--|:--|
| Route | Marginal PACE | `fit_sparse_fpca` |
| Kernel | `KernelSmoothing.GaussianKernel()` | `:gaussian` |
| Curve weighting | `weighting=:subj` | Equal total subject weight |
| Mean, covariance, and diagonal bandwidths | `0.3` each | `bandwidth=0.3` |
| Binning | `binning=nothing` | No binning |
| Measurement noise variances | `[0.04, 0.07]` | `[0.04, 0.07]` |
| Selected components | `ncomponents=2` | `ncomponents=2` |
| Covariance rank for conditional scores | `maxK=21` | All retained positive eigenvalues |
| Relative eigenvalue cutoff | `eigen_tol=100eps(Float64)` | `100eps(Float64)` |

The sparse test compares smoothed means, selected eigenvalues,
eigenfunctions, full projected covariances, and conditional scores at
relative error below `1e-8`. It aligns eigenfunction and score signs. Sparse
eigenvalues come from the smoothed covariance; there is no dense `n/(n-1)`
correction.

The fixture expresses FGM's settings explicitly rather than using FDA's
univariate defaults. In particular, FDA's default relative eigenvalue cutoff
is `1e-10`, whereas FGM uses `100eps(Float64)`. FDA also limits the covariance
rank through `maxK`; the comparison retains the complete positive spectrum
by setting `maxK` to the work-grid length. Both packages use all retained
covariance components in the conditional-score denominator, even though only
two scores are returned. FGM's noise-estimation rule differs from FDA's;
the fixture supplies noise variances so that neither rule enters the
comparison. For a singular observation covariance FGM falls back to a
Moore–Penrose inverse, whereas FDA raises an informative error and requests a
positive noise variance. The positive supplied variances in this fixture
make the observation covariances positive definite, so that difference is
not exercised. Pooled PACE is not implemented in FDA.

To reproduce the fixtures, first copy the pinned FGM checkout to a temporary
directory. Run its `scripts/instantiate.jl` there, then run
`test/reference/fgm_m4.jl` with that temporary project and FDA's `test/data`
directory as the output argument. The script header gives the commands.
Never instantiate the preserved reference worktree itself. Each generated
CSV header records the FGM branch and full commit, Julia version, dependency
versions, and dependency tree hashes. The committed values were generated
with Julia 1.13.2 and BSplineKit 0.19.3.
