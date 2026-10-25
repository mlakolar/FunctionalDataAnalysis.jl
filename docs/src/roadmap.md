# Roadmap

- Milestone 0: package skeleton, CI, docs.
- Milestone 1 (done): data types (grid, irregular, basis, multivariate), conversions,
  Tables.jl input, simulation generators.
- Milestone 2 (in progress): per-curve local polynomials and penalized splines
  are implemented. Smoothing returns grid or basis data with a copied `metadata`
  named tuple retaining tuning settings. Common-lambda GCV stores the candidate
  grid, summed per-curve scores, and selected lambda in `metadata.selection`.
  `StatsAPI.fit` entry points return the same functional data objects.
  Pooled means use `StatsAPI.fit(SmoothedMean, ...)`, with observation or subject
  weighting. `CurveCV` selects bandwidths by leaving out whole curves and stores
  candidates, errors, and fold assignments on the result. Fixed-bandwidth mean
  tests match fdapace for both weighting schemes.
  Covariances use `StatsAPI.fit(SmoothedCovariance, fd, mean; ...)`, with
  off-diagonal product-kernel local linear smoothing, optional time binning,
  curve-level CV conditional on the supplied mean, bilinear evaluation, and
  rotated full-interval noise variance (selected in milestone 5, issue #17).
  The original surface-diagonal, middle-half rule remains explicit.
  `diagonal_bandwidth` defaults to the
  covariance bandwidth and may be set separately as a number or `CurveCV`.
  The mean `bandwidth` and `diagonal_bandwidth` also accept KernelSmoothing
  regression `BandwidthSelector`s and pilot-centred `CurveCV(; pilot, multipliers)`;
  the 2D covariance bandwidth needs a number or explicit `CurveCV` candidates.
  Both weighting schemes match fixed-bandwidth fdapace covariance fixtures.
  Mean uncertainty is available through the milestone 5 curve bootstrap and
  KernelSmoothing curve-cluster sandwich intervals.
- Milestone 3 (done): dense univariate FPCA is implemented, with a
  stored weighted eigenbasis, quadrature scores, FVE or fixed component
  selection, optional per-curve presmoothing, prediction, and reconstruction.
  The dense default component cap is `max(1, min(length(grid)-2, nobs(data)-2))`,
  following the fdapace source audit and replacing the plan's cap of 20.
  Source-audited fdapace fixtures test the documented eigenvalue convention.
  PACE now reuses pooled mean and covariance smoothing, accepts supplied
  means, surfaces, and noise variance, and stores conditional-expectation
  scores and their conditional covariances. FVE, fixed K, AIC, and BIC are
  supported; AIC/BIC retain the full criterion curve and global minimizer
  alongside fdapace's first-increase choice. Sparse prediction reuses the
  fitted mean, eigenfunctions, full projected covariance, and noise variance.
  No-keyword fits use source-audited PACE defaults: a 51-point work grid,
  Gaussian kernel, observation weighting, and mean/covariance bandwidths of
  5%/10% of the observed range. Every choice is retained and shown.
  `pace_bandwidth_candidates` exposes fdapace's geometric candidates for
  optional whole-curve CV. Automatic binning follows the audited bin-count
  rule and reuses the existing weighted covariance-pair aggregation.
  End-to-end fixed-bandwidth and default references, sparse recovery,
  dense-versus-PACE agreement, and a bounded-time sparse fit complete the
  milestone validation. Intentional fdapace differences are documented on
  the FPCA and audit pages. Roughness-penalized FPCA, FACE, and conditional
  score `rho` regularization remain outside this milestone's current scope.
- Milestone 4 (done in FDA): marginal multivariate FPCA delegates to
  dense or PACE fits per node. Pooled complete-grid FPCA uses the average node
  sample covariance, a shared weighted eigenbasis, and node-specific means
  and score variances. `score_array` provides subject × node × component
  order for FunctionalGraphicalModels.jl. `CrossCovariance` returns exact
  dense sample cross-covariances; `SmoothedCrossCovariance` reuses the local
  linear smoother with scalar or separate axis bandwidths and subject CV.
  `cross_covariances` returns diagonal covariance and transposed cross blocks,
  optionally using task 1 FPCA means. Same-time pairs are kept under independent
  cross-node noise, or explicitly excluded. Happ--Greven `MFPCA` combines the
  marginal scores through a weighted joint eigenproblem, with prediction,
  reconstruction, unequal domains/designs, and R MFPCA 1.3-11 validation.
  FGM estimator migration is maintained in the FGM repository.
  See [Multivariate FPCA validation](m4_report.md) for defaults and checks.
  See [Cross-covariance validation](m4_cross_report.md) for task 3.
  See [Multivariate FPCA (MFPCA)](mfpca.md) for task 4, its source audit,
  defaults, and validation.
- Milestone 5 (in progress): see and trust the fit. Task 1 implements a
  plotting-library-independent coordinate layer and RecipesBase recipes for
  Plots.jl, including observed data, means, modes, scree plots, covariance
  surfaces, observation designs, and fitted paths. See [Plotting](plotting.md)
  for defaults, simulated figures, and the fdapace 0.6.0 source audit.
  Task 2 adds exact derivative bases, polynomial curve/mean derivatives,
  FPC and DPC derivative fits, reconstruction and nodewise plotting. The
  derivative reference audit documents the consistent weighted DPC quantity
  and its departures from fdapace.
  Task 3 adds conditional PACE fitted-curve intervals and simultaneous bands,
  curve-bootstrap mean intervals and sup-t bands, curve-cluster sandwich mean
  intervals, and ribbon recipes. See [Uncertainty](uncertainty.md).
  Task 4 adds six classic datasets through a lazy artifact, with offline
  samples and source/licence provenance. DTI retains both cca and rcst on
  separate grids, with masks and scan/subject covariates. See [Datasets](datasets.md).
  Task 5 adds the rotated noise diagonal (fdapace `PC_CovE`; Yao, Müller and
  Wang 2005), full-interval integration and the Discovery benchmark in
  `benchmark/sigma2/` (issue #17); the rotated diagonal over the full interval
  is now the default. Makie recipes are deferred.
  This scope follows the Tier 1 Gap Map and Mladen's 24 October decision,
  replacing the original milestone 5 scope.
- Milestone 6 (in progress): scale and cross-structure. Dense-grid FACE is
  available through `fit(FPCA, data; method=:face, face=(; ...))`, including
  an internal coefficient-difference P-spline building block, GCV, missing-data
  iteration, shrinkage scores, pointwise posterior bands, and marginal/MFPCA
  reuse. The source audit and scale/variance departures from refund 0.1-40
  are documented on the [FPCA page](fpca.md). The cross-structure task adds
  weighted FSVD with dense integration and joint PACE scores, concurrent
  correlation through five local linear smooths, and dynamic correlation; see
  [Cross-structure](cross_structure.md). Roughness-penalized basis FPCA reuses
  the FPCA result and eigenstep,
  with exact basis prediction/reconstruction and explicit leave-one-curve-out
  lambda selection with an explicit fixed component count. Functional CCA adds
  paired basis weights, variates,
  penalized squared correlations, and optional leading-variate CV.
  The fda 6.3.0 reference and defaults are on the [FPCA](fpca.md#Smoothed-FPCA)
  and [CCA](cca.md) pages. Sparse FACE and the Discovery scaling study remain
  subsequent tasks. Multi-penalty REML belongs to milestone 7.
- Milestone 7: functional regression.
- Milestone 8: depth, outliers, and clustering.
- Milestone 9: registration.
