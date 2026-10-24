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
  middle-half integrated noise variance. `diagonal_bandwidth` defaults to the
  covariance bandwidth and may be set separately as a number or `CurveCV`.
  The mean `bandwidth` and `diagonal_bandwidth` also accept KernelSmoothing
  regression `BandwidthSelector`s and pilot-centred `CurveCV(; pilot, multipliers)`;
  the 2D covariance bandwidth needs a number or explicit `CurveCV` candidates.
  Both weighting schemes match fixed-bandwidth fdapace covariance fixtures. Full cross-version tests and docs remain controller gates.
  Pointwise mean bands await cluster-robust variance support in KernelSmoothing.jl.
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
  Cross-version tests and the docs build remain controller gates.
- Milestone 4 (task 1 implemented): marginal multivariate FPCA delegates to
  dense or PACE fits per node. Pooled complete-grid FPCA uses the average node
  sample covariance, a shared weighted eigenbasis, and node-specific means
  and score variances. `score_array` provides subject × node × component
  order for FunctionalGraphicalModels.jl. Its migration, cross-covariance
  surfaces, and Happ--Greven multivariate FPCA remain subsequent tasks.
  See [Multivariate FPCA validation](m4_report.md) for defaults and checks.
- Milestone 5: registration, functional regression, depth and outliers,
  plotting recipes.
