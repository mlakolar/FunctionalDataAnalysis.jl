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
- Milestone 3 (in progress): dense univariate FPCA is implemented, with a
  stored weighted eigenbasis, quadrature scores, FVE or fixed component
  selection, optional per-curve presmoothing, prediction, and reconstruction.
  The dense default component cap is `max(1, min(length(grid)-2, nobs(data)-2))`,
  following the fdapace source audit and replacing the plan's cap of 20.
  Source-audited fdapace fixtures test the documented eigenvalue convention.
  PACE conditional-expectation scores and AIC/BIC selection are next.
  Cross-version tests and the docs build remain controller gates.
- Milestone 4: multivariate FPCA; FunctionalGraphicalModels.jl migrates.
- Milestone 5: registration, functional regression, depth and outliers,
  plotting recipes.
