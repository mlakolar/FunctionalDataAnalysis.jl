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
  middle-half integrated noise variance (diagonal bandwidth equals the selected
  covariance bandwidth). Both weighting schemes match fixed-bandwidth fdapace
  covariance fixtures. Full cross-version tests and docs remain controller gates.
  Pointwise mean bands await cluster-robust variance support in KernelSmoothing.jl.
- Milestone 3: FPCA, dense and sparse (PACE), validated against fdapace.
- Milestone 4: multivariate FPCA; FunctionalGraphicalModels.jl migrates.
- Milestone 5: registration, functional regression, depth and outliers,
  plotting recipes.
