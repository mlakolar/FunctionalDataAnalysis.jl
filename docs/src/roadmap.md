# Roadmap

- Milestone 0: package skeleton, CI, docs.
- Milestone 1 (done): data types (grid, irregular, basis, multivariate), conversions,
  Tables.jl input, simulation generators.
- Milestone 2 (in progress): per-curve local polynomials and penalized splines
  are implemented. Smoothing returns grid or basis data with a copied `metadata`
  named tuple retaining tuning settings. Common-lambda GCV stores the candidate
  grid, summed per-curve scores, and selected lambda in `metadata.selection`.
  `StatsAPI.fit` entry points return the same functional data objects.
  Remaining tasks: pooled mean, covariance surface,
  noise variance, and curve-level bandwidth selection on KernelSmoothing.jl.
- Milestone 3: FPCA, dense and sparse (PACE), validated against fdapace.
- Milestone 4: multivariate FPCA; FunctionalGraphicalModels.jl migrates.
- Milestone 5: registration, functional regression, depth and outliers,
  plotting recipes.
