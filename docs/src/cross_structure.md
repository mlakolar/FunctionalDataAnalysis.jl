# Cross-structure

These tools summarize dependence between two processes measured on the same
subjects. They complement a conditional-dependence graph: cross-covariance,
concurrent correlation and dynamic correlation are marginal summaries.
Subject counts must agree, and available subject IDs must match in order.

## Functional singular components

```julia
using FunctionalDataAnalysis
import StatsAPI

weather = load_dataset(:canadian_weather; sample=true).data
temperature = component(weather, 1)
precipitation = component(weather, 2)
f = StatsAPI.fit(FSVD, temperature, precipitation; ncomponents=1)
f.singular_values
f.left_functions, f.right_functions
f.left_scores, f.right_scores
f.fve[1:1]
```

FSVD decomposes the cross-covariance operator. The two processes may have
unequal domains and nonuniform grids. With stored diagonal quadrature matrices
`W1,W2`, the package computes `svd(sqrt(W1)*C*sqrt(W2))`, then divides the left
and right vectors by their respective square-root weights. Each set of
functions is orthonormal under its stored quadrature. The largest absolute
entry of each left function is positive; the paired right function and scores
use the same sign. Repeated singular values identify subspaces rather than
unique functions.

Complete grid pairs use sample means and the fdapace FSVD divisor **n−2**.
This differs from standalone `CrossCovariance`, whose divisor is **n−1**.
The default component cap is `min(20,n-2)` and default selection is
`FVE(0.9999)`, applied to cumulative **squared** singular values within the cap.
A fixed integer disables FVE selection and throws if the retained rank is too
small. Values at most `1e-10` of the largest are discarded. At least three
subjects are required. `f.fve` retains the full cumulative curve within the
cap; `f.selection` records the chosen count and criterion.

If either node is irregular or masked, both marginal fits use PACE, including
a complete-grid partner, and the cross surface uses `SmoothedCrossCovariance`.
This provides marginal measurement-error variances for conditional scores.
`method=:auto` is resolved to `:pace`; explicit `method=:dense` is rejected.
`settings.fpca_options1/2` retain the requested options and
`settings.effective_fpca_options1/2` record the options passed to FPCA.
Supply `bandwidth` as a number, a pair of axis bandwidths, or explicit-candidate
`CurveCV`. The Gaussian cross-smoothing
kernel is the default. Equal-time pairs are retained, assuming independent
measurement errors between nodes. The marginal means, covariances and noise
variances come from the existing `FPCA` interface; pass named tuples through
`fpca_options1` and `fpca_options2` to change their tuning or supply inputs.
Dense presmoothing must be performed explicitly on the input data.

Dense singular scores are quadrature integrals of centered curves. Sparse
scores condition jointly on both observed processes. The stacked marginal and
cross-covariance surface is projected to be positive semidefinite using the
existing eigen step. Marginal noise variances are then added before linear
interpolation to observation times. The conditional solver requires a positive
definite observation covariance; supply positive marginal noise variances if
needed. No extra regularization or `rho` is introduced. All fits, tuning,
component selection and numerical conventions are retained on the result.

## Concurrent correlation

```julia
cc = concurrent_correlation(temperature, precipitation; bandwidth=30.0)
cc.grid, cc.correlation
cc.affected1, cc.affected2
```

This estimates `corr(X(t),Y(t))` for simultaneous observations. Observation
times must agree within every subject. The estimator assumes no measurement
error and uses five local linear smooths: two means at pooled observed times,
then two squared residuals and one residual cross-product at the output grid.
All five smooths use KernelSmoothing with observation weights one.

Bandwidth is required, as in fdapace `FCCor`. Supply one positive number or
one/five numbers in a vector or tuple. Five numbers are ordered as mean X,
mean Y, variance X, variance Y, cross-product. The actual source default kernel
is **Epanechnikov**. Output defaults to sorted unique observation times;
`grid` may select another increasing grid inside that range. A nonpositive
variance estimate is replaced by NaN with a warning. `affected1` and
`affected2` retain the affected masks. Correlations are not clipped to [−1,1].
The result also stores all five bandwidths, kernel, means, variances,
covariance and their grids.

## Dynamic correlation

```julia
dc = dynamic_correlation(temperature, precipitation)
dc.correlation             # average over stations
dc.individual              # one value per station
```

`dynamic_correlation` requires complete curves on the same grid with matching
weights. For each process it subtracts each curve's time average, subtracts
the population mean of the resulting curves, and scales each residual curve
to time RMS one. Each individual correlation is the time-averaged inner
product of the paired standardized curves. The reported correlation averages
those values over subjects. It describes synchrony of residual trajectories,
not pointwise association or conditional independence.

Stored quadrature weights are divided by the domain length for every time
average. There are no bandwidth or component defaults. Missing values require
an explicit conversion before fitting; they are not silently interpolated as
in R. Zero residual RMS is undefined and throws. The result stores individual
correlations, standardized curves, grid, weights and centering conventions.

## Reference validation

`test/reference/fdapace_m6.R` uses **fdapace 0.6.0**. Run it from the repository
root with `Rscript test/reference/fdapace_m6.R`; CI reads its committed CSVs
without running R. Headers record the generating R and package versions.
Tests require relative errors below `1e-8` for singular values, both functions,
both score matrices, cumulative fractions, the concurrent curve and individual
dynamic correlations. Largest-entry-positive left-function signs are used in
both languages.

The dense fixture has singular functions that vanish at the endpoints, so
fdapace's rectangular operator and the weighted operator agree. Its values,
functions and scores are compared directly with native `FSVD`. Sparse inputs
reuse the milestone 4 cross-covariance fixture with fixed bandwidths and
supplied marginal means, covariances and noise. The native sparse `FSVD`
results are preserved as reference-only `m6_sparse_native_*.csv`, which Julia
tests do not read. Since the discretizations differ, the compared sparse
quantity is an independent R weighted SVD of the
native cross surface, followed by joint conditional scores with weighted
marginal integrals. The R script spells out that calculation. It retains the
reference's unweighted PSD projection of the stacked surface. FCCor and
DynCorr are compared directly with native outputs. No optimizer agreement or
benchmark claim is inferred from these fixed-tuning comparisons.

The [fdapace source audit](fdapace.md#Cross-structure-source-audit) details
defaults, source discrepancies and deliberate departures.

```@docs
FSVD
concurrent_correlation
dynamic_correlation
```
