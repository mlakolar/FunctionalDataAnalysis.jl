# Scalar-on-function regression study

Every configured replicate is retained for every route; failures are counted and never replaced. Quality summaries are conditional on success. Errors are relative: integrated squared coefficient error over the scoring grid divided by that of the true coefficient (square root), and held-out squared prediction error against the noise-free mean response divided by the signal variance. Coverage is that of nominal 95% pointwise bands; "whole curve" is the fraction of replicates in which the pointwise band covers the coefficient at every grid point (no simultaneous band is offered). Complexity is the retained component count (FPC) or the effective degrees of freedom of the functional term (penalized). Peak RSS is the cumulative route-worker process peak.

## Accuracy and coverage (medians of relative errors)

| Cell | Design | Beta | n | R² | Route | OK | Failures | Beta error (median) | Beta error (mean) | Prediction error (median) | Pointwise coverage | Whole curve | Band width | Complexity |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dense | leading | 40 | 0.8 | fpc | 2/2 | 0 | 4.01 | 4.01 | 0.133 | 0.99 | 0 | 27.5 | 22.5 |
| 1 | dense | leading | 40 | 0.8 | fpc_bic | 2/2 | 0 | 0.324 | 0.324 | 0.0509 | 0.678 | 0 | 0.928 | 2.5 |
| 1 | dense | leading | 40 | 0.8 | pfr | 2/2 | 0 | 0.573 | 0.573 | 0.0563 | 0.861 | 0 | 1.84 | 2.46 |
| 2 | sparse | smooth | 40 | 0.5 | fpc | 2/2 | 0 | 3.42 | 3.42 | 0.459 | 0.653 | 0 | 3.16 | 3.5 |
| 2 | sparse | smooth | 40 | 0.5 | fpc_bic | 2/2 | 0 | 2.53 | 2.53 | 0.428 | 0.134 | 0 | 1.19 | 2 |
| 2 | sparse | smooth | 40 | 0.5 | pfr | 2/2 | 0 | 2.78 | 2.78 | 0.393 | 0.802 | 0 | 3.94 | 2.7 |

## Resources (all recorded attempts)

| Cell | Design | n | Route | Wall seconds (mean) | Allocated MB (mean) | Max process peak RSS MB |
|---|---|---:|---|---:|---:|---:|
| 1 | dense | 40 | fpc | 0.00136 | 1.1 | 1.26e+03 |
| 1 | dense | 40 | fpc_bic | 0.00378 | 5.58 | 1.19e+03 |
| 1 | dense | 40 | pfr | 0.00186 | 1.28 | 1.21e+03 |
| 2 | sparse | 40 | fpc | 0.0403 | 4.68 | 1.16e+03 |
| 2 | sparse | 40 | fpc_bic | 0.0436 | 6.28 | 1.19e+03 |
| 2 | sparse | 40 | pfr | 0.0568 | 5.56 | 1.09e+03 |
