# Noise variance benchmark

Bias and MSE use the truncated estimate. Variance uses divisor R-1. If fits fail, moments are conditional on success; failures are shown. No failed replicate is replaced.

| Cell | Design | n | Ratio | Weight | Bandwidth | Estimator | Successes | Failures | Truncated | Bias | Variance | MSE |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | sparse | 100 | 0.05 | obs | default | surface_middle_half | 20/20 | 0 | 0 | 0.29944197 | 0.015985618 | 0.10485183 |
| 1 | sparse | 100 | 0.05 | obs | default | surface_full | 20/20 | 0 | 0 | 0.26735865 | 0.014494808 | 0.085250716 |
| 1 | sparse | 100 | 0.05 | obs | default | rotated_middle_half | 20/20 | 0 | 2 | 0.14722968 | 0.020957519 | 0.041586222 |
| 1 | sparse | 100 | 0.05 | obs | default | rotated_full | 20/20 | 0 | 1 | 0.13971713 | 0.019763211 | 0.038295927 |
| 1 | sparse | 100 | 0.05 | obs | default | projected_fgm | 20/20 | 0 | 0 | 0.2382034 | 0.014314953 | 0.070340067 |
| 2 | dense | 100 | 0.2 | subj | cv | surface_middle_half | 20/20 | 0 | 0 | 0.08253775 | 0.0001514477 | 0.0069563555 |
| 2 | dense | 100 | 0.2 | subj | cv | surface_full | 20/20 | 0 | 0 | 0.073739356 | 0.00013757531 | 0.0055681892 |
| 2 | dense | 100 | 0.2 | subj | cv | rotated_middle_half | 20/20 | 0 | 0 | -0.0079882555 | 0.00013099662 | 0.00018825901 |
| 2 | dense | 100 | 0.2 | subj | cv | rotated_full | 20/20 | 0 | 0 | 0.0035653155 | 7.3695369e-05 | 8.2722075e-05 |
| 2 | dense | 100 | 0.2 | subj | cv | projected_fgm | 20/20 | 0 | 0 | 0.071959639 | 0.00013463372 | 0.0053060917 |
