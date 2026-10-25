# Long-grid FPCA benchmark

Each configured replicate is retained for both routes. Failures are counted and are never replaced. Accuracy and component summaries are conditional on success. Resource summaries include recorded failed attempts; their CSV counts show measurement coverage. Peak RSS is the maximum cumulative route-worker process peak, including startup, JIT, simulation, fitting, and metrics. Dense FPCA has no noise-variance estimate; its sigma2-error count is zero. SD uses divisor R-1 and is undefined for one observation.

| Cell | Design | n | m | Ratio | Route | Successes | Failures | Wall seconds | Allocated bytes | Max process peak RSS bytes | Eigenvalue error | Eigenfunction L2 error | Sigma2 error | Components | Reconstruction error |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dense | 12 | 40 | 0.05 | dense | 1/1 | 0 | 0.00024875 | 195168 | 8.6786048e+08 | 0.22264661 | 0.55008897 | nan | 6 | 0.062279624 |
| 1 | dense | 12 | 40 | 0.05 | face | 1/1 | 0 | 0.003732083 | 6031360 | 8.5429453e+08 | 0.1039376 | 0.5420522 | 0.025921393 | 3 | 0.011159038 |
| 2 | sparse | 12 | 21 | 0.2 | pace | 1/1 | 0 | 0.004262292 | 904864 | 8.2046157e+08 | 0.4007349 | 0.810701 | 0.036516394 | 4 | 0.70624067 |
| 2 | sparse | 12 | 21 | 0.2 | face_sparse | 1/1 | 0 | 0.001973584 | 2480352 | 8.5175501e+08 | 0.52880627 | 0.67658569 | 0.21135406 | 1 | 0.71352062 |
