# Outlier detection and clustering study

Every configured replicate is retained for every route; failures are counted and never replaced. Quality summaries are conditional on success. True and false positive rates are the fractions of true outliers and of typical curves flagged; ARI is the adjusted Rand index of the partition against the true groups. "Not converged" counts successful fits whose recorded status is not `converged`. Peak RSS is the cumulative route-worker process peak.

## Outlier detection

| Cell | Model | Outlier type | Contamination | Design | n | Route | OK | Failures | TPR (mean) | FPR (mean) | Flagged (mean) | Seconds (mean) |
|---:|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | model6 | mixed | 0.1 | sparse | 40 | boxplot | 2/2 | 0 | 0.125 | 0 | 0.5 | 0.0446 |
| 1 | model6 | mixed | 0.1 | sparse | 40 | outliergram | 2/2 | 0 | 0.75 | 0.0278 | 4 | 0.0455 |
| 1 | model6 | mixed | 0.1 | sparse | 40 | msplot | 2/2 | 0 | 0.875 | 0.0417 | 5 | 0.0524 |

## Clustering

| Cell | Groups differ in | Groups | Design | n | Route | OK | Failures | Not converged | ARI (median) | ARI (mean) | ARI (SD) | Seconds (mean) |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | covariance | 2 | dense | 40 | kmeans_curves | 2/2 | 0 | 0 | 0.0338 | 0.0338 | 0.0255 | 5.61e-05 |
| 2 | covariance | 2 | dense | 40 | kmeans_scores | 2/2 | 0 | 0 | 0.0182 | 0.0182 | 0.035 | 0.00026 |
| 2 | covariance | 2 | dense | 40 | kcfc | 2/2 | 0 | 0 | 0.0343 | 0.0343 | 0.0617 | 0.000986 |
| 2 | covariance | 2 | dense | 40 | mixture | 2/2 | 0 | 0 | 0.514 | 0.514 | 0.0518 | 0.00357 |

## Resources (all recorded attempts)

| Cell | Study | Design | n | Route | Wall seconds (mean) | Allocated MB (mean) | Max process peak RSS MB |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | outlier | sparse | 40 | boxplot | 0.0446 | 4.84 | 1.23e+03 |
| 1 | outlier | sparse | 40 | outliergram | 0.0455 | 4.95 | 1.18e+03 |
| 1 | outlier | sparse | 40 | msplot | 0.0524 | 39.9 | 1.29e+03 |
| 2 | clustering | dense | 40 | kmeans_curves | 5.61e-05 | 0.12 | 1.29e+03 |
| 2 | clustering | dense | 40 | kmeans_scores | 0.00026 | 0.308 | 1.32e+03 |
| 2 | clustering | dense | 40 | kcfc | 0.000986 | 2.28 | 1.4e+03 |
| 2 | clustering | dense | 40 | mixture | 0.00357 | 10.7 | 1.27e+03 |
