# Outlier detection and clustering study

Every configured replicate is retained for every route; failures are counted and never replaced. Quality summaries are conditional on success. True and false positive rates are the fractions of true outliers and of typical curves flagged; ARI is the adjusted Rand index of the partition against the true groups. "Not converged" counts successful fits whose recorded status is not `converged`. Peak RSS is the cumulative route-worker process peak.

## Outlier detection

| Cell | Model | Outlier type | Contamination | Design | n | Route | OK | Failures | TPR (mean) | FPR (mean) | Flagged (mean) | Seconds (mean) |
|---:|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | model1 | shift | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 1 | 0.000421 | 5.04 | 0.0006 |
| 1 | model1 | shift | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0 | 0.0448 | 4.26 | 0.00157 |
| 1 | model1 | shift | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0725 | 11.9 | 0.0332 |
| 2 | model1 | shift | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 1 | 0 | 25 | 0.00311 |
| 2 | model1 | shift | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0 | 0.0483 | 23 | 0.0122 |
| 2 | model1 | shift | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.127 | 85.2 | 0.0929 |
| 3 | model1 | shift | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 1 | 0.00532 | 5.5 | 0.238 |
| 3 | model1 | shift | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.003 | 0.0812 | 7.73 | 0.239 |
| 3 | model1 | shift | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.999 | 0.156 | 19.8 | 0.271 |
| 4 | model1 | shift | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 1 | 0.00194 | 25.9 | 1.17 |
| 4 | model1 | shift | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0 | 0.0817 | 38.8 | 1.18 |
| 4 | model1 | shift | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 1 | 0.215 | 127 | 1.31 |
| 5 | model1 | shift | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 1 | 0.000722 | 10.1 | 0.000563 |
| 5 | model1 | shift | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0 | 0.0461 | 4.15 | 0.00173 |
| 5 | model1 | shift | 0.1 | dense | 100 | msplot | 200/200 | 0 | 0.999 | 0.069 | 16.2 | 0.0333 |
| 6 | model1 | shift | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.999 | 0 | 50 | 0.00299 |
| 6 | model1 | shift | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0 | 0.0483 | 21.7 | 0.0142 |
| 6 | model1 | shift | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.113 | 101 | 0.0926 |
| 7 | model1 | shift | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 1 | 0.00511 | 10.5 | 0.237 |
| 7 | model1 | shift | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.0005 | 0.0861 | 7.75 | 0.238 |
| 7 | model1 | shift | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 1 | 0.131 | 21.8 | 0.271 |
| 8 | model1 | shift | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 1 | 0.00136 | 50.6 | 1.17 |
| 8 | model1 | shift | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0 | 0.0903 | 40.7 | 1.18 |
| 8 | model1 | shift | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 1 | 0.201 | 141 | 1.3 |
| 9 | model2 | magnitude | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.56 | 0.000474 | 2.85 | 0.000585 |
| 9 | model2 | magnitude | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.078 | 0.0387 | 4.07 | 0.00205 |
| 9 | model2 | magnitude | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.074 | 12 | 0.0336 |
| 10 | model2 | magnitude | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.39 | 0 | 9.76 | 0.00315 |
| 10 | model2 | magnitude | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.074 | 0.0384 | 20.1 | 0.0153 |
| 10 | model2 | magnitude | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.124 | 83.8 | 0.0959 |
| 11 | model2 | magnitude | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.117 | 0.00305 | 0.875 | 0.236 |
| 11 | model2 | magnitude | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.238 | 0.0659 | 7.45 | 0.237 |
| 11 | model2 | magnitude | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.388 | 0.158 | 16.9 | 0.27 |
| 12 | model2 | magnitude | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0776 | 0.000484 | 2.17 | 1.18 |
| 12 | model2 | magnitude | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.235 | 0.0711 | 39.6 | 1.18 |
| 12 | model2 | magnitude | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.454 | 0.225 | 118 | 1.31 |
| 13 | model2 | magnitude | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.489 | 0.000722 | 4.95 | 0.00058 |
| 13 | model2 | magnitude | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.0725 | 0.0364 | 4 | 0.00208 |
| 13 | model2 | magnitude | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0682 | 16.1 | 0.0329 |
| 14 | model2 | magnitude | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.24 | 0 | 12 | 0.0031 |
| 14 | model2 | magnitude | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.0661 | 0.0335 | 18.4 | 0.0154 |
| 14 | model2 | magnitude | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.112 | 100 | 0.0936 |
| 15 | model2 | magnitude | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.114 | 0.00294 | 1.41 | 0.235 |
| 15 | model2 | magnitude | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.22 | 0.0656 | 8.1 | 0.241 |
| 15 | model2 | magnitude | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.407 | 0.158 | 18.3 | 0.269 |
| 16 | model2 | magnitude | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.0787 | 0.000489 | 4.16 | 1.17 |
| 16 | model2 | magnitude | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.218 | 0.0714 | 43 | 1.18 |
| 16 | model2 | magnitude | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.467 | 0.232 | 128 | 1.3 |
| 17 | model3 | magnitude | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.895 | 0.00105 | 4.58 | 0.000615 |
| 17 | model3 | magnitude | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.483 | 0.0374 | 5.96 | 0.00173 |
| 17 | model3 | magnitude | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0765 | 12.3 | 0.0359 |
| 18 | model3 | magnitude | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.73 | 0 | 18.2 | 0.0031 |
| 18 | model3 | magnitude | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.468 | 0.0372 | 29.4 | 0.0126 |
| 18 | model3 | magnitude | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.124 | 84.1 | 0.0988 |
| 19 | model3 | magnitude | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.747 | 0.00232 | 3.96 | 0.237 |
| 19 | model3 | magnitude | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.421 | 0.0584 | 7.65 | 0.237 |
| 19 | model3 | magnitude | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.939 | 0.142 | 18.2 | 0.268 |
| 20 | model3 | magnitude | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.711 | 0.000253 | 17.9 | 1.17 |
| 20 | model3 | magnitude | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.448 | 0.0564 | 38 | 1.18 |
| 20 | model3 | magnitude | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.947 | 0.196 | 117 | 1.31 |
| 21 | model3 | magnitude | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.874 | 0.0005 | 8.79 | 0.000608 |
| 21 | model3 | magnitude | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.434 | 0.0296 | 7 | 0.0017 |
| 21 | model3 | magnitude | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0716 | 16.4 | 0.0354 |
| 22 | model3 | magnitude | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.661 | 0 | 33.1 | 0.00299 |
| 22 | model3 | magnitude | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.449 | 0.0276 | 34.9 | 0.0141 |
| 22 | model3 | magnitude | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.115 | 102 | 0.091 |
| 23 | model3 | magnitude | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.748 | 0.00178 | 7.64 | 0.236 |
| 23 | model3 | magnitude | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.416 | 0.0479 | 8.47 | 0.237 |
| 23 | model3 | magnitude | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.945 | 0.125 | 20.7 | 0.267 |
| 24 | model3 | magnitude | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.656 | 0.0002 | 32.9 | 1.17 |
| 24 | model3 | magnitude | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.437 | 0.0439 | 41.6 | 1.18 |
| 24 | model3 | magnitude | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.948 | 0.179 | 128 | 1.29 |
| 25 | model4 | shape | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.196 | 5.26e-05 | 0.985 | 0.000608 |
| 25 | model4 | shape | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.996 | 0.0181 | 6.7 | 0.00197 |
| 25 | model4 | shape | 0.05 | dense | 100 | msplot | 200/200 | 0 | 0.999 | 0.0508 | 9.82 | 0.0341 |
| 26 | model4 | shape | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.0014 | 0 | 0.035 | 0.00291 |
| 26 | model4 | shape | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.995 | 0.0149 | 31.9 | 0.0136 |
| 26 | model4 | shape | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.0894 | 67.5 | 0.0916 |
| 27 | model4 | shape | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.29 | 0.000474 | 1.5 | 0.238 |
| 27 | model4 | shape | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.948 | 0.022 | 6.83 | 0.241 |
| 27 | model4 | shape | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.998 | 0.106 | 15.1 | 0.276 |
| 28 | model4 | shape | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0728 | 0 | 1.82 | 1.17 |
| 28 | model4 | shape | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.973 | 0.0196 | 33.6 | 1.18 |
| 28 | model4 | shape | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 1 | 0.157 | 99.5 | 1.29 |
| 29 | model4 | shape | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.122 | 0 | 1.22 | 0.000589 |
| 29 | model4 | shape | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.956 | 0.0122 | 10.7 | 0.00192 |
| 29 | model4 | shape | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0427 | 13.8 | 0.0351 |
| 30 | model4 | shape | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.0011 | 0 | 0.055 | 0.00309 |
| 30 | model4 | shape | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.967 | 0.00982 | 52.8 | 0.0141 |
| 30 | model4 | shape | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.0782 | 85.2 | 0.0959 |
| 31 | model4 | shape | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.246 | 0.000111 | 2.46 | 0.237 |
| 31 | model4 | shape | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.855 | 0.0128 | 9.71 | 0.241 |
| 31 | model4 | shape | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 1 | 0.101 | 19.1 | 0.268 |
| 32 | model4 | shape | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.034 | 0 | 1.7 | 1.17 |
| 32 | model4 | shape | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.915 | 0.0116 | 51 | 1.18 |
| 32 | model4 | shape | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.999 | 0.144 | 115 | 1.29 |
| 33 | model5 | shape | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.478 | 0.000842 | 2.47 | 0.00059 |
| 33 | model5 | shape | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.808 | 0.0312 | 7 | 0.00189 |
| 33 | model5 | shape | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0777 | 12.4 | 0.0331 |
| 34 | model5 | shape | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.19 | 0 | 4.74 | 0.00297 |
| 34 | model5 | shape | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.819 | 0.0317 | 35.6 | 0.0107 |
| 34 | model5 | shape | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.127 | 85.5 | 0.0948 |
| 35 | model5 | shape | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.133 | 0.00253 | 0.905 | 0.237 |
| 35 | model5 | shape | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.457 | 0.0574 | 7.74 | 0.238 |
| 35 | model5 | shape | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.795 | 0.136 | 16.9 | 0.27 |
| 36 | model5 | shape | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0668 | 0.000232 | 1.78 | 1.17 |
| 36 | model5 | shape | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.45 | 0.0576 | 38.6 | 1.18 |
| 36 | model5 | shape | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.85 | 0.199 | 116 | 1.3 |
| 37 | model5 | shape | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.466 | 0.000389 | 4.7 | 0.000582 |
| 37 | model5 | shape | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.754 | 0.0207 | 9.4 | 0.00176 |
| 37 | model5 | shape | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0674 | 16.1 | 0.0329 |
| 38 | model5 | shape | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.164 | 1.11e-05 | 8.23 | 0.00309 |
| 38 | model5 | shape | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.784 | 0.0197 | 48 | 0.0102 |
| 38 | model5 | shape | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.113 | 101 | 0.0988 |
| 39 | model5 | shape | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.133 | 0.00228 | 1.53 | 0.237 |
| 39 | model5 | shape | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.447 | 0.0491 | 8.88 | 0.237 |
| 39 | model5 | shape | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.772 | 0.13 | 19.4 | 0.269 |
| 40 | model5 | shape | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.0555 | 0.0002 | 2.87 | 1.17 |
| 40 | model5 | shape | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.419 | 0.0493 | 43.1 | 1.18 |
| 40 | model5 | shape | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.821 | 0.189 | 126 | 1.3 |
| 41 | model6 | mixed | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.045 | 0.000474 | 0.27 | 0.000579 |
| 41 | model6 | mixed | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.868 | 0.0333 | 7.5 | 0.00187 |
| 41 | model6 | mixed | 0.05 | dense | 100 | msplot | 200/200 | 0 | 0.999 | 0.0837 | 12.9 | 0.0342 |
| 42 | model6 | mixed | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.0016 | 0 | 0.04 | 0.00309 |
| 42 | model6 | mixed | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.869 | 0.0301 | 36 | 0.0126 |
| 42 | model6 | mixed | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.125 | 84.2 | 0.0934 |
| 43 | model6 | mixed | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.041 | 0.00179 | 0.375 | 0.236 |
| 43 | model6 | mixed | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.568 | 0.0535 | 7.92 | 0.237 |
| 43 | model6 | mixed | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.833 | 0.122 | 15.8 | 0.27 |
| 44 | model6 | mixed | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0044 | 2.11e-05 | 0.12 | 1.18 |
| 44 | model6 | mixed | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.607 | 0.0462 | 37.1 | 1.18 |
| 44 | model6 | mixed | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.9 | 0.169 | 103 | 1.27 |
| 45 | model6 | mixed | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.033 | 0.000167 | 0.345 | 0.000609 |
| 45 | model6 | mixed | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.81 | 0.0192 | 9.83 | 0.00172 |
| 45 | model6 | mixed | 0.1 | dense | 100 | msplot | 200/200 | 0 | 0.998 | 0.0679 | 16.1 | 0.0337 |
| 46 | model6 | mixed | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.0008 | 0 | 0.04 | 0.00319 |
| 46 | model6 | mixed | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.833 | 0.0187 | 50.1 | 0.0108 |
| 46 | model6 | mixed | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.115 | 102 | 0.096 |
| 47 | model6 | mixed | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.0335 | 0.001 | 0.425 | 0.236 |
| 47 | model6 | mixed | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.565 | 0.0354 | 8.84 | 0.238 |
| 47 | model6 | mixed | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.838 | 0.0947 | 16.9 | 0.268 |
| 48 | model6 | mixed | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.0034 | 1.11e-05 | 0.175 | 1.17 |
| 48 | model6 | mixed | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.589 | 0.0349 | 45.2 | 1.19 |
| 48 | model6 | mixed | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.904 | 0.147 | 112 | 1.3 |
| 49 | model7 | shape | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.047 | 0.000421 | 0.275 | 0.000588 |
| 49 | model7 | shape | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.919 | 0.0339 | 7.82 | 0.00193 |
| 49 | model7 | shape | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0798 | 12.6 | 0.0336 |
| 50 | model7 | shape | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.0018 | 1.05e-05 | 0.05 | 0.00318 |
| 50 | model7 | shape | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.935 | 0.0307 | 38 | 0.0127 |
| 50 | model7 | shape | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.124 | 83.7 | 0.0963 |
| 51 | model7 | shape | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.035 | 0.00242 | 0.405 | 0.238 |
| 51 | model7 | shape | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.607 | 0.0503 | 7.81 | 0.238 |
| 51 | model7 | shape | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.872 | 0.126 | 16.3 | 0.27 |
| 52 | model7 | shape | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0036 | 0.000242 | 0.205 | 1.17 |
| 52 | model7 | shape | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.627 | 0.0515 | 40.2 | 1.18 |
| 52 | model7 | shape | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.938 | 0.191 | 114 | 1.3 |
| 53 | model7 | shape | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.0355 | 0.000444 | 0.395 | 0.000616 |
| 53 | model7 | shape | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.887 | 0.0201 | 10.7 | 0.00176 |
| 53 | model7 | shape | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0732 | 16.6 | 0.0328 |
| 54 | model7 | shape | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.0008 | 1.11e-05 | 0.045 | 0.00316 |
| 54 | model7 | shape | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.9 | 0.0174 | 52.9 | 0.0104 |
| 54 | model7 | shape | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.112 | 100 | 0.0947 |
| 55 | model7 | shape | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.023 | 0.00128 | 0.345 | 0.237 |
| 55 | model7 | shape | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.628 | 0.0321 | 9.17 | 0.239 |
| 55 | model7 | shape | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.887 | 0.11 | 18.8 | 0.269 |
| 56 | model7 | shape | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.002 | 0.000178 | 0.18 | 1.17 |
| 56 | model7 | shape | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.623 | 0.0336 | 46.3 | 1.18 |
| 56 | model7 | shape | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.954 | 0.177 | 127 | 1.27 |
| 57 | model8 | shape | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0.537 | 0.000947 | 2.77 | 0.000585 |
| 57 | model8 | shape | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.998 | 0.0299 | 7.83 | 0.00198 |
| 57 | model8 | shape | 0.05 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.0761 | 12.2 | 0.0332 |
| 58 | model8 | shape | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0.0716 | 0 | 1.79 | 0.0032 |
| 58 | model8 | shape | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.999 | 0.0295 | 39 | 0.016 |
| 58 | model8 | shape | 0.05 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.124 | 83.9 | 0.0965 |
| 59 | model8 | shape | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.009 | 0.00521 | 0.54 | 0.236 |
| 59 | model8 | shape | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.13 | 0.0731 | 7.59 | 0.238 |
| 59 | model8 | shape | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.244 | 0.156 | 16 | 0.27 |
| 60 | model8 | shape | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0.0022 | 0.00175 | 0.885 | 1.17 |
| 60 | model8 | shape | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.129 | 0.0806 | 41.5 | 1.18 |
| 60 | model8 | shape | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.341 | 0.245 | 125 | 1.3 |
| 61 | model8 | shape | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.495 | 0.000167 | 4.96 | 0.000596 |
| 61 | model8 | shape | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.998 | 0.0193 | 11.7 | 0.00198 |
| 61 | model8 | shape | 0.1 | dense | 100 | msplot | 200/200 | 0 | 1 | 0.07 | 16.3 | 0.0329 |
| 62 | model8 | shape | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0.0526 | 0 | 2.63 | 0.00318 |
| 62 | model8 | shape | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.996 | 0.0168 | 57.4 | 0.0161 |
| 62 | model8 | shape | 0.1 | dense | 500 | msplot | 200/200 | 0 | 1 | 0.113 | 101 | 0.0916 |
| 63 | model8 | shape | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.0095 | 0.00544 | 0.585 | 0.237 |
| 63 | model8 | shape | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.123 | 0.0683 | 7.38 | 0.238 |
| 63 | model8 | shape | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.257 | 0.165 | 17.4 | 0.268 |
| 64 | model8 | shape | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0.003 | 0.00184 | 0.98 | 1.17 |
| 64 | model8 | shape | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.121 | 0.0739 | 39.3 | 1.18 |
| 64 | model8 | shape | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.327 | 0.234 | 121 | 1.3 |
| 65 | model9 | amplitude | 0.05 | dense | 100 | boxplot | 200/200 | 0 | 0 | 0 | 0 | 0.000581 |
| 65 | model9 | amplitude | 0.05 | dense | 100 | outliergram | 200/200 | 0 | 0.526 | 0.000158 | 2.65 | 0.00144 |
| 65 | model9 | amplitude | 0.05 | dense | 100 | msplot | 200/200 | 0 | 0.952 | 0.0139 | 6.08 | 0.0331 |
| 66 | model9 | amplitude | 0.05 | dense | 500 | boxplot | 200/200 | 0 | 0 | 0 | 0 | 0.00301 |
| 66 | model9 | amplitude | 0.05 | dense | 500 | outliergram | 200/200 | 0 | 0.568 | 4.21e-05 | 14.2 | 0.00909 |
| 66 | model9 | amplitude | 0.05 | dense | 500 | msplot | 200/200 | 0 | 0.996 | 0.0191 | 34 | 0.0965 |
| 67 | model9 | amplitude | 0.05 | sparse | 100 | boxplot | 200/200 | 0 | 0.009 | 0 | 0.045 | 0.238 |
| 67 | model9 | amplitude | 0.05 | sparse | 100 | outliergram | 200/200 | 0 | 0.352 | 0.000158 | 1.77 | 0.239 |
| 67 | model9 | amplitude | 0.05 | sparse | 100 | msplot | 200/200 | 0 | 0.926 | 0.0172 | 6.26 | 0.27 |
| 68 | model9 | amplitude | 0.05 | sparse | 500 | boxplot | 200/200 | 0 | 0 | 0 | 0 | 1.17 |
| 68 | model9 | amplitude | 0.05 | sparse | 500 | outliergram | 200/200 | 0 | 0.315 | 0 | 7.88 | 1.17 |
| 68 | model9 | amplitude | 0.05 | sparse | 500 | msplot | 200/200 | 0 | 0.991 | 0.0321 | 40 | 1.3 |
| 69 | model9 | amplitude | 0.1 | dense | 100 | boxplot | 200/200 | 0 | 0.002 | 0 | 0.02 | 0.000591 |
| 69 | model9 | amplitude | 0.1 | dense | 100 | outliergram | 200/200 | 0 | 0.328 | 0 | 3.28 | 0.00139 |
| 69 | model9 | amplitude | 0.1 | dense | 100 | msplot | 200/200 | 0 | 0.91 | 0.00978 | 9.98 | 0.0345 |
| 70 | model9 | amplitude | 0.1 | dense | 500 | boxplot | 200/200 | 0 | 0 | 0 | 0 | 0.00305 |
| 70 | model9 | amplitude | 0.1 | dense | 500 | outliergram | 200/200 | 0 | 0.322 | 0 | 16.1 | 0.00887 |
| 70 | model9 | amplitude | 0.1 | dense | 500 | msplot | 200/200 | 0 | 0.992 | 0.0162 | 56.9 | 0.0947 |
| 71 | model9 | amplitude | 0.1 | sparse | 100 | boxplot | 200/200 | 0 | 0.004 | 0 | 0.04 | 0.239 |
| 71 | model9 | amplitude | 0.1 | sparse | 100 | outliergram | 200/200 | 0 | 0.222 | 0 | 2.22 | 0.24 |
| 71 | model9 | amplitude | 0.1 | sparse | 100 | msplot | 200/200 | 0 | 0.912 | 0.0133 | 10.3 | 0.27 |
| 72 | model9 | amplitude | 0.1 | sparse | 500 | boxplot | 200/200 | 0 | 0 | 0 | 0 | 1.17 |
| 72 | model9 | amplitude | 0.1 | sparse | 500 | outliergram | 200/200 | 0 | 0.171 | 0 | 8.54 | 1.18 |
| 72 | model9 | amplitude | 0.1 | sparse | 500 | msplot | 200/200 | 0 | 0.983 | 0.0229 | 59.4 | 1.3 |

## Clustering

| Cell | Groups differ in | Groups | Design | n | Route | OK | Failures | Not converged | ARI (median) | ARI (mean) | ARI (SD) | Seconds (mean) |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 73 | mean | 2 | dense | 100 | kmeans_curves | 200/200 | 0 | 0 | 0.605 | 0.615 | 0.0972 | 0.00024 |
| 73 | mean | 2 | dense | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.636 | 0.618 | 0.0979 | 0.000863 |
| 73 | mean | 2 | dense | 100 | kcfc | 200/200 | 0 | 0 | 9.79e-05 | 0.0106 | 0.0272 | 0.0077 |
| 73 | mean | 2 | dense | 100 | mixture | 200/200 | 0 | 0 | 0.225 | 0.283 | 0.254 | 0.011 |
| 74 | mean | 2 | dense | 500 | kmeans_curves | 200/200 | 0 | 0 | 0.62 | 0.621 | 0.0423 | 0.000565 |
| 74 | mean | 2 | dense | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.62 | 0.622 | 0.0417 | 0.000892 |
| 74 | mean | 2 | dense | 500 | kcfc | 200/200 | 0 | 0 | 0.000305 | 0.00218 | 0.00528 | 0.0218 |
| 74 | mean | 2 | dense | 500 | mixture | 200/200 | 0 | 0 | 0.583 | 0.557 | 0.114 | 0.0439 |
| 75 | mean | 2 | sparse | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.604 | 0.599 | 0.107 | 0.237 |
| 75 | mean | 2 | sparse | 100 | kcfc | 200/200 | 0 | 0 | 0.00449 | 0.0208 | 0.0413 | 2.55 |
| 75 | mean | 2 | sparse | 100 | mixture | 200/200 | 0 | 0 | 0.254 | 0.285 | 0.233 | 0.252 |
| 76 | mean | 2 | sparse | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.614 | 0.608 | 0.0446 | 1.17 |
| 76 | mean | 2 | sparse | 500 | kcfc | 200/200 | 0 | 0 | 0.000321 | 0.0037 | 0.00805 | 17.7 |
| 76 | mean | 2 | sparse | 500 | mixture | 200/200 | 0 | 0 | 0.565 | 0.525 | 0.137 | 1.25 |
| 77 | mean | 3 | dense | 100 | kmeans_curves | 200/200 | 0 | 0 | 0.595 | 0.599 | 0.0956 | 0.000264 |
| 77 | mean | 3 | dense | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.609 | 0.607 | 0.0884 | 0.000898 |
| 77 | mean | 3 | dense | 100 | kcfc | 199/200 | 1 | 1 | 0.0164 | 0.025 | 0.0346 | 0.0103 |
| 77 | mean | 3 | dense | 100 | mixture | 200/200 | 0 | 0 | 0.386 | 0.373 | 0.131 | 0.0162 |
| 78 | mean | 3 | dense | 500 | kmeans_curves | 200/200 | 0 | 0 | 0.626 | 0.627 | 0.0348 | 0.000827 |
| 78 | mean | 3 | dense | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.626 | 0.627 | 0.0374 | 0.000932 |
| 78 | mean | 3 | dense | 500 | kcfc | 200/200 | 0 | 0 | 0.00349 | 0.00486 | 0.00662 | 0.0483 |
| 78 | mean | 3 | dense | 500 | mixture | 200/200 | 0 | 0 | 0.524 | 0.513 | 0.0972 | 0.07 |
| 79 | mean | 3 | sparse | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.597 | 0.593 | 0.0884 | 0.236 |
| 79 | mean | 3 | sparse | 100 | kcfc | 200/200 | 0 | 4 | 0.0385 | 0.0546 | 0.0589 | 6.12 |
| 79 | mean | 3 | sparse | 100 | mixture | 200/200 | 0 | 0 | 0.409 | 0.404 | 0.132 | 0.251 |
| 80 | mean | 3 | sparse | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.61 | 0.61 | 0.0355 | 1.17 |
| 80 | mean | 3 | sparse | 500 | kcfc | 200/200 | 0 | 7 | 0.00774 | 0.0114 | 0.0149 | 45.6 |
| 80 | mean | 3 | sparse | 500 | mixture | 200/200 | 0 | 0 | 0.534 | 0.516 | 0.0868 | 1.24 |
| 81 | covariance | 2 | dense | 100 | kmeans_curves | 200/200 | 0 | 0 | 0.00963 | 0.0336 | 0.0544 | 0.000223 |
| 81 | covariance | 2 | dense | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.00468 | 0.036 | 0.0577 | 0.000839 |
| 81 | covariance | 2 | dense | 100 | kcfc | 200/200 | 0 | 24 | 0.000292 | 0.0538 | 0.157 | 0.00425 |
| 81 | covariance | 2 | dense | 100 | mixture | 200/200 | 0 | 0 | 0.543 | 0.534 | 0.103 | 0.0109 |
| 82 | covariance | 2 | dense | 500 | kmeans_curves | 200/200 | 0 | 0 | 0.000705 | 0.0286 | 0.0455 | 0.000628 |
| 82 | covariance | 2 | dense | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.0016 | 0.0339 | 0.0476 | 0.000901 |
| 82 | covariance | 2 | dense | 500 | kcfc | 200/200 | 0 | 1 | 0.000332 | 0.0123 | 0.0538 | 0.0163 |
| 82 | covariance | 2 | dense | 500 | mixture | 200/200 | 0 | 0 | 0.547 | 0.548 | 0.0442 | 0.0419 |
| 83 | covariance | 2 | sparse | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.00493 | 0.0336 | 0.0513 | 0.237 |
| 83 | covariance | 2 | sparse | 100 | kcfc | 200/200 | 0 | 30 | 0.00413 | 0.0889 | 0.174 | 1.9 |
| 83 | covariance | 2 | sparse | 100 | mixture | 200/200 | 0 | 0 | 0.485 | 0.479 | 0.105 | 0.247 |
| 84 | covariance | 2 | sparse | 500 | kmeans_scores | 200/200 | 0 | 0 | -6.72e-05 | 0.02 | 0.0399 | 1.17 |
| 84 | covariance | 2 | sparse | 500 | kcfc | 200/200 | 0 | 4 | 0.00201 | 0.0334 | 0.115 | 19.8 |
| 84 | covariance | 2 | sparse | 500 | mixture | 200/200 | 0 | 0 | 0.512 | 0.513 | 0.0447 | 1.21 |
| 85 | covariance | 3 | dense | 100 | kmeans_curves | 200/200 | 0 | 0 | 0.0579 | 0.063 | 0.0371 | 0.000229 |
| 85 | covariance | 3 | dense | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.0604 | 0.0638 | 0.0377 | 0.000947 |
| 85 | covariance | 3 | dense | 100 | kcfc | 193/200 | 7 | 11 | 0.228 | 0.237 | 0.103 | 0.00806 |
| 85 | covariance | 3 | dense | 100 | mixture | 200/200 | 0 | 0 | 0.446 | 0.441 | 0.124 | 0.0169 |
| 86 | covariance | 3 | dense | 500 | kmeans_curves | 200/200 | 0 | 0 | 0.0646 | 0.0666 | 0.0195 | 0.000892 |
| 86 | covariance | 3 | dense | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.0654 | 0.0674 | 0.0209 | 0.00101 |
| 86 | covariance | 3 | dense | 500 | kcfc | 200/200 | 0 | 0 | 0.241 | 0.256 | 0.0792 | 0.0355 |
| 86 | covariance | 3 | dense | 500 | mixture | 200/200 | 0 | 0 | 0.527 | 0.53 | 0.041 | 0.0584 |
| 87 | covariance | 3 | sparse | 100 | kmeans_scores | 200/200 | 0 | 0 | 0.0554 | 0.0578 | 0.0324 | 0.237 |
| 87 | covariance | 3 | sparse | 100 | kcfc | 191/200 | 9 | 16 | 0.181 | 0.193 | 0.104 | 2.65 |
| 87 | covariance | 3 | sparse | 100 | mixture | 200/200 | 0 | 0 | 0.319 | 0.326 | 0.115 | 0.255 |
| 88 | covariance | 3 | sparse | 500 | kmeans_scores | 200/200 | 0 | 0 | 0.0572 | 0.0584 | 0.019 | 1.17 |
| 88 | covariance | 3 | sparse | 500 | kcfc | 200/200 | 0 | 3 | 0.221 | 0.225 | 0.0608 | 37.3 |
| 88 | covariance | 3 | sparse | 500 | mixture | 200/200 | 0 | 0 | 0.482 | 0.483 | 0.0388 | 1.23 |

## Resources (all recorded attempts)

| Cell | Study | Design | n | Route | Wall seconds (mean) | Allocated MB (mean) | Max process peak RSS MB |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | outlier | dense | 100 | boxplot | 0.0006 | 0.13 | 1.36e+03 |
| 1 | outlier | dense | 100 | outliergram | 0.00157 | 0.769 | 1.36e+03 |
| 1 | outlier | dense | 100 | msplot | 0.0332 | 75.7 | 1.41e+03 |
| 2 | outlier | dense | 500 | boxplot | 0.00311 | 0.707 | 1.36e+03 |
| 2 | outlier | dense | 500 | outliergram | 0.0122 | 7.95 | 1.37e+03 |
| 2 | outlier | dense | 500 | msplot | 0.0929 | 307 | 1.35e+03 |
| 3 | outlier | sparse | 100 | boxplot | 0.238 | 6.18 | 1.37e+03 |
| 3 | outlier | sparse | 100 | outliergram | 0.239 | 6.83 | 1.37e+03 |
| 3 | outlier | sparse | 100 | msplot | 0.271 | 81.4 | 1.34e+03 |
| 4 | outlier | sparse | 500 | boxplot | 1.17 | 15.5 | 1.33e+03 |
| 4 | outlier | sparse | 500 | outliergram | 1.18 | 21.1 | 1.36e+03 |
| 4 | outlier | sparse | 500 | msplot | 1.31 | 321 | 1.38e+03 |
| 5 | outlier | dense | 100 | boxplot | 0.000563 | 0.129 | 1.37e+03 |
| 5 | outlier | dense | 100 | outliergram | 0.00173 | 0.929 | 1.38e+03 |
| 5 | outlier | dense | 100 | msplot | 0.0333 | 75.7 | 1.35e+03 |
| 6 | outlier | dense | 500 | boxplot | 0.00299 | 0.706 | 1.34e+03 |
| 6 | outlier | dense | 500 | outliergram | 0.0142 | 9.36 | 1.36e+03 |
| 6 | outlier | dense | 500 | msplot | 0.0926 | 307 | 1.36e+03 |
| 7 | outlier | sparse | 100 | boxplot | 0.237 | 6.18 | 1.34e+03 |
| 7 | outlier | sparse | 100 | outliergram | 0.238 | 6.93 | 1.34e+03 |
| 7 | outlier | sparse | 100 | msplot | 0.271 | 81.4 | 1.36e+03 |
| 8 | outlier | sparse | 500 | boxplot | 1.17 | 15.5 | 1.36e+03 |
| 8 | outlier | sparse | 500 | outliergram | 1.18 | 21.2 | 1.37e+03 |
| 8 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.37e+03 |
| 9 | outlier | dense | 100 | boxplot | 0.000585 | 0.132 | 1.37e+03 |
| 9 | outlier | dense | 100 | outliergram | 0.00205 | 1.15 | 1.36e+03 |
| 9 | outlier | dense | 100 | msplot | 0.0336 | 75.7 | 1.36e+03 |
| 10 | outlier | dense | 500 | boxplot | 0.00315 | 0.709 | 1.37e+03 |
| 10 | outlier | dense | 500 | outliergram | 0.0153 | 11 | 1.32e+03 |
| 10 | outlier | dense | 500 | msplot | 0.0959 | 307 | 1.36e+03 |
| 11 | outlier | sparse | 100 | boxplot | 0.236 | 6.22 | 1.33e+03 |
| 11 | outlier | sparse | 100 | outliergram | 0.237 | 6.96 | 1.35e+03 |
| 11 | outlier | sparse | 100 | msplot | 0.27 | 81.5 | 1.36e+03 |
| 12 | outlier | sparse | 500 | boxplot | 1.18 | 15.8 | 1.36e+03 |
| 12 | outlier | sparse | 500 | outliergram | 1.18 | 21 | 1.36e+03 |
| 12 | outlier | sparse | 500 | msplot | 1.31 | 321 | 1.38e+03 |
| 13 | outlier | dense | 100 | boxplot | 0.00058 | 0.132 | 1.39e+03 |
| 13 | outlier | dense | 100 | outliergram | 0.00208 | 1.12 | 1.41e+03 |
| 13 | outlier | dense | 100 | msplot | 0.0329 | 75.7 | 1.36e+03 |
| 14 | outlier | dense | 500 | boxplot | 0.0031 | 0.709 | 1.34e+03 |
| 14 | outlier | dense | 500 | outliergram | 0.0154 | 10.3 | 1.38e+03 |
| 14 | outlier | dense | 500 | msplot | 0.0936 | 307 | 1.36e+03 |
| 15 | outlier | sparse | 100 | boxplot | 0.235 | 6.22 | 1.39e+03 |
| 15 | outlier | sparse | 100 | outliergram | 0.241 | 6.94 | 1.33e+03 |
| 15 | outlier | sparse | 100 | msplot | 0.269 | 81.6 | 1.33e+03 |
| 16 | outlier | sparse | 500 | boxplot | 1.17 | 15.9 | 1.32e+03 |
| 16 | outlier | sparse | 500 | outliergram | 1.18 | 20.7 | 1.35e+03 |
| 16 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.36e+03 |
| 17 | outlier | dense | 100 | boxplot | 0.000615 | 0.131 | 1.38e+03 |
| 17 | outlier | dense | 100 | outliergram | 0.00173 | 0.874 | 1.36e+03 |
| 17 | outlier | dense | 100 | msplot | 0.0359 | 75.7 | 1.41e+03 |
| 18 | outlier | dense | 500 | boxplot | 0.0031 | 0.707 | 1.38e+03 |
| 18 | outlier | dense | 500 | outliergram | 0.0126 | 8.07 | 1.39e+03 |
| 18 | outlier | dense | 500 | msplot | 0.0988 | 307 | 1.36e+03 |
| 19 | outlier | sparse | 100 | boxplot | 0.237 | 6.2 | 1.37e+03 |
| 19 | outlier | sparse | 100 | outliergram | 0.237 | 6.81 | 1.36e+03 |
| 19 | outlier | sparse | 100 | msplot | 0.268 | 81.5 | 1.36e+03 |
| 20 | outlier | sparse | 500 | boxplot | 1.17 | 15.7 | 1.34e+03 |
| 20 | outlier | sparse | 500 | outliergram | 1.18 | 21.1 | 1.33e+03 |
| 20 | outlier | sparse | 500 | msplot | 1.31 | 321 | 1.37e+03 |
| 21 | outlier | dense | 100 | boxplot | 0.000608 | 0.13 | 1.38e+03 |
| 21 | outlier | dense | 100 | outliergram | 0.0017 | 0.854 | 1.39e+03 |
| 21 | outlier | dense | 100 | msplot | 0.0354 | 75.6 | 1.41e+03 |
| 22 | outlier | dense | 500 | boxplot | 0.00299 | 0.705 | 1.39e+03 |
| 22 | outlier | dense | 500 | outliergram | 0.0141 | 9.68 | 1.38e+03 |
| 22 | outlier | dense | 500 | msplot | 0.091 | 307 | 1.33e+03 |
| 23 | outlier | sparse | 100 | boxplot | 0.236 | 6.2 | 1.35e+03 |
| 23 | outlier | sparse | 100 | outliergram | 0.237 | 6.86 | 1.33e+03 |
| 23 | outlier | sparse | 100 | msplot | 0.267 | 81.5 | 1.34e+03 |
| 24 | outlier | sparse | 500 | boxplot | 1.17 | 15.8 | 1.29e+03 |
| 24 | outlier | sparse | 500 | outliergram | 1.18 | 21.6 | 1.36e+03 |
| 24 | outlier | sparse | 500 | msplot | 1.29 | 321 | 1.4e+03 |
| 25 | outlier | dense | 100 | boxplot | 0.000608 | 0.139 | 1.38e+03 |
| 25 | outlier | dense | 100 | outliergram | 0.00197 | 1.1 | 1.37e+03 |
| 25 | outlier | dense | 100 | msplot | 0.0341 | 75.7 | 1.39e+03 |
| 26 | outlier | dense | 500 | boxplot | 0.00291 | 0.726 | 1.34e+03 |
| 26 | outlier | dense | 500 | outliergram | 0.0136 | 10 | 1.36e+03 |
| 26 | outlier | dense | 500 | msplot | 0.0916 | 307 | 1.37e+03 |
| 27 | outlier | sparse | 100 | boxplot | 0.238 | 6.26 | 1.38e+03 |
| 27 | outlier | sparse | 100 | outliergram | 0.241 | 6.92 | 1.33e+03 |
| 27 | outlier | sparse | 100 | msplot | 0.276 | 81.6 | 1.38e+03 |
| 28 | outlier | sparse | 500 | boxplot | 1.17 | 16 | 1.33e+03 |
| 28 | outlier | sparse | 500 | outliergram | 1.18 | 20.7 | 1.36e+03 |
| 28 | outlier | sparse | 500 | msplot | 1.29 | 322 | 1.39e+03 |
| 29 | outlier | dense | 100 | boxplot | 0.000589 | 0.139 | 1.35e+03 |
| 29 | outlier | dense | 100 | outliergram | 0.00192 | 1.07 | 1.4e+03 |
| 29 | outlier | dense | 100 | msplot | 0.0351 | 75.7 | 1.37e+03 |
| 30 | outlier | dense | 500 | boxplot | 0.00309 | 0.726 | 1.36e+03 |
| 30 | outlier | dense | 500 | outliergram | 0.0141 | 10.1 | 1.34e+03 |
| 30 | outlier | dense | 500 | msplot | 0.0959 | 307 | 1.36e+03 |
| 31 | outlier | sparse | 100 | boxplot | 0.237 | 6.25 | 1.35e+03 |
| 31 | outlier | sparse | 100 | outliergram | 0.241 | 6.9 | 1.37e+03 |
| 31 | outlier | sparse | 100 | msplot | 0.268 | 81.6 | 1.32e+03 |
| 32 | outlier | sparse | 500 | boxplot | 1.17 | 15.9 | 1.36e+03 |
| 32 | outlier | sparse | 500 | outliergram | 1.18 | 20.8 | 1.36e+03 |
| 32 | outlier | sparse | 500 | msplot | 1.29 | 322 | 1.4e+03 |
| 33 | outlier | dense | 100 | boxplot | 0.00059 | 0.131 | 1.37e+03 |
| 33 | outlier | dense | 100 | outliergram | 0.00189 | 1.01 | 1.39e+03 |
| 33 | outlier | dense | 100 | msplot | 0.0331 | 75.7 | 1.4e+03 |
| 34 | outlier | dense | 500 | boxplot | 0.00297 | 0.709 | 1.36e+03 |
| 34 | outlier | dense | 500 | outliergram | 0.0107 | 6.51 | 1.35e+03 |
| 34 | outlier | dense | 500 | msplot | 0.0948 | 307 | 1.36e+03 |
| 35 | outlier | sparse | 100 | boxplot | 0.237 | 6.23 | 1.36e+03 |
| 35 | outlier | sparse | 100 | outliergram | 0.238 | 6.95 | 1.31e+03 |
| 35 | outlier | sparse | 100 | msplot | 0.27 | 81.5 | 1.33e+03 |
| 36 | outlier | sparse | 500 | boxplot | 1.17 | 15.8 | 1.37e+03 |
| 36 | outlier | sparse | 500 | outliergram | 1.18 | 20.9 | 1.38e+03 |
| 36 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.33e+03 |
| 37 | outlier | dense | 100 | boxplot | 0.000582 | 0.13 | 1.39e+03 |
| 37 | outlier | dense | 100 | outliergram | 0.00176 | 0.896 | 1.36e+03 |
| 37 | outlier | dense | 100 | msplot | 0.0329 | 75.7 | 1.39e+03 |
| 38 | outlier | dense | 500 | boxplot | 0.00309 | 0.709 | 1.41e+03 |
| 38 | outlier | dense | 500 | outliergram | 0.0102 | 5.55 | 1.38e+03 |
| 38 | outlier | dense | 500 | msplot | 0.0988 | 307 | 1.36e+03 |
| 39 | outlier | sparse | 100 | boxplot | 0.237 | 6.22 | 1.35e+03 |
| 39 | outlier | sparse | 100 | outliergram | 0.237 | 6.92 | 1.39e+03 |
| 39 | outlier | sparse | 100 | msplot | 0.269 | 81.5 | 1.37e+03 |
| 40 | outlier | sparse | 500 | boxplot | 1.17 | 15.8 | 1.36e+03 |
| 40 | outlier | sparse | 500 | outliergram | 1.18 | 21.1 | 1.35e+03 |
| 40 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.36e+03 |
| 41 | outlier | dense | 100 | boxplot | 0.000579 | 0.131 | 1.38e+03 |
| 41 | outlier | dense | 100 | outliergram | 0.00187 | 0.994 | 1.38e+03 |
| 41 | outlier | dense | 100 | msplot | 0.0342 | 75.6 | 1.4e+03 |
| 42 | outlier | dense | 500 | boxplot | 0.00309 | 0.709 | 1.39e+03 |
| 42 | outlier | dense | 500 | outliergram | 0.0126 | 7.73 | 1.38e+03 |
| 42 | outlier | dense | 500 | msplot | 0.0934 | 307 | 1.36e+03 |
| 43 | outlier | sparse | 100 | boxplot | 0.236 | 6.22 | 1.3e+03 |
| 43 | outlier | sparse | 100 | outliergram | 0.237 | 6.96 | 1.36e+03 |
| 43 | outlier | sparse | 100 | msplot | 0.27 | 81.6 | 1.38e+03 |
| 44 | outlier | sparse | 500 | boxplot | 1.18 | 15.9 | 1.36e+03 |
| 44 | outlier | sparse | 500 | outliergram | 1.18 | 22 | 1.35e+03 |
| 44 | outlier | sparse | 500 | msplot | 1.27 | 322 | 1.36e+03 |
| 45 | outlier | dense | 100 | boxplot | 0.000609 | 0.131 | 1.39e+03 |
| 45 | outlier | dense | 100 | outliergram | 0.00172 | 0.891 | 1.36e+03 |
| 45 | outlier | dense | 100 | msplot | 0.0337 | 75.7 | 1.4e+03 |
| 46 | outlier | dense | 500 | boxplot | 0.00319 | 0.709 | 1.39e+03 |
| 46 | outlier | dense | 500 | outliergram | 0.0108 | 6.49 | 1.33e+03 |
| 46 | outlier | dense | 500 | msplot | 0.096 | 307 | 1.32e+03 |
| 47 | outlier | sparse | 100 | boxplot | 0.236 | 6.23 | 1.33e+03 |
| 47 | outlier | sparse | 100 | outliergram | 0.238 | 6.94 | 1.34e+03 |
| 47 | outlier | sparse | 100 | msplot | 0.268 | 81.6 | 1.36e+03 |
| 48 | outlier | sparse | 500 | boxplot | 1.17 | 15.9 | 1.33e+03 |
| 48 | outlier | sparse | 500 | outliergram | 1.19 | 21.6 | 1.34e+03 |
| 48 | outlier | sparse | 500 | msplot | 1.3 | 322 | 1.35e+03 |
| 49 | outlier | dense | 100 | boxplot | 0.000588 | 0.131 | 1.4e+03 |
| 49 | outlier | dense | 100 | outliergram | 0.00193 | 1.02 | 1.39e+03 |
| 49 | outlier | dense | 100 | msplot | 0.0336 | 75.6 | 1.37e+03 |
| 50 | outlier | dense | 500 | boxplot | 0.00318 | 0.709 | 1.34e+03 |
| 50 | outlier | dense | 500 | outliergram | 0.0127 | 8.18 | 1.34e+03 |
| 50 | outlier | dense | 500 | msplot | 0.0963 | 307 | 1.35e+03 |
| 51 | outlier | sparse | 100 | boxplot | 0.238 | 6.24 | 1.35e+03 |
| 51 | outlier | sparse | 100 | outliergram | 0.238 | 6.99 | 1.37e+03 |
| 51 | outlier | sparse | 100 | msplot | 0.27 | 81.6 | 1.36e+03 |
| 52 | outlier | sparse | 500 | boxplot | 1.17 | 16 | 1.39e+03 |
| 52 | outlier | sparse | 500 | outliergram | 1.18 | 22.5 | 1.36e+03 |
| 52 | outlier | sparse | 500 | msplot | 1.3 | 322 | 1.38e+03 |
| 53 | outlier | dense | 100 | boxplot | 0.000616 | 0.131 | 1.4e+03 |
| 53 | outlier | dense | 100 | outliergram | 0.00176 | 0.919 | 1.36e+03 |
| 53 | outlier | dense | 100 | msplot | 0.0328 | 75.7 | 1.38e+03 |
| 54 | outlier | dense | 500 | boxplot | 0.00316 | 0.71 | 1.34e+03 |
| 54 | outlier | dense | 500 | outliergram | 0.0104 | 6.13 | 1.34e+03 |
| 54 | outlier | dense | 500 | msplot | 0.0947 | 307 | 1.36e+03 |
| 55 | outlier | sparse | 100 | boxplot | 0.237 | 6.26 | 1.36e+03 |
| 55 | outlier | sparse | 100 | outliergram | 0.239 | 7 | 1.38e+03 |
| 55 | outlier | sparse | 100 | msplot | 0.269 | 81.6 | 1.35e+03 |
| 56 | outlier | sparse | 500 | boxplot | 1.17 | 16 | 1.38e+03 |
| 56 | outlier | sparse | 500 | outliergram | 1.18 | 22.2 | 1.39e+03 |
| 56 | outlier | sparse | 500 | msplot | 1.27 | 322 | 1.36e+03 |
| 57 | outlier | dense | 100 | boxplot | 0.000585 | 0.126 | 1.4e+03 |
| 57 | outlier | dense | 100 | outliergram | 0.00198 | 0.983 | 1.4e+03 |
| 57 | outlier | dense | 100 | msplot | 0.0332 | 75.6 | 1.37e+03 |
| 58 | outlier | dense | 500 | boxplot | 0.0032 | 0.681 | 1.42e+03 |
| 58 | outlier | dense | 500 | outliergram | 0.016 | 10 | 1.35e+03 |
| 58 | outlier | dense | 500 | msplot | 0.0965 | 307 | 1.36e+03 |
| 59 | outlier | sparse | 100 | boxplot | 0.236 | 6.18 | 1.38e+03 |
| 59 | outlier | sparse | 100 | outliergram | 0.238 | 6.61 | 1.36e+03 |
| 59 | outlier | sparse | 100 | msplot | 0.27 | 81.5 | 1.36e+03 |
| 60 | outlier | sparse | 500 | boxplot | 1.17 | 15.8 | 1.37e+03 |
| 60 | outlier | sparse | 500 | outliergram | 1.18 | 21 | 1.37e+03 |
| 60 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.34e+03 |
| 61 | outlier | dense | 100 | boxplot | 0.000596 | 0.124 | 1.37e+03 |
| 61 | outlier | dense | 100 | outliergram | 0.00198 | 0.962 | 1.36e+03 |
| 61 | outlier | dense | 100 | msplot | 0.0329 | 75.7 | 1.36e+03 |
| 62 | outlier | dense | 500 | boxplot | 0.00318 | 0.681 | 1.39e+03 |
| 62 | outlier | dense | 500 | outliergram | 0.0161 | 10.1 | 1.35e+03 |
| 62 | outlier | dense | 500 | msplot | 0.0916 | 307 | 1.33e+03 |
| 63 | outlier | sparse | 100 | boxplot | 0.237 | 6.19 | 1.39e+03 |
| 63 | outlier | sparse | 100 | outliergram | 0.238 | 6.63 | 1.35e+03 |
| 63 | outlier | sparse | 100 | msplot | 0.268 | 81.4 | 1.32e+03 |
| 64 | outlier | sparse | 500 | boxplot | 1.17 | 15.7 | 1.32e+03 |
| 64 | outlier | sparse | 500 | outliergram | 1.18 | 21 | 1.34e+03 |
| 64 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.38e+03 |
| 65 | outlier | dense | 100 | boxplot | 0.000581 | 0.137 | 1.36e+03 |
| 65 | outlier | dense | 100 | outliergram | 0.00144 | 0.685 | 1.37e+03 |
| 65 | outlier | dense | 100 | msplot | 0.0331 | 75.5 | 1.38e+03 |
| 66 | outlier | dense | 500 | boxplot | 0.00301 | 0.699 | 1.35e+03 |
| 66 | outlier | dense | 500 | outliergram | 0.00909 | 4.72 | 1.36e+03 |
| 66 | outlier | dense | 500 | msplot | 0.0965 | 306 | 1.36e+03 |
| 67 | outlier | sparse | 100 | boxplot | 0.238 | 6.23 | 1.39e+03 |
| 67 | outlier | sparse | 100 | outliergram | 0.239 | 6.73 | 1.38e+03 |
| 67 | outlier | sparse | 100 | msplot | 0.27 | 81.6 | 1.36e+03 |
| 68 | outlier | sparse | 500 | boxplot | 1.17 | 15.6 | 1.32e+03 |
| 68 | outlier | sparse | 500 | outliergram | 1.17 | 19.2 | 1.33e+03 |
| 68 | outlier | sparse | 500 | msplot | 1.3 | 321 | 1.38e+03 |
| 69 | outlier | dense | 100 | boxplot | 0.000591 | 0.137 | 1.39e+03 |
| 69 | outlier | dense | 100 | outliergram | 0.00139 | 0.647 | 1.36e+03 |
| 69 | outlier | dense | 100 | msplot | 0.0345 | 75.6 | 1.39e+03 |
| 70 | outlier | dense | 500 | boxplot | 0.00305 | 0.698 | 1.39e+03 |
| 70 | outlier | dense | 500 | outliergram | 0.00887 | 4.66 | 1.35e+03 |
| 70 | outlier | dense | 500 | msplot | 0.0947 | 306 | 1.35e+03 |
| 71 | outlier | sparse | 100 | boxplot | 0.239 | 6.23 | 1.35e+03 |
| 71 | outlier | sparse | 100 | outliergram | 0.24 | 6.71 | 1.36e+03 |
| 71 | outlier | sparse | 100 | msplot | 0.27 | 81.7 | 1.32e+03 |
| 72 | outlier | sparse | 500 | boxplot | 1.17 | 15.7 | 1.28e+03 |
| 72 | outlier | sparse | 500 | outliergram | 1.18 | 19 | 1.31e+03 |
| 72 | outlier | sparse | 500 | msplot | 1.3 | 322 | 1.37e+03 |
| 73 | clustering | dense | 100 | kmeans_curves | 0.00024 | 0.252 | 1.43e+03 |
| 73 | clustering | dense | 100 | kmeans_scores | 0.000863 | 0.429 | 1.45e+03 |
| 73 | clustering | dense | 100 | kcfc | 0.0077 | 12.9 | 1.42e+03 |
| 73 | clustering | dense | 100 | mixture | 0.011 | 22.5 | 1.43e+03 |
| 74 | clustering | dense | 500 | kmeans_curves | 0.000565 | 1.15 | 1.42e+03 |
| 74 | clustering | dense | 500 | kmeans_scores | 0.000892 | 1.23 | 1.42e+03 |
| 74 | clustering | dense | 500 | kcfc | 0.0218 | 83.5 | 1.42e+03 |
| 74 | clustering | dense | 500 | mixture | 0.0439 | 98.8 | 1.43e+03 |
| 75 | clustering | sparse | 100 | kmeans_scores | 0.237 | 5.98 | 1.46e+03 |
| 75 | clustering | sparse | 100 | kcfc | 2.55 | 112 | 1.43e+03 |
| 75 | clustering | sparse | 100 | mixture | 0.252 | 28.1 | 1.42e+03 |
| 76 | clustering | sparse | 500 | kmeans_scores | 1.17 | 14.4 | 1.42e+03 |
| 76 | clustering | sparse | 500 | kcfc | 17.7 | 384 | 1.44e+03 |
| 76 | clustering | sparse | 500 | mixture | 1.25 | 112 | 1.5e+03 |
| 77 | clustering | dense | 100 | kmeans_curves | 0.000264 | 0.259 | 1.42e+03 |
| 77 | clustering | dense | 100 | kmeans_scores | 0.000898 | 0.434 | 1.42e+03 |
| 77 | clustering | dense | 100 | kcfc | 0.0103 | 18.9 | 1.41e+03 |
| 77 | clustering | dense | 100 | mixture | 0.0162 | 28.3 | 1.42e+03 |
| 78 | clustering | dense | 500 | kmeans_curves | 0.000827 | 1.17 | 1.42e+03 |
| 78 | clustering | dense | 500 | kmeans_scores | 0.000932 | 1.25 | 1.45e+03 |
| 78 | clustering | dense | 500 | kcfc | 0.0483 | 145 | 1.44e+03 |
| 78 | clustering | dense | 500 | mixture | 0.07 | 142 | 1.43e+03 |
| 79 | clustering | sparse | 100 | kmeans_scores | 0.236 | 5.95 | 1.43e+03 |
| 79 | clustering | sparse | 100 | kcfc | 6.12 | 375 | 1.47e+03 |
| 79 | clustering | sparse | 100 | mixture | 0.251 | 32.7 | 1.42e+03 |
| 80 | clustering | sparse | 500 | kmeans_scores | 1.17 | 14.2 | 1.43e+03 |
| 80 | clustering | sparse | 500 | kcfc | 45.6 | 1.21e+03 | 1.51e+03 |
| 80 | clustering | sparse | 500 | mixture | 1.24 | 141 | 1.42e+03 |
| 81 | clustering | dense | 100 | kmeans_curves | 0.000223 | 0.254 | 1.42e+03 |
| 81 | clustering | dense | 100 | kmeans_scores | 0.000839 | 0.43 | 1.46e+03 |
| 81 | clustering | dense | 100 | kcfc | 0.00425 | 6.32 | 1.45e+03 |
| 81 | clustering | dense | 100 | mixture | 0.0109 | 21.4 | 1.41e+03 |
| 82 | clustering | dense | 500 | kmeans_curves | 0.000628 | 1.16 | 1.46e+03 |
| 82 | clustering | dense | 500 | kmeans_scores | 0.000901 | 1.24 | 1.42e+03 |
| 82 | clustering | dense | 500 | kcfc | 0.0163 | 55.9 | 1.42e+03 |
| 82 | clustering | dense | 500 | mixture | 0.0419 | 93.6 | 1.43e+03 |
| 83 | clustering | sparse | 100 | kmeans_scores | 0.237 | 5.98 | 1.43e+03 |
| 83 | clustering | sparse | 100 | kcfc | 1.9 | 83.7 | 1.47e+03 |
| 83 | clustering | sparse | 100 | mixture | 0.247 | 27 | 1.43e+03 |
| 84 | clustering | sparse | 500 | kmeans_scores | 1.17 | 14.4 | 1.41e+03 |
| 84 | clustering | sparse | 500 | kcfc | 19.8 | 435 | 1.51e+03 |
| 84 | clustering | sparse | 500 | mixture | 1.21 | 107 | 1.47e+03 |
| 85 | clustering | dense | 100 | kmeans_curves | 0.000229 | 0.26 | 1.42e+03 |
| 85 | clustering | dense | 100 | kmeans_scores | 0.000947 | 0.44 | 1.45e+03 |
| 85 | clustering | dense | 100 | kcfc | 0.00806 | 14.2 | 1.46e+03 |
| 85 | clustering | dense | 100 | mixture | 0.0169 | 30.1 | 1.42e+03 |
| 86 | clustering | dense | 500 | kmeans_curves | 0.000892 | 1.18 | 1.43e+03 |
| 86 | clustering | dense | 500 | kmeans_scores | 0.00101 | 1.28 | 1.42e+03 |
| 86 | clustering | dense | 500 | kcfc | 0.0355 | 117 | 1.43e+03 |
| 86 | clustering | dense | 500 | mixture | 0.0584 | 128 | 1.44e+03 |
| 87 | clustering | sparse | 100 | kmeans_scores | 0.237 | 6.02 | 1.42e+03 |
| 87 | clustering | sparse | 100 | kcfc | 2.65 | 159 | 1.48e+03 |
| 87 | clustering | sparse | 100 | mixture | 0.255 | 36.5 | 1.43e+03 |
| 88 | clustering | sparse | 500 | kmeans_scores | 1.17 | 14.6 | 1.43e+03 |
| 88 | clustering | sparse | 500 | kcfc | 37.3 | 999 | 1.51e+03 |
| 88 | clustering | sparse | 500 | mixture | 1.23 | 142 | 1.43e+03 |
