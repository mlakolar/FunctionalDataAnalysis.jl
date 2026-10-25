# Discovery noise-variance results (issue #17)

Reference run `20261025-fda-sigma2` on USC Discovery, **25 October 2026**.

| Setting | Reference value |
|:--|:--|
| Source repository | `mlakolar/FunctionalDataAnalysis.jl` |
| Source commit | `d26162c17fe1b221d0648023ca5b69728095abc1` (clean tracked sources) |
| Julia | `1.13.2` |
| FunctionalDataAnalysis | `0.1.0` |
| KernelSmoothing | `0.1.0`, source tree `3178afc11653d934bdb8aec4a68539fe1473ec46` |
| Work grid / kernel | 51 observed-range points / Gaussian |
| Threads | One Julia thread and one BLAS thread per worker |
| Configurations | The 48 rows in `../configurations.csv`, 500 paired-data replicates per row |
| Estimators | Surface and rotated diagonals, each middle-half and full, plus projected FGM reference |

`summary.csv` is the original machine-readable output of `summarize.py`,
without added comment lines so that standard CSV readers can use it directly.
Its SHA-256 is `bc5b9dc8fb1c01990339f7fac9bd5d433a3a5eb2bc7a6f7f74956beafd52060d`.
`summary.md` contains the original per-cell table with an added provenance
paragraph. It retains all 240 estimator rows, including successes, failures,
truncations, bias, sample variance and MSE. All 120,000 estimates succeeded.

The prescribed Fourier-process benchmark favors rotated/full in 41 of 48
configurations. Rotated/middle-half wins six and the projected reference wins
one. Rotated/full beats rotated/middle-half in 42 configurations. Dense obs/subj
configurations share data and yield identical estimates; the cells are not
independent replications. These results do not assert universal optimality.

## Recreation

Use Julia 1.13.2 and the source commit above, with the recorded KernelSmoothing
source tree. The configuration grid and `common.jl`, `run.jl`, and `summarize.py`
in this checkout match those used for the reference run. The
[benchmark README](../README.md) documents process, bandwidth, seeds,
weighting, failure handling, and Discovery submission settings. Run each of the
48 configurations on Discovery into a fresh output directory using that recipe.
With `SIGMA2_OUTPUT` set to the resulting directory and `SIGMA2_SUMMARY` to a
fresh output prefix, recreate the tables from the repository root:

```sh
python3 benchmark/sigma2/summarize.py benchmark/sigma2/configurations.csv "$SIGMA2_OUTPUT" "$SIGMA2_SUMMARY"
```

This regenerates `summary.csv` byte-for-byte; the generated Markdown omits the
provenance paragraph added here. The condensed table in
`docs/src/m5_sigma2_report.md` takes an unweighted mean of each estimator's
bias, variance and MSE over the 48 configurations. Its win counts compare all
five MSEs within each configuration. Raw replicate outputs are not included in
this results directory.
