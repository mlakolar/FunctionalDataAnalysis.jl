# Naming migration

Versions and UUIDs are unchanged. The development branch is `master`; the
previous `main` branch remains available. Develop and test with the sibling
`KernelSmoothing.jl` master checkout, as specified in the project sources.

| Previous public name | Canonical name |
| --- | --- |
| `basismatrix` | `basis_matrix` |
| `modesplot`, `screeplot` | `modes_plot`, `scree_plot` |
| `covplot`, `designplot`, `fitplot`, `bandplot` | `cov_plot`, `design_plot`, `fit_plot`, `band_plot` |
| FPCA keyword and stored key `maxK` | `max_k` |
| Stored key `requested_maxK` | `requested_max_k` |

Old function names remain deprecated bindings to the same generic functions.
FPCA accepts the old `maxK` keyword and rejects simultaneous non-default old
and new values. Stored selection/settings named tuples retain the old keys
alongside canonical keys during the transition. Numerical defaults and
external StatsAPI, LinearAlgebra and Tables interfaces are unchanged.
