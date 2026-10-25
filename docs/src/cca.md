# Functional canonical correlation analysis

`fit(FunctionalCCA, fd1::BasisFunctionalData, fd2::BasisFunctionalData)`
finds pairs of functional weights with high penalized correlation. Subjects
must correspond in row order. IDs must match when supplied; the two bases
and domains may differ. `predict` evaluates new pairs using the training means
and their respective training bases.

```@example functional_cca
using FunctionalDataAnalysis
import StatsAPI
weather = load_dataset(:canadian_weather; sample=true).data
temperature = component(weather, 1)
precipitation = component(weather, 2)
b = FourierBasis(domain(temperature), 3)
x = to_basis(temperature, b)
y = to_basis(precipitation, b)
canonical = StatsAPI.fit(FunctionalCCA, x, y; lambda=1e4)
(squared_canonical_correlations(canonical), size(canonical_variates(canonical).left))
```

`weights1` and `weights2` are exact `BasisFunctionalData`, one weight per row.
`variates1` and `variates2` have subjects in rows and canonical pairs in
columns. Evaluate weights with `evaluate(canonical.weights1, times)`.
`correlations` contains the penalized singular values, and
`squared_correlations` their squares. These are the penalized criterion values,
which generally differ from empirical squared correlations of the variates.

Use `canonical_weights(canonical)` and `canonical_variates(canonical)` to
obtain `(left, right)` named tuples referencing those stored objects, and
`canonical_correlations` / `squared_canonical_correlations` for their stored
criterion vectors. `nobs` counts paired training subjects and `ncomponents`
counts canonical pairs. Use `deepcopy` for independent weights and `copy`
for independent arrays; these accessors do not copy the fit's storage.

The reference is fda **6.3.0**
[`cca.fd.R`](https://github.com/cran/fda/blob/6.3.0/R/cca.fd.R) and
[`geigen.R`](https://github.com/cran/fda/blob/6.3.0/R/geigen.R).
Let X=C₁G₁ and Y=C₂G₂ be centered coefficient rows multiplied by their exact
basis Gram matrices. The algorithm whitens XᵀY/n using the two metrics
XᵀX/n + λ₁P₁ and YᵀY/n + λ₂P₂, then takes an SVD. It back-transforms the
weights and normalizes each coefficient vector to Euclidean length one,
following R. This coefficient normalization is basis-dependent; it does not
set weight L² norms or variate variances to one. No ridge or rank truncation
is added; a singular penalized metric raises an informative error.

| Choice | Default and source |
|:--|:--|
| `ncan` | `2`, fda `cca.fd`; any positive integer up to the smaller basis dimension |
| `lambda` | `(1e-10,1e-10)`, fda `cca.fd`; a scalar sets both sides, a pair sets them separately |
| `penalty` | `(2,2)`, fda `cca.fd`; each side can instead use a derivative order, Fourier harmonic-acceleration symbol, or symmetric positive-semidefinite matrix. A scalar setting applies to both. |
| `center` | `true`, fda `centerfns`; false uses zero means |
| Covariance scale | Denominator n, fda `cca.fd` |
| Sign rule | Make the first side's largest absolute coefficient positive, and flip both sides together; a deterministic package convention |
| Selection | None unless `lambda=LambdaCV(candidates)` is explicit; use the same selected lambda for both sides and all canonical pairs |

R reuses the first parameter object by default on the second side. We apply
the second-derivative default to each side's own basis, including when their
sizes differ. R's `cca.fd(ncan=1)` drops a matrix dimension and fails; this
implementation supports one canonical pair. The reference generator requests
two pairs and extracts the leading pair for CV.

The standard opt-in criterion is in Ramsay and Silverman (2005), §11.3.3,
following [Leurgans, Moyeed and Silverman (1993)](https://doi.org/10.1111/j.2517-6161.1993.tb01936.x).
For each candidate, leave out each subject pair, fit the leading weights, and
collect their two held-out variates. Maximize the squared correlation across
these held-out pairs. We center each training fold and subtract those training
means from the held-out curves. Fold weights use the same coefficient norm
and paired sign convention as the final fit; no full-data pilot is fitted.
This makes the numerical scale and sign across folds explicit, details left
unspecified by the criterion's formula. CV is for the leading pair; subsequent
pairs reuse its selected lambda, as recommended in §11.3.3. Only a common
lambda is searched; fixed unequal penalties remain available as a pair.
Candidate order, scores, selected index, fold centering, and sign convention
are retained in `settings.lambda_selection` and shown by `show`. Ties select
the first maximum, and undefined held-out variate variance or singular fold
metrics throw.

See [Smoothed FPCA](fpca.md#Smoothed-FPCA) for the growth/weather fixtures,
recreation command, and sign/scale conversions. All CCA weight, variate, and
correlation comparisons use relative tolerance `1e-8`.

```@docs
FunctionalCCA
canonical_weights
canonical_variates
canonical_correlations
squared_canonical_correlations
```
