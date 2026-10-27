# Classification

Supervised classification assigns curves to known classes. Four classifiers
follow fda.usc 2.2.0: k nearest neighbours ([`FunctionalKNN`](@ref)), the
maximum-depth classifier ([`DepthClassifier`](@ref)), the DD-classifier
([`DDClassifier`](@ref)) and discriminant analysis on FPC scores
([`FunctionalLDA`](@ref), [`FunctionalQDA`](@ref)). All are fitted with
`fit(Type, x, y; ...)`, where `y` holds one label per curve of any sortable
type. The classes are the sorted distinct labels, stored in `classes`.
`predict(f, newx)` returns labels, `predict(f, newx; type=:probability)` the
class probabilities (columns in `classes` order) where the method defines
them, and `fitted(f)` the training labels.

```@example classification
using FunctionalDataAnalysis, Random
import StatsAPI: fit
rng = Xoshiro(2024)
grid = collect(range(0, 1; length=31))
n = 60
labels = repeat(["low", "high"], n ÷ 2)
shift = [label == "high" ? 0.8 : 0.0 for label in labels]
level = shift .+ 0.4 .* randn(rng, n)
amplitude = 0.5 .+ 0.2 .* randn(rng, n)
values = [level[i] + amplitude[i] * sin(2pi * t) + 0.05 * randn(rng) for i in 1:n, t in grid]
train = GridFunctionalData(values[1:50, :], grid)
test = GridFunctionalData(values[51:60, :], grid)
ytrain, ytest = labels[1:50], labels[51:60]
knn = fit(FunctionalKNN, train, ytrain)
(knn.k, knn.loo_error, count(predict(knn, test) .== ytest))
```

```@example classification
predict(knn, test; type=:probability)[1:3, :]
```

## Methods and defaults

| Classifier | Rule | Defaults and source |
|:--|:--|:--|
| [`FunctionalKNN`](@ref) | class shares among the curves within the k-th smallest L² distance | `k` by leave-one-out over `3, 5, ..., min(15, n-1)`, fda.usc `classif.knn` |
| [`DepthClassifier`](@ref) | the class in which the curve is deepest | `RandomProjectionDepth()`, fda.usc `classif.depth` (`depth="RP"`) |
| [`DDClassifier`](@ref) | a second classifier on the depths with respect to each class | `FraimanMuniz()` and logistic regression, fda.usc `classif.DD` (`depth="FM"`, `classif="glm"`); `FraimanMuniz()` integrates with trapezoidal weights (see [Depth and outliers](@ref)), and `FraimanMuniz(integration=:uniform)` reproduces fda.usc's equal grid weights |
| [`FunctionalLDA`](@ref), [`FunctionalQDA`](@ref) | Gaussian classes in the leading FPC scores | FPCA as [`FPCRegression`](@ref); equal priors, fda.usc `classif.lda`/`classif.qda` |

Curves reach a common grid as for [`depth`](@ref): complete grid data use
their own grid, basis data are evaluated on the `grid` keyword, and irregular
or incomplete curves are represented by their FPCA reconstruction (PACE by
default, tuned with `fpca`). New curves are brought to the training
representation, with the stored FPCA fit when there is one; complete-grid
classifiers need new curves on the training grid. Multivariate functional
data use `sqrt(Σ_c ‖x_c - y_c‖²)` for kNN and the multivariate depths of
[`depth`](@ref) for the depth classifiers.

```@example classification
md = fit(DepthClassifier, train, ytrain; method=ModifiedBandDepth())
dd = fit(DDClassifier, train, ytrain)        # Fraiman–Muniz depths, logistic rule
(count(predict(md, test) .== ytest), count(predict(dd, test) .== ytest))
```

```@example classification
predict(dd, test; type=:depth)[1:3, :]       # the DD-plot coordinates
```

The DD-classifier's second stage (`classifier`) is logistic regression
(`:glm`, the default), Gaussian discriminant analysis of the depths (`:lda`,
`:qda`), kNN on the depth vectors (`:knn`) or the degree-one rule of Li,
Cuesta-Albertos and Liu (2012) (`:dd1`, two classes). Training depths include
each curve in its own class, as in fda.usc, so with small classes the
training depths can separate the classes perfectly; logistic regression then
has no finite maximum and the fit throws, while R's `glm` warns and returns
very large coefficients. Use `:lda`, `:qda`, `:knn` or `:dd1` in that case.

Random projections are drawn once at fit time and stored on the classifier,
so every class and every prediction uses the same directions. Without a
bandwidth, `ModalDepth` uses one bandwidth per class in `DepthClassifier`
(as `classif.depth`) and the pooled-sample bandwidth for every class in
`DDClassifier` (as `classif.DD`).

```@example classification
lda = fit(FunctionalLDA, train, ytrain; ncomponents=2)
(lda.settings.fpca_method, lda.priors, count(predict(lda, test) .== ytest))
```

Discriminant analysis works on the leading scores of each covariate's FPCA
(several covariates are concatenated). As for [`FPCRegression`](@ref),
complete grid data use FACE by default and irregular or incomplete curves use
PACE, whose conditional-expectation scores make sparse curves usable. LDA
pools the class covariances with divisor `n - G` and QDA uses one covariance
per class with divisor `n_g - 1`, as `MASS::lda` and `MASS::qda`.

## Functional logistic regression

Two-class functional logistic regression is a scalar-on-function GLM and is
fitted by the regression models: [`FPCRegression`](@ref) with
`family=:binomial` (fda.usc `fregre.glm` on FPC scores) or
[`PenalizedFunctionalRegression`](@ref) with `family=:binomial` (refund
`pfr`). Code the response as 0 and 1; `predict(...; type=:response)` returns
the probability of class 1.

```@example classification
y01 = Float64.(ytrain .== "high")
logistic = fit(FPCRegression, train, y01; family=:binomial, ncomponents=2)
probability = predict(logistic, test; type=:response)
(round.(probability[1:3]; digits=3), count((probability .> 0.5) .== (ytest .== "high")))
```

## Reference audit

The implementations follow the sources of fda.usc 2.2.0 (current CRAN) and
MASS 7.3-65, read in full for the functions named here.

- `classif.knn` calls `classif.np` with `S.KNN`. With leave-one-out
  (`cv=TRUE`) the diagonal of the distance matrix is set to `Inf`, the
  neighbourhood radius is the type-4 quantile at `k/n` of each row (the k-th
  smallest distance) and the uniform kernel keeps every curve within it, so
  curves tied at the radius all count. `weights="equal"` gives every curve
  weight one, so probabilities are class shares. The default `knn` grid is
  `seq(3, min(15, n), by=2)`; the selected `k` is the first minimizer of the
  leave-one-out misclassification. Distances are `metric.lp`, the trapezoidal
  L² norm.
- When classes tie, `classif.np` and `pred2np` intend to choose the class of
  the nearest neighbour but index the distance matrix with class numbers
  (`mdist[ii, ll]`, `nmdist[ii, l]`) instead of curve indices, so the tie is
  resolved by the distance to curves 1, 2, .... This package uses the nearest
  tied neighbour. The golden fixture has two classes and odd `k`, so no tie
  arises.
- `classif.depth` computes the depth of every curve with respect to each
  class (the class's own curves included) and takes `which.max`. For the
  random-projection depths it draws 500 directions uniformly on the unit
  sphere of grid vectors when none are given, while `depth.RP` itself uses 50
  Gaussian-process directions (`rproc2fdata`); this package uses the latter,
  whose distribution does not depend on the grid resolution. `depth.mode`
  without `h` takes each class's own 0.15 quantile of distances. Its
  `switch` has no branch for the modified band depth, so `depth="MB"` fails
  ("number of items to replace ..."), and `classif.DD` rejects `"MB"`; here
  every depth of [`depth`](@ref) works with both classifiers.
- `classif.DD` with `depth="mode"` first computes the bandwidth of the whole
  sample and passes it to every class. Its `glm` rule is `classif.glm2boost`:
  one logistic regression of class 2 against class 1 with an intercept and
  one column per class depth, or one class against the rest for more classes
  (labels from the smallest probability of "not the class"). R's `glm` stops
  when the relative change of the deviance is below `1e-8`, which leaves the
  coefficients about `1e-8` (relative) from the maximum; this package iterates
  to convergence. `lda` and `qda` call MASS with its default prior (class
  proportions); `knn` uses Euclidean distances on the depths and the grid
  `seq(3, 19, by=2)`.
- The polynomial rule `DD1` (`MCR0`) searches the slopes `D₁/D₂` of the
  training points although its rule compares `k·D₁` with `D₂`, and evaluates
  its second error term on `dep[-ind[, i], 1]` with a logical `ind`, which
  drops only the first curve instead of selecting the other class. The `:dd1`
  rule here searches the slopes `D₂/D₁` that pass through training points
  and counts the training misclassification directly. `DD2`, `DD3` (random
  subsets of curves and a smoothed loss) and the `gam`, `np` and tree rules
  are not implemented.
- `classif.lda` and `classif.qda` with `weights="equal"` average the weights
  within each class and pass the normalized class averages as `prior`, so
  every class gets probability `1/G` regardless of its size. The scores come
  from `fdata2model` (with `basis.x=create.pc.basis(...)`, the PCA of the
  discretized curves) and new curves are projected with `inprod.fdata` after
  centring with the training mean.

## Validation

`test/reference/classification_m8.R` simulates two-class curves (40 training
and 10 new curves on 21 grid points) and three-class curves in the span of
`sin(kπt)`, `k = 1, 2, 3` (45 and 9 curves), and writes
`test/data/m8_classification_input.csv` and
`test/data/m8_classification_golden.csv` with R 4.5.1, fda.usc 2.2.0 and MASS
7.3-65. Recreate them from the repository root with
`Rscript test/reference/classification_m8.R`. The tests compare, at relative
error below `1e-8`: `classif.knn` leave-one-out errors, chosen `k`,
leave-one-out probabilities and new-curve probabilities; `classif.depth`
depths and labels with `depth="FM"` (`FraimanMuniz(integration=:uniform)`),
`"mode"` and `"RP"` (fixed projections); the same rule with the modified band
depth (`ModifiedBandDepth()`, whose trapezoidal weights match fda.usc's
internal `depth.MB` evaluated with respect to each class, since
`classif.depth` has no `"MB"` branch); `classif.DD` depths, logistic
coefficients (refitted to convergence in the script), fitted and new-curve
probabilities, the `lda`, `qda` and `knn` second stages and the pooled
modal-depth bandwidth; and `classif.lda`/`classif.qda` posterior
probabilities for training and new curves, using `fpca_options=(; method=:dense)`
with three components (the curves lie exactly in a three-dimensional space,
where discriminant probabilities do not depend on the eigenproblem weights).

```@docs
FunctionalKNN
StatsAPI.predict(::FunctionalKNN, ::Any)
DepthClassifier
StatsAPI.predict(::DepthClassifier, ::Any)
DDClassifier
StatsAPI.predict(::DDClassifier, ::Any)
FunctionalDiscriminant
FunctionalLDA
FunctionalQDA
StatsAPI.predict(::FunctionalDiscriminant, ::Any)
StatsAPI.fitted(::Union{FunctionalKNN,DepthClassifier,FunctionalDiscriminant,DDClassifier})
```
