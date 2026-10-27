# Depth and outliers

A functional depth orders curves from the centre of a sample outwards. Deep
curves are typical; curves of low depth are candidates for outliers.
`depth(fd, method)` computes the depth of every curve with respect to its own
sample, and `depth(new, reference, method)` the depth of new curves with
respect to a reference sample. Both return a [`FunctionalDepth`](@ref) holding
the depths, the ordering from deepest to least deep, the deepest curve (the
functional median) and every setting used.

```@example depth
using FunctionalDataAnalysis
grid = collect(range(0, 1; length=31))
values = [a + b * sin(2pi * t) for (a, b) in zip(sin.(1:20), cos.(1:20) ./ 2), t in grid]
values[20, :] .+= 3                       # one shifted curve
fd = GridFunctionalData(values, grid)
d = depth(fd, ModifiedBandDepth())
(d.median, d.ordering[end], round.(d.values[[d.median, 20]]; digits=3))
```

```@example depth
new = GridFunctionalData(values[1:3, :] .+ 0.1, grid)
depth(new, fd, FraimanMuniz()).values
```

The modified epigraph and hypograph indices measure how much of the time a
curve lies below or above the others; they underlie the outliergram.

```@example depth
(modified_epigraph_index(fd)[20], modified_hypograph_index(fd)[20])
```

## Methods

| Method | Depth | Default and source |
|:--|:--|:--|
| [`FraimanMuniz`](@ref) | grid average of `1 - 2 abs(1/2 - F(x(t)))` | `scale=true` from fda.usc `depth.FM` (`FM1` with `scale=TRUE`); multivariate: Mahalanobis depth per grid point, fda.usc `depth.FMp` |
| [`BandDepth`](@ref) | share of curve pairs whose band contains the graph (J = 2) | `definition=:graph`, López-Pintado and Romo (2009); `:roahd` reproduces roahd `BD` |
| [`ModifiedBandDepth`](@ref) | grid average of the share of pairs whose band contains `x(t)` | `ties=:exact` (fda.usc `depth.MB`, roahd `MBD(manage_ties=TRUE)`); `:average` reproduces roahd's default |
| [`RandomProjectionDepth`](@ref) | average Tukey depth of `⟨x, z_j⟩` | `nproj=50` exponential-covariance Gaussian projections, fda.usc `depth.RP`; `scale` false (univariate) or true (multivariate, `depth.RPp`) |
| [`ModalDepth`](@ref) | `Σ_j K(‖x - x_j‖/h)` | Gaussian kernel; `h` the 0.15 (univariate, `depth.mode`) or 0.10 (multivariate, `depth.modep`) quantile of reference distances |

`integration=:quadrature` (the default for `FraimanMuniz` and
`ModifiedBandDepth`, and for the indices) averages over the grid with the
stored trapezoidal weights, which is the integral in the definitions and
what fda.usc `depth.MB` does. `integration=:uniform` gives each grid point
equal weight, as roahd and fda.usc `depth.FM` do. On uniform grids the two
differ only through the half weights at the end points; on nonuniform grids
equal weights over-count densely sampled stretches.

Modified band depth uses the rank formula: with `a` reference values strictly
above `x(t)` and `b` strictly below, `C(n, 2) - C(a, 2) - C(b, 2)` pairs
contain `x(t)`. Sorting each grid column once costs `O((n + n_ref) log n_ref · m)`
instead of the `O(n · n_ref² · m)` pairwise count; the tests check both
agree, with ties. Band depth checks every pair, using bit masks of the grid
points where a reference curve lies above or below `x`.

For a sample, the curve itself belongs to the reference: its pairs with every
other curve contain it, and it counts in the empirical distributions. Except
for roahd's compatibility options (`BandDepth(definition=:roahd)` and
`ModifiedBandDepth(ties=:average)`), a sample's depth therefore equals its
depth with respect to an identical reference (for random projections, with
the same projections).

## Curve representations

Depths compare curves on a common grid.

- Complete `GridFunctionalData` uses its own grid. Target and reference must
  share it; use `to_grid` to change grids explicitly.
- `BasisFunctionalData` is evaluated exactly on the `grid` keyword, which is
  required unless the other side is grid data.
- `IrregularFunctionalData`, and grid data with missing values, are represented
  by FPCA reconstructions. FPCA is fitted to the reference with
  `fit(FPCA, reference; method=:pace, fpca...)` (PACE by default; `fpca` passes
  other FPCA keywords) and new curves are reconstructed from
  `project_scores`. When either side needs FPCA, both sides use the
  reconstruction. An existing `FPCA` fit can be passed directly.

`representation` (`:grid`, `:basis_evaluated` or `:fpca_reconstruction`) and
the fitted FPCA are stored on the result, with one entry per component for
multivariate data. The modified epigraph and hypograph indices accept complete
grid data, basis data on a grid and FPCA fits; they do not fit FPCA
themselves.

## Multivariate data

For `MultivariateFunctionalData`, `BandDepth` and `ModifiedBandDepth`
average the component depths with `weights` (uniform by default; nonnegative
and summing to one), as roahd's `multiBD` and `multiMBD` do; components may
have different grids. `FraimanMuniz` uses the Mahalanobis depth
`1/(1 + d²)` of the component vector at each grid point (fda.usc `depth.FMp`)
and `RandomProjectionDepth` averages the Tukey depths of every component's
projection on shared directions (fda.usc `depth.RPp`); both need components
on one grid. `ModalDepth` uses the distance `sqrt(Σ_c ‖x_c - y_c‖²)`
(fda.usc `depth.modep`).

## Outlier displays

Three displays flag outlying curves and return plain numbers for plotting.
Each result has a data function in the plot-data layer and a RecipesBase
recipe: `plot(result)` or [`outlier_plot`](@ref) with Plots.jl loaded.

| Display | Flags | Data | Defaults and source |
|:--|:--|:--|:--|
| [`functional_boxplot`](@ref) | magnitude outliers outside the fences | [`boxplot_data`](@ref) | MBD, factor 1.5, central 50% (fda `fbplot`, roahd `fbplot`) |
| [`outliergram`](@ref) | shape outliers far below the MEI–MBD parabola | [`outliergram_data`](@ref) | factor 1.5, `p_check = 0.05`, shift check on (roahd `outliergram`) |
| [`magnitude_shape`](@ref) | curves with large robust distance of `(MO, VO)` | [`magnitude_shape_data`](@ref) | MCD of MASS `cov.rob`, Hardin–Rocke cutoff at 0.993, 200 directions (fdaoutlier `msplot`) |

```@example depth
b = functional_boxplot(fd)
(b.outliers, b.median, length(b.central))
```

The functional boxplot (Sun and Genton, 2011) orders curves by a depth
(any method of [`depth`](@ref)), takes the band of the deepest half as the
central region and inflates it by 1.5 times its pointwise range for the
fences. `factor=AdjustedFactor()` replaces 1.5 by the adjusted factor of Sun
and Genton (2012), calibrated by simulating Gaussian curves with a robust
covariance estimate so that about 0.7% of typical curves are flagged.
Multivariate data are ordered by the multivariate depth and flagged when they
leave the fences of any component.

```@example depth
shape = copy(values)
shape[7, :] .= 1.2 .* cos.(4pi .* grid)        # a curve of unusual shape
o = outliergram(GridFunctionalData(shape, grid))
(o.outliers, round(o.threshold; digits=4))
```

The outliergram (Arribas-Gil and Romo, 2014) plots MEI against MBD. A curve
that crosses no other curve lies on the parabola
`a0 + a1·MEI + n²a0·MEI²`; the distance below it grows with the number of
crossings, and distances beyond `Q3 + 1.5 IQR` flag shape outliers. Curves
at the top or bottom of the sample are checked again after a vertical shift,
since a shape outlier that lies entirely above or below the others crosses
none of them.

```@example depth
s = magnitude_shape(GridFunctionalData(shape, grid))
(s.outliers, argmax(s.variation_outlyingness), round(s.cutoff; digits=1))
```

The magnitude–shape plot (Dai and Genton, 2018) summarizes each curve by the
mean `MO` and variation `VO` of its directional outlyingness: `MO` measures
how far the curve sits from the pointwise median, `VO` how much its
outlyingness changes along the curve. Squared robust Mahalanobis distances of
`(MO, VO)` from the minimum covariance determinant estimate are compared with
the Hardin–Rocke cutoff; above, the shifted curve 20 (large `MO`) and the
shape outlier 7 (largest `VO`) are flagged. For multivariate data the
outlyingness is the Stahel–Donoho outlyingness over random projection
directions, and `‖MO‖` is plotted. The Hardin–Rocke cutoff is conservative
in small samples, where its F approximation has few denominator degrees of
freedom.

### Worked example: Canadian temperatures

The full Canadian weather data (35 stations, daily temperatures; downloaded
on first use, see the Datasets page) show the three displays side by side.

```julia
weather = load_dataset(:canadian_weather).data
temperature = component(weather, 1)
stations = ids(temperature)
stations[functional_boxplot(temperature).outliers]
# Scheffervll, Churchill, Yellowknife, Iqaluit, Inuvik, Resolute
stations[outliergram(temperature).outliers]
# St. Johns, Pr. Rupert
stations[magnitude_shape(temperature).outliers]
# Iqaluit, Inuvik, Resolute
using Random
stations[magnitude_shape(weather; rng=Xoshiro(1)).outliers]   # temperature and precipitation
# St. Johns, Sydney, Pr. Rupert, Resolute
```

The boxplot flags the cold northern stations, whose winters leave the fences.
The outliergram flags the two maritime stations, St. John's and Prince
Rupert, whose seasonal cycle is much flatter than the rest: their curves
cross many others. The magnitude–shape plot flags the three Arctic stations
with the most extreme magnitude; Prince Rupert has the next largest distance.
With precipitation as a second component, the wet maritime stations and
Resolute stand out.

### [Choosing a method](@id outlier-method-choice)

The three displays answer different questions, as the Canadian example shows:

- The functional boxplot flags a curve that leaves the fences anywhere. It
  finds magnitude outliers, isolated or persistent, but a curve of unusual
  shape that stays inside the band of typical curves is not flagged.
- The outliergram flags curves that cross many others, that is, shape
  outliers. A curve that is merely shifted crosses few others and is not a
  shape outlier; the shift check re-examines curves at the top or bottom of
  the sample for unusual shape that their position hides.
- The magnitude–shape plot combines both: `MO` measures magnitude and `VO`
  shape, and one robust distance flags either.

For sparse curves all three work on PACE reconstructions, which keep the
leading modes of variation and smooth away short isolated features.

The [outlier and clustering study](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/master/benchmark/outlier_study)
runs the three displays with their defaults on the nine simulation models of
fdaoutlier 0.2.1 (`simulation_model1` … `simulation_model9`) at their
default parameters, so that the rates are comparable with published studies
that use these generators. By outlier type the models are:

| Type | fdaoutlier models |
|:--|:--|
| shift (constant offset) | 1 |
| magnitude (isolated or partial) | 2, 3 |
| shape | 4 (reversed peak), 5 (rougher covariance), 7 and 8 (phase) |
| mixed (offset with a shape feature) | 6 |
| amplitude | 9 |

Each model runs at 5% and 10% contamination, on dense (fdaoutlier's 50-point
grid) and sparse (5–10 points per curve) designs, with `n ∈ {100, 500}` and
200 replicates per cell. The study records true and false positive rates,
the number of curves flagged, time and failures.

Results from the MBZUAI Titan cluster (Julia 1.13.3; 72 outlier cells,
14,400 runs per display, no failures) are in
[`benchmark/outlier_study/results`](https://github.com/mlakolar/FunctionalDataAnalysis.jl/tree/master/benchmark/outlier_study/results).
Mean true positive rates, as a range over models of that type, `n` and
contamination (dense / sparse):

| Type | Functional boxplot | Outliergram | Magnitude–shape |
|:--|:--|:--|:--|
| shift | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| magnitude | 0.24–0.90 / 0.08–0.75 | 0.07–0.48 / 0.22–0.45 | 1.00 / 0.39–0.95 |
| shape | 0.00–0.54 / 0.00–0.29 | 0.75–1.00 / 0.12–0.97 | 1.00 / 0.24–1.00 |
| mixed | 0.00–0.04 / 0.00–0.04 | 0.81–0.87 / 0.56–0.61 | 1.00 / 0.83–0.90 |
| amplitude | 0.00 / 0.00–0.01 | 0.32–0.57 / 0.17–0.35 | 0.91–1.00 / 0.91–0.99 |

The magnitude–shape plot detects almost every outlier type on dense curves,
but flags 4–13% of the clean curves on dense data and 10–25% on sparse data,
above the nominal level of its cutoff. The functional boxplot almost never
flags a clean curve (false positive rate at most 0.5%) but finds only shift
and part of the magnitude outliers. The outliergram finds shape and mixed
outliers with false positive rates of 1–8%, and by design not shift outliers.
Sparse designs lower every rate, because PACE reconstructions smooth away
short features. On dense curves the displays take a few milliseconds
(boxplot) to about 0.1 s (magnitude–shape) per sample; on sparse curves the
PACE fit dominates, at about 1.2 s for every display.

## Reference audit

The implementations follow the sources of roahd 1.4.3 and fda.usc 2.2.0
(current CRAN), read in full for the functions named here.

- roahd `BD` and `BD_relative` count reference curves strictly above and
  strictly below `x` on the whole grid and multiply the counts. Pairs whose
  two curves cross while still enclosing `x` are missed, so the result can be
  smaller than the band depth of López-Pintado and Romo (2009); a curve at
  zero between `(1, -1)` and `(-1, 1)` has depth 1 by the definition and 2/3
  in roahd. The default `definition=:graph` follows the definition; the test
  fixture includes data where the two differ.
- roahd `MBD` (default `manage_ties=FALSE`) and `MBD_relative` use average
  ranks, which do not count bands exactly when values tie. `manage_ties=TRUE`
  and fda.usc `depth.MB` count exactly; this is the default here.
- fda.usc `depth.MB` is internal (not exported) in 2.2.0. It weights grid
  points by the trapezoidal rule, while roahd's `MBD` and fda.usc's
  `depth.FM` give them equal weight.
- fda.usc `TD1` evaluates `1 - F(v-)` as `1 - F(v - 1e-15)`. Once
  `|v| > 16` the offset vanishes in double precision and a sample curve's
  projection no longer counts itself on the upper side: among projections
  `(1, 2, 20)` the largest gets Tukey depth 0 rather than 1/3. This package
  counts values `>= v` exactly; the golden fixture keeps projections far below
  16 in magnitude, where the two agree.
- fda.usc `depth.RP` projects with the trapezoidal inner product
  (`inprod.fdata`), `depth.RPp` with a plain sum over grid points. This
  package uses the trapezoidal inner product for both. The fixture
  projections vanish at both ends of a uniform grid, where the two inner
  products differ by the grid step, a positive factor Tukey depth ignores.
- fda.usc `depth.mode` takes the bandwidth quantile over all reference
  distances, diagonal zeros included. The default quantile is then zero for
  samples of at most five curves (`depth.mode`, quantile 0.15) or nine curves
  (`depth.modep`, quantile 0.10), and R returns depth zero for every curve.
  This package throws and asks for a bandwidth. Both default quantiles are
  kept.
- fda.usc `mdepth.MhD` (used by `depth.FMp`) replaces a singular covariance
  by an SVD pseudo-inverse with a warning; this package throws.
- roahd's `multiMBD` requires `sum(weights) == 1` exactly; this package accepts
  sums within `sqrt(eps())` of one and does not renormalize.

The outlier displays follow roahd 1.4.3, fda 6.3.0, fdaoutlier 0.2.1 (current
CRAN), MASS 7.3-65 and robustbase 0.99-7, read in full for the functions
named in `test/reference/outliers_m8.R`, including fdaoutlier's C++ and
MASS's `lqs.c`.

- Central region. roahd `fbplot` takes every curve whose depth is at least
  the median depth, which with ties can be more than half the curves; fda
  `fbplot` takes exactly `⌈n/2⌉` curves in depth order, as Sun and Genton
  define it. `central_ties=:first` (default) follows the definition,
  `central_ties=:all` roahd.
- Fences and median. roahd flags curves strictly outside a fence; fda also
  flags curves touching it, which matters only on exact equality. fda's
  median curve averages tied deepest curves; roahd and this package take the
  first. Both draw whiskers as the envelope of the curves not flagged.
- Depth. roahd's `fbplot` uses `MBD(manage_ties=TRUE)`, but inside its
  adjusted-factor search the simulated samples use `MBD` with average ranks;
  fda uses average ranks. Both weight grid points equally. This package uses
  one `method` throughout, by default the exact-tie trapezoidal MBD.
- Adjusted factor. roahd estimates the covariance with robustbase
  `covOGK(sigmamu=s_Qn)`, whose pairwise covariances use the τ-scale (the Qn
  scale is not passed on to `covGK`), and returns the raw, not the
  reweighted, OGK matrix. Each trial's factor comes from `uniroot` on
  `[0.5, 5]` with tolerance `1e-3`, although the flagged proportion is a step
  function of the factor; this package computes the factor exactly from the
  simulated curves' thresholds and needs no bracket. roahd does not adjust
  multivariate boxplots.
- robustbase `Qn` truncates candidate differences to single precision during
  its search (`(float)(y[i] - y[n - jh])` in `qn_sn.c`) and returns the
  truncated value when the search ends early, so it can differ from the
  exact order statistic by about `1e-8` relative (in the fixture,
  `1.4236938223` against `1.4236938612`). This package computes the exact
  order statistic with robustbase's constants.
- Outliergram. roahd's `outliergram` computes MBD with average ranks and
  equal weights, flags distances `>= Q3 + 1.5 IQR` (quantile type 7), and
  shifts candidates by the smallest gap to the second-lowest (or
  second-highest) curve where they are extreme. Two defects: when the first
  stage flags no curve, `MEI_data[-ID_shape_outlier]` is empty and the shift
  check is skipped; and candidates are matched with `grep(ID, ID_to_check)`,
  a regular-expression match that confuses, for example, curves 1 and 16.
  This package always runs the check (`shift=false` reproduces roahd in the
  first case) and matches indices exactly.
- Directional outlyingness. fdaoutlier's univariate `dir_out` uses R's
  `median` and `mad`, the mean over grid points and, for `VO`, the sample
  variance (divisor `m - 1`); the multivariate version divides by `m`. Its
  multivariate projection depth (C++ `projectionDepth`) takes the one-sided
  maximum of `(u'x - med)/MAD` over directions, with the upper median and an
  unscaled MAD, instead of the absolute value in Zuo's projection depth; its
  directions are uniform in the cube `[-1, 1]^d` and normalized, regenerated
  from the same seed at every grid point. `definition=:fdaoutlier` reproduces
  all of this; the default uses the two-sided outlyingness, quadrature
  weights and the population variance.
- MCD. MASS `cov.rob(method="mcd")` divides each column by its IQR,
  enumerates all `(p + 1)`-subsets when there are fewer than 5000 (otherwise
  samples `min(500(p + 1), 3000)`), runs up to four concentration steps per
  subset, stopping when the log determinant falls by less than 0.1%, and
  reweights with a `(1 + 15/(n - p))²` inflation and a 97.5% χ² cut. This
  package ports that search exactly. fdaoutlier measures distances from the
  reweighted estimate; Dai and Genton apply the Hardin–Rocke cutoff to the
  raw MCD estimate (the subset's mean and covariance), which is the default
  here.
- Hardin–Rocke cutoff. fdaoutlier's `hardin_factor_numeric` reads a table of
  simulated factors for `n < 1000` and `q ≤ 3`, and otherwise computes
  Croux–Haesbroeck factors with `c` estimated by Monte Carlo from
  `rchisq(10000, q, h/n)`, which passes `h/n` as the noncentrality rather than
  using the `h/n` quantile; the resulting `c` is about twice Hardin and
  Rocke's `P(χ²_{q+2} < χ²_{q,h/n})/(h/n)`. Its cutoff is therefore random
  and differs from the formula. This package evaluates the formula exactly
  (its F quantile matches fdaoutlier's deterministic part) and accepts a
  numeric `cutoff` for fdaoutlier's value.

The outlier study's designs follow fdaoutlier 0.2.1's
`R/simulation_models.R` (`covfunexp`, `determine` and the exported
`simulation_model1` … `simulation_model9`), read in full.

- Every model draws its process on `seq(0, 1, length.out = p)` through the
  Cholesky factor of `α exp(-β|s - t|^ν)`. Model 4 uses `α = 0.3`,
  `β = 1/0.3`; model 5's outliers use `α = 5`, `β = 2`, `ν = 0.5`; all others
  `α = β = ν = 1`.
- The default `outlier_rate` is 0.05, except 0.1 for model 6. With the
  default `deterministic = TRUE` exactly `round(n·outlier_rate)` curves are
  outlying, chosen at random.
- Model 9's sines and cosines take `seq(0, 2π, length.out = p)`, so its
  curves are `a₁ sin 2πt + a₂ cos 2πt`; the two coefficients of a curve are
  drawn independently. The file also keeps an older, commented-out model 9
  (an added `0.5 sin 40πt` oscillation), which is not exported.
- Model 2's outlying interval has length `l = 0.05`, about 2.5 steps of the
  default 50-point grid; model 8's `2 sin 15πt` has about 6.5 grid points
  per period.

## Validation

`test/reference/depth_m8.R` generates small fixtures (12 curves on 15 grid
points, a tied variant rounded to one decimal with a duplicated curve, a
nonuniform grid and a two-component multivariate sample) and writes
`test/data/m8_depth_input.csv` and `test/data/m8_depth_golden.csv` with R
4.5.1, roahd 1.4.3 and fda.usc 2.2.0. Recreate them from the repository root
with `Rscript test/reference/depth_m8.R`. The tests compare roahd `BD`,
`BD_relative`, `MBD` (both tie options), `MBD_relative`, `MEI`, `MHI`,
`multiMBD`, `multiBD` and fda.usc `depth.FM`, `depth.MB`, `depth.RP`,
`depth.mode` (with its bandwidth), `depth.FMp`, `depth.RPp` and `depth.modep`,
for samples and for new curves, at relative error below `1e-8`, using the
options above that reproduce each reference. The band depth by definition is
compared with a direct pairwise count in the same script.

`test/reference/outliers_m8.R` writes `test/data/m8_outliers_input.csv` and
`test/data/m8_outliers_golden.csv` with R 4.5.1, roahd 1.4.3, fda 6.3.0,
fdaoutlier 0.2.1, MASS 7.3-65 and robustbase 0.99-7; recreate them with
`Rscript test/reference/outliers_m8.R`. The fixtures are 24 curves on 20
points with magnitude and shape outliers, a variant with tied depths at the
median, and a two-component sample of 20 curves. The tests compare, at
relative error below `1e-8` and with exact flagged indices:

- roahd `fbplot` depths, central envelopes, fences and outliers for
  univariate, tied and multivariate data (`method=ModifiedBandDepth(integration=:uniform)`,
  `central_ties=:all`), and fda `fbplot` depths, outliers and median
  (`method=ModifiedBandDepth(ties=:average, integration=:uniform)`);
- roahd `outliergram` MEI, MBD, distances, quartiles, first-stage and final
  outliers (`method=ModifiedBandDepth(ties=:average, integration=:uniform)`),
  on data where the shift check adds a curve;
- fdaoutlier `msplot` and `dir_out` `MO`, `VO`, distances, MCD subset,
  center, covariance, outliers and median for univariate and bivariate data
  (`definition=:fdaoutlier`, fdaoutlier's directions and tabulated cutoff),
  and the raw MCD estimate and rescaled `VO` of the default definition;
- the F quantile of fdaoutlier's `croux_hesbroeck_asymptotic` and the exact
  Hardin–Rocke `c`;
- robustbase `covOGK` with the exact Qn (and `Qn` itself where it is exact),
  and one adjusted-factor trial on a stored simulated sample: roahd's
  `uniroot` root lies within its reported precision of the exact factor, and
  both flag the same curves.

The outlier study's generators are checked outside the test suite:
`benchmark/outlier_study/reference/fdaoutlier_models.R` (R 4.5.1,
fdaoutlier 0.2.1) writes the deterministic part of every model, with the
process switched off and each curve's random parameters recovered from
fdaoutlier's RNG stream, and the three covariances; the study's
`check.jl` reproduces them at relative error below `1e-8`.

```@docs
depth
FunctionalDepth
FraimanMuniz
BandDepth
ModifiedBandDepth
RandomProjectionDepth
ModalDepth
modified_epigraph_index
modified_hypograph_index
functional_boxplot
FunctionalBoxplot
AdjustedFactor
outliergram
Outliergram
magnitude_shape
MagnitudeShape
boxplot_data
outliergram_data
magnitude_shape_data
outlier_plot
```
