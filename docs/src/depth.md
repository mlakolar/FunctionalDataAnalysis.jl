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
```
