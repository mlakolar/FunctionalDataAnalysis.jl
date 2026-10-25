# Plotting

Plot coordinates are available without loading a plotting library. The exported
data functions return named tuples with copied arrays and recorded settings.
RecipesBase supplies Plots.jl recipes; Plots is a docs-only dependency. Install
and load Plots in your own plotting environment. Makie recipes are planned for
a later milestone.

```julia
using FunctionalDataAnalysis
d = curve_data(fd; curves=1:5)  # observed times and values per curve
s = scree_data(model)          # full retained spectrum and selection
```

`curve_data` never interpolates raw observations. Masked grid entries are
absent, and irregular curves keep their own time vectors. A basis object is
evaluated on 201 equally spaced domain points by default. Pass `grid` to
`curve_data` to change this basis evaluation grid. In `plot(basis_data)`, the
corresponding keyword is `evaluation_grid`; `grid` retains its usual Plots
meaning. Raw data reject a requested evaluation grid. Curve selection accepts
an integer, an integer vector or range, a Boolean mask, or `:all`. Selection
order is retained; empty and repeated selections throw.

## Simulated examples

All figures below are generated when this page builds. The seed is fixed;
none of the examples require R. These figures illustrate fitted objects and
do not validate statistical coverage or recovery.

```@example plotting
ENV["GKSwstype"] = "100" # headless GR for documentation builds
using FunctionalDataAnalysis, Random, Plots
import StatsAPI
grid = collect(range(0.0, 1.0; length=51))
functions = fourier_eigenfunctions((0.0, 1.0), 3)
dense = simulate_kl(Xoshiro(54), 60; mean=t -> 1 + t,
    eigenfunctions=functions, eigenvalues=[2.0, 0.7, 0.2],
    sampling=RegularSampling(grid)).data
model = StatsAPI.fit(FPCA, dense; ncomponents=2, maxK=3)
plot(dense; curves=1:8, legend=false)
savefig("plotting-curves.svg"); nothing # hide
```

![](plotting-curves.svg)

Sparse curves have markers and lines through their observed points. These
lines are a display convention, with no interpolated points in `curve_data`.

```@example plotting
sparse = simulate_kl(Xoshiro(55), 60; mean=t -> 1 + t,
    eigenfunctions=functions, eigenvalues=[2.0, 0.7, 0.2], noise_sd=0.2,
    sampling=SparseSampling((0.0, 1.0), 5:9)).data
pace = StatsAPI.fit(FPCA, sparse; grid, ncomponents=2, maxK=3,
    mean_bandwidth=0.3, covariance_bandwidth=0.4, noise_variance=0.04)
plot(sparse; curves=1:5, legend=false)
savefig("plotting-sparse.svg"); nothing # hide
```

![](plotting-sparse.svg)

Basis data are evaluated explicitly for display. They have no observed-point
design plot.

```@example plotting
basis_data = to_basis(dense, MonomialBasis((0.0, 1.0), 4))
plot(basis_data; curves=1:5, evaluation_grid=grid, legend=false)
savefig("plotting-basis.svg"); nothing # hide
```

![](plotting-basis.svg)

Multivariate data use one panel per node. Nodes may have different domains
and representations. The same subject selection applies to every node.

```@example plotting
mfd = MultivariateFunctionalData([dense,
    GridFunctionalData(0.5 .* dense.values, grid .+ 2)])
plot(mfd; curves=1:4, legend=false, size=(700, 600))
savefig("plotting-nodes.svg"); nothing # hide
```

![](plotting-nodes.svg)

```@example plotting
mean_fit = StatsAPI.fit(SmoothedMean, dense; grid, bandwidth=0.25)
plot(mean_fit)
savefig("plotting-mean.svg"); nothing # hide
```

![](plotting-mean.svg)

`modesplot` shows the mean and the four default mode curves. The title gives
the eigenvalue and the component's individual FVE. Joint `MFPCA` and marginal
multivariate FPCA use one panel per node. Pooled multivariate modes use the
shared pooled eigenvalue, rather than each node's score variance.

```@example plotting
modesplot(model, 1)
savefig("plotting-modes.svg"); nothing # hide
```

![](plotting-modes.svg)

`screeplot` shows every positive eigenvalue retained up to the fit's cap,
including unselected components. FVE uses this full retained spectrum.
The vertical line shows selected K. The horizontal line appears only when
the fit used an FVE threshold. A marginal multivariate fit has per-node scree
panels; pooled and joint fits have a shared spectrum.

```@example plotting
screeplot(model)
savefig("plotting-scree.svg"); nothing # hide
```

![](plotting-scree.svg)

`covariance_data` returns matrix rows on `grid1` and columns on `grid2`.
The recipe arranges these axes for Plots, including rectangular
cross-covariances. `surface=:stored` shows the estimated covariance;
`surface=:fitted` uses its positive-semidefinite projection over **all**
retained components, including those beyond selected K. Both surfaces are
stored by FPCA; plotting does not rerun the eigendecomposition.

```@example plotting
covplot(model; surface=:fitted)
savefig("plotting-covariance.svg"); nothing # hide
```

![](plotting-covariance.svg)

For a three-dimensional view, use `covplot(model; seriestype=:surface)`.
The same recipe accepts `SmoothedCovariance`, `CrossCovariance`, and
`SmoothedCrossCovariance`.

```@example plotting
designplot(sparse)
savefig("plotting-design.svg"); nothing # hide
```

![](plotting-design.svg)

`design_data` returns every ordered within-curve pair and its subject index,
plus aggregated exact counts. The recipe plots the occupied pairs once,
using count categories 1, 2, 3, and 4+. Diagonal pairs are omitted by default;
request them with `diagonal=true`. Singleton curves contribute zero pairs
under the default and one pair with the diagonal enabled. Empty pair arrays
are valid. No dense-grid approximation or time binning is used.

```@example plotting
fitplot(pace, 1)
savefig("plotting-fit.svg"); nothing # hide
```

![](plotting-fit.svg)

Fitted paths reuse training scores and `reconstruct`. The points are copied
original observations, including when a dense fit used presmoothing.
`fitplot(model, i; grid=newgrid)` chooses a finite increasing reconstruction
grid inside the fitted domain. Joint MFPCA uses node panels, with either a
shared grid or a vector of node grids. Plotting does not run a new score solve.

## Defaults and implementation choices

The source audit is in [Comparison with fdapace](fdapace.md#Plotting-source-audit).

| Setting | Default and source |
|:--|:--|
| Curves | `:all`, milestone 5 specification |
| Raw-data grid | `nothing`, observed points only, milestone 5 specification |
| Basis display grid | 201 equally spaced domain points, milestone 5 specification |
| Mean/mode grid | Stored fit grid, milestone 5 specification |
| Mode multipliers | `(-2,-1,1,2)`, fdapace 0.6.0 shaded-envelope boundaries |
| Design diagonal | Excluded, fdapace 0.6.0 `noDiagonal=TRUE` |
| Design count display | 1, 2, 3, 4+, fdapace 0.6.0; exact counts remain in the data |
| Fitted-path grid and K | Stored grid(s) and selected K, fdapace 0.6.0 |
| Covariance view | `:stored` and `seriestype=:heatmap`, package display choices |
| Sparse markers | Circles for `sampling_design(fd)==:sparse`, package display choice |
| Node layout | One column of panels, package display choice; Plots attributes may override it |

Other choices left open by the specification: tuple fields use separate
per-curve time/value vectors; multivariate results wrap per-node tuples in
`nodes`; invalid or repeated curve indices throw; raw data reject evaluation
grids; basis display grids require at least two increasing in-domain points.
Mode curves replace fdapace's shaded polygons. Scree data expose the full
retained spectrum and use `nothing` for inapplicable thresholds. Design data
retain exact ordered pairs and uncapped counts without fdapace's optional
dense-grid reduction. `FPCA.training_data` retains copied original data;
dense FPCA also stores its full projected covariance, as PACE already did.
Copied training observations use the fit's promoted numeric type. The design
recipe uses the Plots color scale to respect backend themes and color overrides;
it retains fdapace's count categories rather than its fixed base-R palette.
These fields add storage proportional to the input and a grid-square matrix
for dense fits. Plot results are plain named tuples, whose standard `show`
displays their settings; no new fitted estimator is introduced.

## API

Precomputed fitted-curve and mean uncertainty results also support
[`band_data`](@ref) and [`bandplot`](@ref). See [Uncertainty](uncertainty.md)
for the covered targets and assumptions. Plotting bands never runs a new fit
or bootstrap. Multivariate fitted bands use node panels, and `coverage` on
mean-band plots selects among the types already computed.

```@docs
curve_data
mean_data
modes_of_variation
scree_data
covariance_data
design_data
fitted_curve_data
modesplot
screeplot
covplot
designplot
fitplot
```

Derivative fits support `curve_data(d; curves=:all, grid=nothing)` and
`plot(d; curves=1:3)` after loading Plots. They display reconstructed derivative
paths, including for PACE fits. `mean_data(d)` returns the stored mean derivative.
For nodewise derivatives, both data helpers return node payloads and the recipe
creates one panel per node. See [Derivatives](fpca.md#Derivatives).
