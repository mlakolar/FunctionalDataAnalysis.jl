const _ObservedData = Union{GridFunctionalData,IrregularFunctionalData}

function _plot_curves(fd, curves)
    curves === :all && return collect(1:nobs(fd))
    index = if curves isa Integer && !(curves isa Bool)
        [curves]
    elseif curves isa AbstractVector{Bool}
        length(curves) == nobs(fd) || throw(DimensionMismatch("curves mask length must equal nobs"))
        findall(curves)
    elseif curves isa AbstractVector{<:Integer}
        Base.require_one_based_indexing(curves)
        collect(curves)
    else
        throw(ArgumentError("curves must be :all, an integer, an integer vector, or a Boolean mask"))
    end
    isempty(index) && throw(ArgumentError("curves selection must be nonempty"))
    all(i -> !(i isa Bool) && 1 <= i <= nobs(fd), index) ||
        throw(ArgumentError("curves indices must lie in 1:$(nobs(fd))"))
    length(unique(index)) == length(index) || throw(ArgumentError("curves indices must be distinct"))
    Vector{Int}(index)
end

"""
    curve_data(fd; grid=nothing, curves=:all)

Return copied plotting coordinates as `(; grid, curves, times, values, design,
representation)`. `times` and `values` contain one vector per selected curve,
in the requested order. `curves` accepts an index, integer vector/range, Boolean
mask, or `:all` (the task's default); empty or repeated selections throw.

Grid and irregular data return only observed points, without interpolation.
Their `grid` keyword must be `nothing`; the result records the original grid
for grid data and `nothing` for irregular data. Masked points are absent.
Basis data are evaluated on a finite increasing grid inside their domain;
the default is 201 equally spaced points, as specified for milestone 5.
Multivariate data return `(; nodes, curves)`, one result per node. A supplied
grid is shared across basis nodes; use `grid=nothing` for mixed raw/basis data.
No plotting library is needed.
"""
function curve_data(fd::_ObservedData; grid=nothing, curves=:all)
    grid === nothing || throw(ArgumentError("curve_data grid must be nothing for observed data; no interpolation is performed"))
    index = _plot_curves(fd, curves)
    points = [fd[i] for i in index]
    (; grid=fd isa GridFunctionalData ? copy(fd.grid) : nothing,
        curves=index, times=first.(points), values=last.(points),
        design=sampling_design(fd), representation=:observed)
end

function curve_data(fd::BasisFunctionalData; grid=nothing, curves=:all)
    index = _plot_curves(fd, curves)
    grid === nothing || grid isa AbstractVector{<:Real} ||
        throw(ArgumentError("curve_data grid must be a real vector or nothing"))
    points = grid === nothing ? collect(range(domain(fd)...; length=201)) : _basispoints(fd.basis, grid)
    _validate_strict_grid(points)
    values = evaluate(fd[index], points)
    (; grid=copy(points), curves=index, times=[copy(points) for _ in index],
        values=[copy(row) for row in eachrow(values)], design=sampling_design(fd),
        representation=:basis)
end

function curve_data(fd::MultivariateFunctionalData; grid=nothing, curves=:all)
    index = _plot_curves(fd, curves)
    (; nodes=[curve_data(c; grid, curves=index) for c in fd.components], curves=index)
end

"""
    mean_data(m::Union{SmoothedMean,FPCA,MFPCA,MultivariateFPCA})

Return copied `(; grid, values)` for a smoothed mean or an FPCA training mean.
Multivariate fits return `(; nodes)`, one mean per node on its stored grid.
No grid or smoothing choice is made.
"""
mean_data(m::SmoothedMean) = (; grid=copy(m.grid), values=copy(m.values))
_plot_mean(m::GridFunctionalData) = (; grid=copy(m.grid), values=vec(copy(m.values)))
_plot_mean(m::BasisFunctionalData, grid) = (; grid=copy(grid), values=vec(evaluate(m, grid)))
_plot_mean(m, grid) = _plot_mean(m)
mean_data(f::FPCA) = _plot_mean(f.mean, f.grid)
mean_data(f::MultivariateFPCA) = (; nodes=_plot_mean.(f.means))
mean_data(f::MFPCA) = mean_data(f.univariate)

function _plot_multipliers(multipliers)
    (multipliers isa Tuple || multipliers isa AbstractVector) ||
        throw(ArgumentError("multipliers must be a tuple or vector of finite real numbers"))
    isempty(multipliers) && throw(ArgumentError("multipliers must be nonempty"))
    all(c -> c isa Real && isfinite(c), multipliers) ||
        throw(ArgumentError("multipliers must be finite real numbers"))
    collect(float.(multipliers))
end

function _mode_data(mu, basis, eigenvalues, fractions, k, multipliers)
    k isa Integer && !(k isa Bool) && 1 <= k <= length(eigenvalues) ||
        throw(ArgumentError("mode k must lie in 1:$(length(eigenvalues)) selected components"))
    c = _plot_multipliers(multipliers)
    values = mu.values .+ (sqrt(eigenvalues[k]) .* basis.values[:, k]) * transpose(c)
    all(isfinite, values) || throw(ArgumentError("mode values must be finite; rescale multipliers or data"))
    (; grid=copy(mu.grid), mean=copy(mu.values), values, multipliers=c,
        component=Int(k), eigenvalue=eigenvalues[k], fve=fractions[k])
end

"""
    modes_of_variation(fit, k; multipliers=(-2, -1, 1, 2))

Return `(; grid, mean, values, multipliers, component, eigenvalue, fve)` for
`μ + c*sqrt(λ[k])*φ[k]`. Columns of `values` follow `multipliers`; `k` must
be a selected component. All arrays are copied. The default multipliers are
the boundaries of fdapace 0.6.0 `CreateModeOfVarPlot`'s shaded envelopes.
This package draws their curves and the mean instead of filled envelopes.

MFPCA returns `(; nodes)` using the joint eigenvalue and each node's joint
eigenfunction. Marginal `MultivariateFPCA` uses each node's own spectrum;
pooled fits use the shared pooled spectrum (not the node score variances).
`fve` uses the full retained spectrum up to the fit's cap, as in [`scree_data`](@ref).
"""
function modes_of_variation(f::FPCA, k; multipliers=(-2, -1, 1, 2))
    _mode_data(mean_data(f), f.basis, eigvals(f), scree_data(f).individual, k, multipliers)
end

function modes_of_variation(f::MFPCA, k; multipliers=(-2, -1, 1, 2))
    fractions = scree_data(f).individual
    (; nodes=[_mode_data(mu, b, eigvals(f), fractions, k, multipliers)
        for (mu, b) in zip(mean_data(f).nodes, f.eigenfunctions)])
end

function modes_of_variation(f::MultivariateFPCA, k; multipliers=(-2, -1, 1, 2))
    if f.basis === :marginal
        return (; nodes=[modes_of_variation(node, k; multipliers) for node in f.fits])
    end
    fractions = scree_data(f).individual
    (; nodes=[_mode_data(mu, b, f.pooled.eigenvalues, fractions, k, multipliers)
        for (mu, b) in zip(mean_data(f).nodes, f.bases)])
end

function _scree_data(values, selection)
    # Scaling first avoids overflow when eigenvalues are large but finite.
    scaled = values ./ maximum(values)
    individual = scaled ./ sum(scaled)
    cumulative = cumsum(individual)
    cumulative[end] = one(eltype(cumulative))
    (; components=collect(eachindex(values)), eigenvalues=copy(values), individual,
        cumulative, chosen=selection.chosen, threshold=selection.threshold,
        rule=selection.rule)
end

"""
    scree_data(fit)

Return `(; components, eigenvalues, individual, cumulative, chosen, threshold,
rule)` over the full retained positive spectrum, including components beyond
selected K. `threshold` is `nothing` for fixed K, AIC, or BIC; no threshold is
invented. FVE is normalized after the fit's component cap, following fdapace
0.6.0. `CreateScreePlot` draws individual bars and a cumulative line; this
package also exposes unselected components and the recorded selection.
MFPCA uses its joint spectrum. Marginal `MultivariateFPCA` returns `(; nodes)`;
pooled fits return the shared spectrum. Arrays are independent copies.
"""
scree_data(f::FPCA) = _scree_data(f.all_eigenvalues, f.selection)
scree_data(f::MFPCA) = _scree_data(f.settings.all_eigenvalues, f.settings.selection)
function scree_data(f::MultivariateFPCA)
    f.basis === :marginal && return (; nodes=scree_data.(f.fits))
    _scree_data(f.pooled.all_eigenvalues, f.pooled.selection)
end

"""
    covariance_data(x)
    covariance_data(f::FPCA; surface=:stored)

Return copied `(; grid1, grid2, values, surface)`. Matrix rows correspond to
`grid1` and columns to `grid2`, including rectangular cross-covariances.
Accepts `SmoothedCovariance`, `SmoothedCrossCovariance`, or exact dense
`CrossCovariance`. Smoothers use `surface=:smoothed`; exact blocks use `:stored`.

FPCA defaults to its stored estimated surface (`:stored`, a package display
choice). `surface=:fitted` returns the positive-semidefinite projection over
all retained components up to `max_k`, including unselected components, as in
fdapace 0.6.0's full fitted covariance used for CE scores. It does not refit
or diagonalize a surface. Other symbols throw.
"""
covariance_data(c::SmoothedCovariance) =
    (; grid1=copy(c.grid), grid2=copy(c.grid), values=copy(c.surface), surface=:smoothed)
covariance_data(c::SmoothedCrossCovariance) =
    (; grid1=copy(c.grid1), grid2=copy(c.grid2), values=copy(c.surface), surface=:smoothed)
covariance_data(c::CrossCovariance) =
    (; grid1=copy(c.grid1), grid2=copy(c.grid2), values=copy(c.surface), surface=:stored)
function covariance_data(f::FPCA; surface=:stored)
    surface isa Symbol && surface in (:stored, :fitted) ||
        throw(ArgumentError("covariance surface must be :stored or :fitted"))
    values = surface === :stored ? f.covariance : f.settings.full_projected_covariance
    (; grid1=copy(f.grid), grid2=copy(f.grid), values=Matrix(values), surface)
end

"""
    design_data(fd; diagonal=false)

Return all ordered within-curve observed pairs as `(; x, y, curves, grid,
counts, diagonal)`. `curves` identifies the subject contributing each pair;
`grid` is the sorted observed-time union. `counts=(; x, y, values, display)`
aggregates coincident pairs in lexicographic order, with exact integer counts
and display counts capped at four. Empty pair sets are retained as empty arrays.

The default excludes diagonal pairs, following fdapace 0.6.0
`CreateDesignPlot(noDiagonal=TRUE)`. Its color legend uses 1, 2, 3, 4+;
exact counts here are never capped. Set `diagonal=true` to include them.
Grid data use only observed entries; irregular curves retain their own times.
Multivariate data return `(; nodes)`. Basis data have no observation design
and throw. No resampling or dense-grid reduction is performed.
"""
function design_data(fd::_ObservedData; diagonal=false)
    diagonal isa Bool || throw(ArgumentError("design diagonal must be a Boolean"))
    T = eltype(fd)
    x, y, curves, times = T[], T[], Int[], T[]
    counts = Dict{Tuple{T,T},Int}()
    for i in 1:nobs(fd)
        t, _ = fd[i]
        append!(times, (iszero(s) ? zero(T) : s for s in t))
        for a in t, b in t
            !diagonal && a == b && continue
            push!(x, a)
            push!(y, b)
            push!(curves, i)
            # Numerically identical signed zeros must count as the same time.
            key = (iszero(a) ? zero(T) : a, iszero(b) ? zero(T) : b)
            counts[key] = get(counts, key, 0) + 1
        end
    end
    pairs = sort!(collect(keys(counts)))
    values = [counts[key] for key in pairs]
    (; x, y, curves, grid=sort!(unique(times)),
        counts=(; x=T[first(key) for key in pairs], y=T[last(key) for key in pairs],
            values, display=min.(values, 4)), diagonal)
end
design_data(fd::MultivariateFunctionalData; diagonal=false) =
    (; nodes=[design_data(c; diagonal) for c in fd.components])
design_data(::BasisFunctionalData; diagonal=false) =
    throw(ArgumentError("design_data requires observed grid or irregular data, not basis data"))

function _fitted_curve_index(f, i)
    i isa Integer && !(i isa Bool) && 1 <= i <= nobs(f) ||
        throw(ArgumentError("fitted curve i must lie in 1:$(nobs(f))"))
    Int(i)
end

"""
    fitted_curve_data(f::FPCA, i; grid=f.grid)
    fitted_curve_data(f::MFPCA, i; grid=nothing)

Return `(; curve, observed, grid, values, ncomponents, method)` with copied
`observed=(; times, values)` and the selected-K training reconstruction.
Original observations are retained even when dense fitting used presmoothing.
The default work grid and selected K follow fdapace 0.6.0 `CreatePathPlot`.
Reconstruction delegates to [`reconstruct`](@ref), including its explicit
evaluation and grid validation; scores are not recomputed. Basis fits evaluate
their exact input curves on the requested grid for `observed` coordinates.
MFPCA returns `(; nodes, curve)`; `grid=nothing` uses each node's work grid,
or pass a shared real vector or a vector of node grids as for `reconstruct`.
"""
function fitted_curve_data(f::FPCA, i; grid=f.grid)
    index = _fitted_curve_index(f, i)
    grid isa AbstractVector{<:Real} || throw(ArgumentError("fitted curve grid must be a real vector"))
    recovered = reconstruct(f, f.scores[index:index, :]; grid)
    times, values = _fpca_observations(f.training_data, index, grid)
    (; curve=index, observed=(; times, values), grid=copy(recovered.grid),
        values=vec(copy(recovered.values)), ncomponents=ncomponents(f), method=f.method)
end

_fpca_observations(fd, i, grid) = fd[i]
_fpca_observations(fd::BasisFunctionalData, i, grid) =
    (copy(grid), vec(evaluate(fd[i], grid)))

function _plot_observations(f::FPCA, i, ::Type{T}) where {T}
    times, values = f.training_data[i]
    (; times=Vector{T}(times), values=Vector{T}(values))
end

function fitted_curve_data(f::MFPCA, i; grid=nothing)
    index = _fitted_curve_index(f, i)
    recovered = reconstruct(f, f.scores[index:index, :]; grid)
    _joint_fitted_curve_data(f, recovered, index)
end

function _joint_fitted_curve_data(f::MFPCA, recovered::MultivariateFunctionalData{T}, index) where {T}
    nodes = NamedTuple{(:curve, :observed, :grid, :values, :ncomponents, :method),
        Tuple{Int,NamedTuple{(:times, :values),Tuple{Vector{T},Vector{T}}},
            Vector{T},Vector{T},Int,Symbol}}[]
    for j in eachindex(f.weights)
        observed = _plot_observations(component(f, j), index, T)
        node = component(recovered, j)::GridFunctionalData{T}
        push!(nodes, (; curve=index, observed, grid=copy(node.grid),
            values=vec(copy(node.values)), ncomponents=ncomponents(f), method=:mfpca))
    end
    (; nodes, curve=index)
end

"""
    curve_data(d::FPCADerivative; grid=nothing, curves=:all)
    curve_data(d::MultivariateFPCADerivative; grid=nothing, curves=:all)

Copied derivative curve coordinates from `reconstruct(d)`. The default grid is
stored on the fit. Multivariate results return one payload per node. No
smoothing or score estimation is repeated. All derivative curves are drawn
as paths, including curves reconstructed from sparse observations.
"""
function curve_data(d::FPCADerivative; grid=nothing, curves=:all)
    recovered = grid === nothing ? reconstruct(d) : reconstruct(d; grid)
    data = curve_data(recovered; curves)
    merge(data, (; representation=:derivative, design=:regular))
end
function curve_data(d::MultivariateFPCADerivative; grid=nothing, curves=:all)
    index = _plot_curves(d, curves)
    grids = _multivariate_grids(d, grid)
    (; nodes=[curve_data(node; grid=g, curves=index) for (node, g) in zip(d.nodes, grids)], curves=index)
end

"""
    mean_data(d::Union{FPCADerivative,MultivariateFPCADerivative})

Copied mean derivative coordinates on the stored work grid; multivariate
results return `(; nodes)`. No resmoothing is performed.
"""
mean_data(d::FPCADerivative) = _plot_mean(d.mean)
mean_data(d::MultivariateFPCADerivative) = (; nodes=mean_data.(d.nodes))

"""
    band_data(b::FittedBands)
    band_data(b::MeanBands; coverage=b.coverage)
    band_data(b::NamedTuple{(:nodes,)})

Return copied plotting coordinates and settings for precomputed uncertainty
results. No model fitting or resampling occurs. Fitted results contain
curve-by-grid `values`, `lower`, and `upper` matrices plus `curves` and `grid`.
Mean results contain one row per band type. For a mean result holding both
types, `coverage=:both` returns two series, labeled by `types`; select
`:interval` or `:band` to display one. Requesting an uncomputed type throws.
Multivariate fitted results return `(; nodes)` for separate node panels.
"""
function band_data(b::FittedBands)
    (; grid=copy(b.grid), values=copy(b.fitted), lower=copy(b.lower),
        upper=copy(b.upper), curves=copy(b.curves), types=fill(b.coverage, length(b.curves)),
        alpha=b.alpha, target=:fitted, ncomponents=b.ncomponents, settings=b.settings)
end

function band_data(b::MeanBands; coverage=b.coverage)
    coverage in (:interval, :band, :both) ||
        throw(ArgumentError("coverage must be :interval, :band, or :both"))
    types = coverage === :both ? [:interval, :band] : [coverage]
    lower, upper = similar(b.bootstrap, length(types), length(b.grid)),
        similar(b.bootstrap, length(types), length(b.grid))
    for (i, type) in enumerate(types)
        limits = type === :interval ? b.interval : b.band
        limits === nothing && throw(ArgumentError("mean result does not contain $type coverage"))
        lower[i, :], upper[i, :] = limits.lower, limits.upper
    end
    (; grid=copy(b.grid), values=repeat(permutedims(b.fitted), length(types), 1),
        lower, upper, curves=fill(1, length(types)), types, alpha=b.alpha,
        target=:mean, ncomponents=nothing, settings=b.settings)
end

band_data(b::NamedTuple{(:nodes,)}) = (; nodes=band_data.(b.nodes))

"""
    band_data(b::NamedTuple{(:grid,:estimate,:se,:lower,:upper,:alpha,:critical_value,:settings)})

Return copied plotting coordinates for one pointwise coefficient interval from
[`coefficient_bands`](@ref). The existing [`band_plot`](@ref) ribbon recipe accepts
this result directly: `band_plot(only(coefficient_bands(model)))`.
"""
function band_data(b::NamedTuple{(:grid,:estimate,:se,:lower,:upper,:alpha,:critical_value,:settings)})
    (; grid=copy(b.grid), values=permutedims(b.estimate), lower=permutedims(b.lower),
        upper=permutedims(b.upper), curves=[1], types=[:interval], alpha=b.alpha,
        target=:coefficient, ncomponents=nothing, settings=b.settings)
end

"""
    boxplot_data(b::FunctionalBoxplot)

Return copied plotting coordinates of a [`functional_boxplot`](@ref) as
`(; grid, values, central, outliers, median, central_lower, central_upper,
fence_lower, fence_upper, whisker_lower, whisker_upper, factor)`. `values` is
the curve-by-grid matrix the boxplot was computed on. Multivariate results
return `(; nodes)`, one payload per component, sharing the flagged indices.
Nothing is recomputed.
"""
function boxplot_data(b::FunctionalBoxplot)
    nodes = [(; grid=copy(r.grid), values=copy(r.values), central=copy(b.central), outliers=copy(b.outliers),
        median=b.median, central_lower=copy(r.central_lower), central_upper=copy(r.central_upper),
        fence_lower=copy(r.fence_lower), fence_upper=copy(r.fence_upper),
        whisker_lower=copy(r.whisker_lower), whisker_upper=copy(r.whisker_upper), factor=b.factor)
        for r in b.regions]
    length(nodes) == 1 ? only(nodes) : (; nodes)
end

"""
    outliergram_data(o::Outliergram)

Return copied coordinates of an [`outliergram`](@ref) as `(; mei, mbd,
outliers, parabola, boundary, threshold, coefficients)`. `parabola` and
`boundary` are `(; mei, mbd)` curves on 101 equally spaced MEI values in
`[0, 1]` (a package display choice): the parabola `a0 + a1·MEI + a2·MEI²` and
the same parabola lowered by the threshold, below which curves are flagged
in the first stage, as roahd 1.4.3 draws them.
"""
function outliergram_data(o::Outliergram)
    x = collect(range(zero(eltype(o.mei)), one(eltype(o.mei)); length=101))
    parabola = _outliergram_parabola.(Ref(o.coefficients), x)
    (; mei=copy(o.mei), mbd=copy(o.mbd), outliers=copy(o.outliers),
        parabola=(; mei=x, mbd=parabola), boundary=(; mei=copy(x), mbd=parabola .- o.threshold),
        threshold=o.threshold, coefficients=o.coefficients)
end

"""
    magnitude_shape_data(s::MagnitudeShape)

Return copied coordinates of a [`magnitude_shape`](@ref) result as
`(; magnitude, variation, outliers, median, cutoff, ellipse, multivariate)`.
`magnitude` is `MO` for univariate data and `‖MO‖` for multivariate data,
as fdaoutlier 0.2.1 plots it; `variation` is `VO`. For univariate data
`ellipse` is the `(; magnitude, variation)` boundary of the region of squared
robust distance at most `cutoff`, on 101 points (a package display choice,
following Dai and Genton's figures); it is `nothing` for multivariate data,
whose region lives in more than two dimensions.
"""
function magnitude_shape_data(s::MagnitudeShape)
    mo = s.mean_outlyingness
    multivariate = size(mo, 2) > 1
    magnitude = multivariate ? sqrt.(vec(sum(abs2, mo; dims=2))) : mo[:, 1]
    ellipse = if multivariate
        nothing
    else
        root = cholesky(Symmetric(s.covariance)).L
        angles = range(0, 2pi; length=101)
        points = s.center .+ sqrt(s.cutoff) .* (root * vcat(cos.(angles)', sin.(angles)'))
        (; magnitude=points[1, :], variation=points[2, :])
    end
    (; magnitude, variation=copy(s.variation_outlyingness), outliers=copy(s.outliers), median=s.median,
        cutoff=s.cutoff, ellipse, multivariate)
end
