"""
    fit(CrossCovariance, fd1::GridFunctionalData, fd2::GridFunctionalData)

Sample cross-covariance of complete common-grid curves, with denominator
`nobs(fd1)-1`, as in fdapace 0.6.0's dense `GetCrCovYX` route. Each node
is centered by its own sample mean. The two nodes may have different grids;
each must share its own grid across subjects. Subject counts must agree, and
IDs must agree in order when both nodes have them. Missing values throw.
No smoothing or noise subtraction is performed. Inputs are copied. The fit
stores both grids, quadrature weights, means, subject IDs, and denominator.
Use [`evaluate`](@ref) for bilinear interpolation inside the two grids.
"""
struct CrossCovariance{T<:AbstractFloat}
    grid1::Vector{T}
    grid2::Vector{T}
    surface::Matrix{T}
    weights1::Vector{T}
    weights2::Vector{T}
    mean1::Vector{T}
    mean2::Vector{T}
    denominator::Int
    ids::Union{Nothing,Vector{Any}}
end

function _cross_subjects(fd1, fd2)
    nobs(fd1) == nobs(fd2) || throw(DimensionMismatch(
        "cross-covariance nodes must have the same number of subjects"))
    _validate_component_ids((fd1, fd2))
    nothing
end

function fit(::Type{CrossCovariance}, fd1::GridFunctionalData, fd2::GridFunctionalData)
    _cross_subjects(fd1, fd2)
    fd1.observed === nothing && fd2.observed === nothing || throw(ArgumentError(
        "CrossCovariance requires complete grid data without missing observations"))
    n = nobs(fd1)
    n >= 2 || throw(ArgumentError("CrossCovariance requires at least two subjects"))
    T = promote_type(eltype(fd1), eltype(fd2))
    mu1 = vec(sum(fd1.values; dims=1) ./ n)
    mu2 = vec(sum(fd2.values; dims=1) ./ n)
    surface = transpose(fd1.values .- transpose(mu1)) *
        (fd2.values .- transpose(mu2)) / (n-1)
    all(isfinite, surface) && all(isfinite, mu1) && all(isfinite, mu2) ||
        throw(ArgumentError("sample cross-covariance must be finite; rescale observations"))
    CrossCovariance(Vector{T}(fd1.grid), Vector{T}(fd2.grid), Matrix{T}(surface),
        Vector{T}(fd1.weights), Vector{T}(fd2.weights), Vector{T}(mu1), Vector{T}(mu2),
        n-1, ids(fd1) === nothing ? ids(fd2) : ids(fd1))
end

"""
    fit(SmoothedCrossCovariance, fd1, fd2, mean1, mean2; grid1, grid2=grid1,
        bandwidth, kernel=KernelSmoothing.GaussianKernel(), weighting=:obs,
        exclude_same_time=false, binning=nothing)

Smooth all within-subject cross-node residual products by the existing
bivariate local linear smoother on `grid1 × grid2`. Nodes are grid or
irregular data with equal subject counts and matching IDs when both have
them. Means are [`SmoothedMean`](@ref) fits or one-curve complete
[`GridFunctionalData`](@ref) (including multivariate FPCA means). Means are
interpolated at original times and must cover every observation. `SmoothedMean`
fits must have `derivative=0`. Work grids must cover their respective
observations. Inputs and means are copied.

The Gaussian kernel and `exclude_same_time=false` follow fdapace 0.6.0's
`GetCrCovYX`. Keeping equal-time pairs assumes independent measurement errors
between nodes. Set `exclude_same_time=true` for correlated errors at shared
times; equality is exact and tested before binning. Each subject must retain
at least one pair. `:obs` gives each pair weight one; `:subj` gives weight
`1/(N1_i*N2_i)`, also after exclusions (no renormalization). Stored data
weights are unused. The surface is not symmetrized or projected to be PSD.

`bandwidth` is a positive scalar (both axes), `(h1,h2)`, or [`CurveCV`](@ref)
with explicit scalar candidates. Unlike fdapace's default Gaussian GCV
selection, bandwidths are required: this reuses milestone 2's whole-subject
CV rather than a pair-level loss ignoring dependence. Both supplied means
remain fixed across folds. All retained raw products contribute to CV.
`binning=nothing` disables snapping; integer `binning>=2` uses the existing
covariance convention separately on each observed axis range. Aggregation
preserves weights and within-cell squared errors; CV queries the snapped
pair coordinates. Exact duplicate pair coordinates are also combined without
snapping, preserving total weights, weighted products and within-cell scatter.
CV splits subjects before combining training and held-out pairs separately.
This leaves fixed-bandwidth fits unchanged. No nearest-neighbor bandwidth floor or hidden bandwidth selection
is used. All grids, weights, means, settings, and CV results are stored and
shown. Use [`evaluate`](@ref) for bilinear interpolation inside the grids.
"""
struct SmoothedCrossCovariance{T<:AbstractFloat,H,K,B,S,M1,M2}
    grid1::Vector{T}
    grid2::Vector{T}
    surface::Matrix{T}
    weights1::Vector{T}
    weights2::Vector{T}
    bandwidth::H
    kernel::K
    weighting::Symbol
    exclude_same_time::Bool
    binning::B
    degree::Int
    design::Tuple{Symbol,Symbol}
    folds::Int
    selection::S
    mean1::M1
    mean2::M2
end

function _cross_pool(fd1, fd2, mean1, mean2, weighting, exclude_same_time, binning; aggregate=false)
    t1, y1, _, subjects1 = _mean_pool(fd1, weighting)
    t2, y2, _, subjects2 = _mean_pool(fd2, weighting)
    r1 = y1 - _covariance_mean_values(mean1, t1)
    r2 = y2 - _covariance_mean_values(mean2, t2)
    all(isfinite, r1) && all(isfinite, r2) || throw(ArgumentError(
        "cross-covariance residuals must be finite"))
    x1, x2 = _covariance_snap(t1, binning), _covariance_snap(t2, binning)
    # A unique axis prevents duplicates across subjects. Optional binning's
    # within-subject duplicates are still combined by the existing cell map.
    aggregate = aggregate && !(allunique(x1) || allunique(x2))
    T = promote_type(eltype(r1), eltype(r2), eltype(t1), eltype(t2))
    pool = (; x=T[], y=T[], z=T[], weights=T[], scatter=T[], subjects=Int[])
    offsets1 = _pooled_subject_offsets(subjects1, nobs(fd1))
    offsets2 = _pooled_subject_offsets(subjects2, nobs(fd2))
    cells = Dict{Tuple{T,T},Int}()
    for i in 1:nobs(fd1)
        ix1 = offsets1[i]:(offsets1[i+1]-1)
        ix2 = offsets2[i]:(offsets2[i+1]-1)
        has_pair = false
        weight = weighting == :obs ? one(T) : one(T)/(length(ix1)*length(ix2))
        aggregate || empty!(cells)
        for j in ix1, k in ix2
            exclude_same_time && t1[j] == t2[k] && continue
            has_pair = true
            _covariance_add_pair!(pool, cells, aggregate ? true : binning,
                x1[j], x2[k], r1[j]*r2[k], weight, i)
        end
        has_pair || throw(ArgumentError(
            exclude_same_time ? "subject $i has no cross-covariance pairs after excluding same-time observations" : "subject $i has no cross-covariance pairs; it lacks observations on one of the nodes"))
    end
    all(isfinite, pool.z) && all(isfinite, pool.scatter) || throw(ArgumentError(
        "cross-covariance raw products must be finite; rescale observations"))
    pool
end

function _cross_bandwidth(bandwidth, kernel)
    LocalPolynomial(; bandwidth, kernel).bandwidth
end
function _cross_bandwidth(bandwidth::Tuple{Any,Any}, kernel)
    map(h -> LocalPolynomial(; bandwidth=h, kernel).bandwidth, bandwidth)
end

function fit(::Type{SmoothedCrossCovariance},
        fd1::Union{GridFunctionalData,IrregularFunctionalData},
        fd2::Union{GridFunctionalData,IrregularFunctionalData},
        mean1::Union{SmoothedMean,GridFunctionalData},
        mean2::Union{SmoothedMean,GridFunctionalData};
        grid1::AbstractVector{<:Real}, grid2::AbstractVector{<:Real}=grid1,
        bandwidth, kernel=KernelSmoothing.GaussianKernel(), weighting=:obs,
        exclude_same_time=false, binning=nothing)
    mean1 isa SmoothedMean && _require_value_mean(mean1)
    mean2 isa SmoothedMean && _require_value_mean(mean2)
    _cross_subjects(fd1, fd2)
    weighting in (:obs, :subj) || throw(ArgumentError("weighting must be :obs or :subj"))
    exclude_same_time isa Bool || throw(ArgumentError("exclude_same_time must be a Bool"))
    binning === nothing || (_integer_in(binning, 2)) ||
        throw(ArgumentError("binning must be nothing or an integer of at least two"))
    binning = binning === nothing ? nothing : Int(binning)
    T = promote_type(eltype(fd1), eltype(fd2), _floattype(eltype(grid1)),
        _floattype(eltype(grid2)), eltype(mean1.values), eltype(mean2.values))
    g1, g2 = Vector{T}(_evaluation_points(fd1, grid1)), Vector{T}(_evaluation_points(fd2, grid2))
    w1, w2 = quadrature_weights(g1), quadrature_weights(g2)
    _pace_observations(fd1, g1)
    _pace_observations(fd2, g2)
    bandwidth isa KernelSmoothing.BandwidthSelector && throw(ArgumentError(
        "bandwidth selectors are univariate; use CurveCV with explicit candidates for cross-covariance"))
    bandwidth isa CurveCV && bandwidth.pilot !== nothing && throw(ArgumentError(
        "pilot CurveCV is univariate; cross-covariance requires explicit candidates"))
    initial = _cross_bandwidth(_initial_bandwidth(bandwidth), kernel)
    chosen_kernel = deepcopy(kernel)
    selecting = bandwidth isa CurveCV
    pool = _cross_pool(fd1, fd2, mean1, mean2, weighting, exclude_same_time, binning;
        aggregate=!selecting)
    selection = bandwidth isa CurveCV ? _covariance_selection(bandwidth, fd1, pool, chosen_kernel) : nothing
    h = selection === nothing ? initial : selection.bandwidths[argmin(selection.errors)]
    selecting && (pool = _covariance_collapse_pairs(pool))
    surface = T.(_local_linear_surface(pool, g1, g2, h, chosen_kernel))
    all(isfinite, surface) || throw(ArgumentError("cross-covariance must remain finite after conversion"))
    SmoothedCrossCovariance(g1, g2, surface, w1, w2, h, chosen_kernel, weighting,
        exclude_same_time, binning, 1, (sampling_design(fd1), sampling_design(fd2)),
        bandwidth isa CurveCV ? bandwidth.folds : 0, selection, deepcopy(mean1), deepcopy(mean2))
end

"""
    evaluate(c::Union{CrossCovariance,SmoothedCrossCovariance}, s::Real, t::Real)

Bilinearly interpolate a cross-covariance surface at finite coordinates
inside its two closed grid ranges. Extrapolation throws `ArgumentError`.
"""
function evaluate(c::Union{CrossCovariance,SmoothedCrossCovariance}, s::Real, t::Real)
    g1, g2 = c.grid1, c.grid2
    isfinite(s) && first(g1) <= s <= last(g1) &&
        isfinite(t) && first(g2) <= t <= last(g2) || throw(ArgumentError(
            "cross-covariance evaluation times must be finite and inside their stored grid ranges"))
    i = min(searchsortedlast(g1, s), length(g1)-1)
    j = min(searchsortedlast(g2, t), length(g2)-1)
    u, v = (s-g1[i])/(g1[i+1]-g1[i]), (t-g2[j])/(g2[j+1]-g2[j])
    (1-u)*(1-v)*c.surface[i,j] + u*(1-v)*c.surface[i+1,j] +
        (1-u)*v*c.surface[i,j+1] + u*v*c.surface[i+1,j+1]
end

_cross_reverse_bandwidth(h::Real) = h
_cross_reverse_bandwidth(h::Tuple) = reverse(h)
function _cross_transpose(c::SmoothedCrossCovariance)
    SmoothedCrossCovariance(copy(c.grid2), copy(c.grid1), copy(transpose(c.surface)),
        copy(c.weights2), copy(c.weights1), _cross_reverse_bandwidth(c.bandwidth),
        deepcopy(c.kernel), c.weighting, c.exclude_same_time, c.binning, c.degree,
        reverse(c.design), c.folds, deepcopy(c.selection), deepcopy(c.mean2), deepcopy(c.mean1))
end

function _cross_block_mean(fd, grid, bandwidth, kernel, weighting)
    if fd isa GridFunctionalData && fd.observed === nothing && bandwidth === nothing
        return GridFunctionalData(sum(fd.values; dims=1) ./ nobs(fd), fd.grid)
    end
    a, b = _pace_observed_range(fd)
    h = bandwidth === nothing ? _pace_default_mean_bandwidth(a, b) : bandwidth
    fit(SmoothedMean, fd; grid, bandwidth=h, kernel, weighting)
end

"""
    cross_covariances(mfd; bandwidth, grid=nothing, fpca=nothing, means=nothing,
        mean_bandwidth=nothing, diagonal_bandwidth=nothing,
        kernel=KernelSmoothing.GaussianKernel(), weighting=:obs,
        exclude_same_time=false, binning=nothing)

Return a matrix of fits: existing [`SmoothedCovariance`](@ref) on the diagonal
and [`SmoothedCrossCovariance`](@ref) off the diagonal. Fit each unordered pair
once and transpose it for the reverse block. `bandwidth` is a shared scalar
or explicit-candidate [`CurveCV`](@ref); per-axis fixed bandwidths are available
through direct cross fits. `diagonal_bandwidth` controls the diagonal blocks'
noise smoothers and defaults to their selected covariance bandwidth.

Supply `fpca::MultivariateFPCA` to use its stored means, or `means`, a vector
of one supported mean per node. Counts and available IDs must match the data.
Otherwise use the multivariate FPCA mean defaults: sample means for complete
grid nodes; local linear Gaussian means with 5% of each observed range for
irregular or masked nodes (fdapace 0.6.0 PACE default). `mean_bandwidth` overrides
this choice, requesting smoothing also for complete grid nodes. It may be
shared or a vector per node, and throws when means are supplied.

`grid=nothing` uses supplied mean grids, or the FPCA default grids (the data
grid for complete grid nodes; 51 equally spaced observed-range points
otherwise). A real vector shares one work grid; a vector of vectors supplies
one per node. Other smoothing defaults and the independence assumption are
those of `SmoothedCrossCovariance`. Diagonal blocks always exclude identical
observation indices, following `SmoothedCovariance`. Means and settings remain
on the fitted blocks. This helper always smooths; use `CrossCovariance` for an
unsmoothed sample cross-covariance.
"""
function cross_covariances(mfd::MultivariateFunctionalData; bandwidth, grid=nothing,
        fpca=nothing, means=nothing, mean_bandwidth=nothing, diagonal_bandwidth=nothing,
        kernel=KernelSmoothing.GaussianKernel(), weighting=:obs,
        exclude_same_time=false, binning=nothing)
    p = ncomponents(mfd)
    exclude_same_time isa Bool || throw(ArgumentError("exclude_same_time must be a Bool"))
    bandwidth isa Real || bandwidth isa CurveCV || throw(ArgumentError(
        "block bandwidth must be one shared number or CurveCV; use direct cross fits for axis bandwidths"))
    all(c -> c isa Union{GridFunctionalData,IrregularFunctionalData}, mfd.components) ||
        throw(ArgumentError("cross_covariances requires grid or irregular nodes"))
    fpca === nothing || fpca isa MultivariateFPCA || throw(ArgumentError(
        "fpca must be a MultivariateFPCA fit or nothing"))
    if fpca !== nothing
        means === nothing || throw(ArgumentError("supply fpca or means, not both"))
        length(fpca.means) == p || throw(DimensionMismatch("fpca must match the number of nodes"))
        all(s -> size(s, 1) == nobs(mfd), fpca.scores) || throw(DimensionMismatch(
            "fpca must match the number of subjects"))
        fpca.ids === nothing || ids(mfd) === nothing || isequal(fpca.ids, ids(mfd)) ||
            throw(ArgumentError("fpca ids must agree with data ids in subject order"))
        means = fpca.means
    end
    means === nothing || (means isa AbstractVector && length(means) == p) ||
        throw(DimensionMismatch("means must contain one mean per node"))
    means === nothing || all(m -> m isa Union{SmoothedMean,GridFunctionalData}, means) ||
        throw(ArgumentError("means must contain SmoothedMean fits or one-curve GridFunctionalData"))
    if means !== nothing
        for mean in means
            mean isa SmoothedMean && _require_value_mean(mean)
        end
    end
    means === nothing || mean_bandwidth === nothing || throw(ArgumentError(
        "mean_bandwidth is unused with supplied means or fpca"))
    split_grid = _multivariate_keyword(grid, :grid, p)
    split_mean = _multivariate_keyword(mean_bandwidth, :mean_bandwidth, p)
    grids = map(1:p) do j
        fd = component(mfd, j)
        g = split_grid ? grid[j] : grid
        resolved = g === nothing ? (means === nothing ?
            (fd isa GridFunctionalData && fd.observed === nothing ? fd.grid : _pace_default_grid(fd)) :
            means[j].grid) : g
        validated = _pace_input_grid(resolved, "cross-covariance node $j")
        _pace_observations(fd, validated)
        validated
    end
    mus = means === nothing ? map(1:p) do j
        _cross_block_mean(component(mfd, j), grids[j],
            split_mean ? mean_bandwidth[j] : mean_bandwidth, kernel, weighting)
    end : means
    blocks = Matrix{Union{SmoothedCovariance,SmoothedCrossCovariance}}(undef, p, p)
    for j in 1:p
        blocks[j,j] = fit(SmoothedCovariance, component(mfd, j), mus[j];
            grid=grids[j], bandwidth, diagonal_bandwidth, kernel, weighting, binning)
        for k in (j+1):p
            c = fit(SmoothedCrossCovariance, component(mfd, j), component(mfd, k), mus[j], mus[k];
                grid1=grids[j], grid2=grids[k], bandwidth, kernel, weighting, exclude_same_time, binning)
            blocks[j,k], blocks[k,j] = c, _cross_transpose(c)
        end
    end
    blocks
end

function Base.show(io::IO, c::CrossCovariance)
    print(io, "CrossCovariance(grid1=", c.grid1, ", grid2=", c.grid2,
        ", denominator=", c.denominator, ", mean1=", c.mean1, ", mean2=", c.mean2,
        ", ids=", c.ids, ")")
end
function Base.show(io::IO, c::SmoothedCrossCovariance)
    print(io, "SmoothedCrossCovariance(grid1=", c.grid1, ", grid2=", c.grid2,
        ", bandwidth=", c.bandwidth, ", kernel=", c.kernel, ", weighting=", c.weighting,
        ", exclude_same_time=", c.exclude_same_time, ", binning=", c.binning,
        ", degree=", c.degree, ", design=", c.design, ", folds=", c.folds,
        ", selection=", c.selection, ", mean1=", c.mean1, ", mean2=", c.mean2, ")")
end
