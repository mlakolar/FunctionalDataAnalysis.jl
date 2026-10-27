abstract type _DepthMethod end

function _depth_integration(integration)
    integration in (:quadrature, :uniform) ||
        throw(ArgumentError("integration must be :quadrature or :uniform"))
    integration
end

"""
    FraimanMuniz(; scale=nothing, integration=:quadrature)

Fraiman–Muniz integrated depth (Fraiman and Muniz, 2001): the average over the
grid of the univariate depth `1 - |1/2 - F(x(t))|`, where `F` is the empirical
distribution of the reference curves at `t` (counting values `<= x(t)`).

`scale=true` (the univariate default, from fda.usc 2.2.0 `depth.FM`, which
passes `scale=TRUE` to `FM1`) uses `1 - 2|1/2 - F|`, with range `[0, 1]`;
`scale=false` is the original `[1/2, 1]` form. For multivariate data the
pointwise depth is the Mahalanobis depth `1/(1 + d²)` of the component vector
at each grid point, with the reference mean and covariance (denominator
`n-1`), as in fda.usc `depth.FMp` with its default `mdepth.MhD`; `scale` must
then be left at `nothing`. A singular reference covariance throws (fda.usc
falls back to an SVD pseudo-inverse with a warning).

`integration=:quadrature` (default) averages with the stored trapezoidal
weights, the integral in the definition; `:uniform` gives every grid point
equal weight, as fda.usc does. The two agree up to the end-point weights on
uniform grids. The departure follows the package rule that integrals use
quadrature weights on possibly nonuniform grids.
"""
struct FraimanMuniz <: _DepthMethod
    scale::Union{Nothing,Bool}
    integration::Symbol
    function FraimanMuniz(; scale::Union{Nothing,Bool}=nothing, integration::Symbol=:quadrature)
        new(scale, _depth_integration(integration))
    end
end

"""
    BandDepth(; definition=:graph)

Band depth with bands formed by two curves (J = 2; López-Pintado and Romo,
2009): the proportion of pairs of reference curves whose band
`min(x_j, x_k) <= x <= max(x_j, x_k)` contains the whole graph of `x`. For the
depth of a sample with respect to itself the pairs include the curve itself,
which always contain it.

`definition=:graph` (default) is that definition on the grid, counted with
bit masks in `O(n · n_ref² · m / 64)` operations. `definition=:roahd`
reproduces roahd 1.4.3 `BD` and `BD_relative`, which count only pairs with one
curve strictly above and the other strictly below on the whole grid
(`(N_above · N_below + n - 1) / C(n, 2)` for a sample). That count misses pairs
whose curves cross while still enclosing `x`, so it can be smaller; the
departure follows the published definition. For multivariate data the depth is
the weighted average of the component depths (roahd `multiBD`), with uniform
weights by default.
"""
struct BandDepth <: _DepthMethod
    definition::Symbol
    function BandDepth(; definition::Symbol=:graph)
        definition in (:graph, :roahd) || throw(ArgumentError("BandDepth definition must be :graph or :roahd"))
        new(definition)
    end
end

"""
    ModifiedBandDepth(; ties=:exact, integration=:quadrature)

Modified band depth (López-Pintado and Romo, 2009): the average over the grid
of the proportion of pairs of reference curves whose band contains `x(t)`.
For a sample the pairs include the curve itself. It is computed from ranks in
`O((n + n_ref) log n_ref · m)` operations: with `a` reference values strictly
above `x(t)` and `b` strictly below, `C(n_ref, 2) - C(a, 2) - C(b, 2)` pairs
contain `x(t)`.

`ties=:exact` (default) is that count, which is exact with ties; it equals
fda.usc 2.2.0's `depth.MB` and roahd 1.4.3's `MBD(manage_ties=TRUE)`.
`ties=:average` reproduces roahd's default `MBD(manage_ties=FALSE)` and
`MBD_relative`, which use average ranks; the two agree without ties.
`integration` is as in [`FraimanMuniz`](@ref): roahd weights grid points
equally (`:uniform`), fda.usc `depth.MB` uses the normalized trapezoidal
weights (`:quadrature`, the default). For multivariate data the depth is the
weighted average of the component depths (roahd `multiMBD`), with uniform
weights by default.
"""
struct ModifiedBandDepth <: _DepthMethod
    ties::Symbol
    integration::Symbol
    function ModifiedBandDepth(; ties::Symbol=:exact, integration::Symbol=:quadrature)
        ties in (:exact, :average) || throw(ArgumentError("ModifiedBandDepth ties must be :exact or :average"))
        new(ties, _depth_integration(integration))
    end
end

"""
    RandomProjectionDepth(; nproj=nothing, projections=nothing, scale=nothing, rng=Random.default_rng())

Random-projection depth (Cuevas, Febrero and Fraiman, 2007; fda.usc 2.2.0
`depth.RP`): the average over projections `z_j` of the univariate Tukey depth
`min(F(v), 1 - F(v-))` of `v = ⟨x, z_j⟩` among the reference projections. Inner
products use the stored trapezoidal weights, as fda.usc's `inprod.fdata`.

By default `nproj = 50` projections (fda.usc's default) are drawn from the
zero-mean Gaussian process with covariance `exp(-|s - t|/θ)`, `θ = 0.2 ×`
domain length, and normalized to unit L² norm (fda.usc `rproc2fdata` with
`sigma="vexponential"`), using `rng`. Supplying `projections` (an
`nproj × m` matrix of functions on the evaluation grid) uses them as given.
The projections used are stored on the result.

`scale=nothing` resolves to `false` for univariate data (fda.usc `depth.RP`,
depth in `[0, 1/2]`) and `true` for multivariate data (fda.usc `depth.RPp`,
whose default `mdepth.TD` doubles the depth). For multivariate data, as in
`depth.RPp`, every component is projected on the same `z_j` and the Tukey
depths are averaged over components and projections; all components must
share a grid. fda.usc's `depth.RPp` projects with a plain sum rather than the
trapezoidal inner product; this package uses quadrature for both.

fda.usc evaluates `F(v-)` as `F(v - 1e-15)`, which equals `F(v)` once
`|v| > 16` in double precision, so its depth of a sample curve with respect
to its own sample then omits the curve itself from one side. This package
counts reference values `>= v` exactly.
"""
struct RandomProjectionDepth{R<:AbstractRNG} <: _DepthMethod
    nproj::Int
    projections::Union{Nothing,Matrix{Float64}}
    scale::Union{Nothing,Bool}
    rng::R
    function RandomProjectionDepth(; nproj::Union{Nothing,Integer}=nothing,
            projections::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
            scale::Union{Nothing,Bool}=nothing, rng::AbstractRNG=Random.default_rng())
        if projections === nothing
            count = nproj === nothing ? 50 : Int(nproj)
            count >= 1 || throw(ArgumentError("nproj must be a positive integer"))
            return new{typeof(rng)}(count, nothing, scale, rng)
        end
        Base.require_one_based_indexing(projections)
        z = Matrix{Float64}(projections)
        size(z, 1) >= 1 || throw(ArgumentError("projections need at least one row"))
        all(isfinite, z) || throw(ArgumentError("projections must be finite"))
        all(j -> any(!iszero, view(z, j, :)), axes(z, 1)) ||
            throw(ArgumentError("every projection must be a nonzero function"))
        nproj === nothing || nproj == size(z, 1) || throw(DimensionMismatch(
            "nproj ($nproj) differs from the number of supplied projections ($(size(z, 1)))"))
        new{typeof(rng)}(size(z, 1), z, scale, rng)
    end
end

"""
    ModalDepth(; bandwidth=nothing, quantile=nothing, kernel=GaussianKernel())

Modal (h-) depth (Cuevas, Febrero and Fraiman, 2007; fda.usc 2.2.0
`depth.mode`): `D(x) = Σ_j K(‖x - x_j‖ / h)` over the reference curves, with
the L² distance under the stored trapezoidal weights (fda.usc `metric.lp`,
trapezoidal rule). The sum is not normalized, as in fda.usc, so values grow
with the reference size. The kernel is a KernelSmoothing.jl kernel; the
default Gaussian kernel is fda.usc's `Ker.norm`.

The bandwidth is `bandwidth` when given; otherwise the `quantile` (R type 7)
of all reference-to-reference distances, diagonal zeros included, as fda.usc
computes it. `quantile=nothing` resolves to 0.15 for univariate data
(`depth.mode`) and 0.10 for multivariate data (`depth.modep`). Multivariate
distances are `sqrt(Σ_c ‖x_c - y_c‖²)` over components (fda.usc
`metric.ldata` with `method="euclidean"`); components may have different
grids. The resolved bandwidth is stored on the result. A zero quantile
(at most five curves at 0.15, nine at 0.10, since the diagonal zeros count)
throws; fda.usc then returns depth zero for every curve.
"""
struct ModalDepth{K<:KernelSmoothing.AbstractKernel} <: _DepthMethod
    bandwidth::Union{Nothing,Float64}
    quantile::Union{Nothing,Float64}
    kernel::K
    function ModalDepth(; bandwidth::Union{Nothing,Real}=nothing, quantile::Union{Nothing,Real}=nothing,
            kernel::KernelSmoothing.AbstractKernel=KernelSmoothing.GaussianKernel())
        bandwidth === nothing || quantile === nothing ||
            throw(ArgumentError("give either a bandwidth or a quantile, not both"))
        bandwidth === nothing || (isfinite(bandwidth) && bandwidth > 0) ||
            throw(ArgumentError("bandwidth must be finite and positive"))
        quantile === nothing || (isfinite(quantile) && 0 < quantile <= 1) ||
            throw(ArgumentError("quantile must lie in (0, 1]"))
        new{typeof(kernel)}(bandwidth === nothing ? nothing : Float64(bandwidth),
            quantile === nothing ? nothing : Float64(quantile), kernel)
    end
end

Base.show(io::IO, m::FraimanMuniz) =
    print(io, "FraimanMuniz(scale=", m.scale, ", integration=:", m.integration, ")")
Base.show(io::IO, m::BandDepth) = print(io, "BandDepth(definition=:", m.definition, ")")
Base.show(io::IO, m::ModifiedBandDepth) =
    print(io, "ModifiedBandDepth(ties=:", m.ties, ", integration=:", m.integration, ")")
Base.show(io::IO, m::RandomProjectionDepth) = print(io, "RandomProjectionDepth(nproj=", m.nproj,
    ", projections=", m.projections === nothing ? "generated" : "supplied", ", scale=", m.scale, ")")
Base.show(io::IO, m::ModalDepth) = print(io, "ModalDepth(bandwidth=", m.bandwidth,
    ", quantile=", m.quantile, ", kernel=", m.kernel, ")")

"""
    FunctionalDepth

Result of [`depth`](@ref). `values` holds one depth per target curve and
`ordering` the curve indices from deepest to least deep (ties keep curve
order). `median` is the index of the deepest curve (the first one on ties, as
R's `which.max`) and `median_curve` its values on the evaluation grid, as
`GridFunctionalData` or, for multivariate data, `MultivariateFunctionalData`.
`method` is the depth method; `reference` is `:sample` for the depth of a
sample with respect to itself and `:reference` for new curves, with
`nreference` reference curves. `representation` records how curves reached a
common grid (`:grid`, `:basis_evaluated` or `:fpca_reconstruction`; one entry
per component for multivariate data) and `fpca` the FPCA fits used
(`nothing` where none was needed). `settings` stores every resolved choice:
integration weights, ties, bandwidth, projections, component weights and the
evaluation grids. `ids` are the target subject labels.
"""
struct FunctionalDepth{T<:AbstractFloat,M<:_DepthMethod,C,R,F,S<:NamedTuple}
    values::Vector{T}
    ordering::Vector{Int}
    median::Int
    median_curve::C
    method::M
    reference::Symbol
    nreference::Int
    representation::R
    fpca::F
    ids::Union{Nothing,Vector{Any}}
    settings::S
end

nobs(d::FunctionalDepth) = length(d.values)

function Base.show(io::IO, d::FunctionalDepth)
    print(io, "FunctionalDepth(method=", d.method, ", n=", nobs(d), ", reference=:", d.reference,
        " (", d.nreference, " curves), representation=", d.representation, ", median=", d.median,
        ", settings=", _depth_show_settings(d.settings), ")")
end
Base.show(io::IO, ::MIME"text/plain", d::FunctionalDepth) = show(io, d)

_depth_show_settings(s::NamedTuple) = map(_depth_show_value, s)
_depth_show_value(x::AbstractVector{<:Real}) = length(x) <= 6 ? x : "$(length(x))-element vector"
_depth_show_value(x::AbstractVector{<:AbstractVector}) = map(_depth_show_value, x)
_depth_show_value(x::AbstractMatrix) = _show_summary(x)
_depth_show_value(x) = x

# --- Bringing curves to a common grid -------------------------------------

_depth_complete(fd::GridFunctionalData) = fd.observed === nothing
_depth_complete(::BasisFunctionalData) = true
_depth_complete(::IrregularFunctionalData) = false

_depth_input(x) = throw(ArgumentError(
    "depth needs GridFunctionalData, IrregularFunctionalData, BasisFunctionalData or an FPCA fit"))
_depth_input(x::Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData,FPCA}) = x

function _depth_grid_keyword(grid)
    grid === nothing && return nothing
    grid isa AbstractVector{<:Real} || throw(ArgumentError("grid must be a real vector"))
    _validate_strict_grid(grid)
    collect(float.(grid))
end

function _depth_fpca_options(fpca)
    fpca isa NamedTuple || throw(ArgumentError("fpca must be a named tuple of FPCA keyword arguments"))
    merge((; method=:pace), fpca)
end

# Curves on a common grid: (values, grid, representation, fit).
function _depth_curves(fd::GridFunctionalData, grid, fpca, allow_fit)
    if fd.observed === nothing
        grid === nothing || throw(ArgumentError(
            "grid data are evaluated on their own grid; convert with to_grid before choosing another"))
        isempty(fpca) || throw(ArgumentError("fpca options apply only to irregular or incomplete curves"))
        return (copy(fd.values), copy(fd.grid), :grid, nothing)
    end
    _depth_fitted_curves(fd, grid, fpca, allow_fit)
end
_depth_curves(fd::IrregularFunctionalData, grid, fpca, allow_fit) =
    _depth_fitted_curves(fd, grid, fpca, allow_fit)
function _depth_curves(fd::BasisFunctionalData, grid, fpca, allow_fit)
    grid === nothing && throw(ArgumentError("basis data need an explicit evaluation grid (grid keyword)"))
    isempty(fpca) || throw(ArgumentError("fpca options apply only to irregular or incomplete curves"))
    _component_domain(fd, grid)
    (evaluate(fd, grid), Vector{eltype(fd)}(grid), :basis_evaluated, nothing)
end
function _depth_curves(f::FPCA, grid, fpca, allow_fit)
    isempty(fpca) || throw(ArgumentError("fpca options cannot be applied to an existing FPCA fit"))
    curves = reconstruct(f; grid=grid === nothing ? f.grid : grid)
    (curves.values, curves.grid, :fpca_reconstruction, f)
end

function _depth_fitted_curves(fd, grid, fpca, allow_fit)
    allow_fit || throw(ArgumentError(
        "irregular or incomplete curves need an FPCA reconstruction; fit FPCA and pass the fit"))
    f = fit(FPCA, fd; _depth_fpca_options(fpca)...)
    _depth_curves(f, grid, (;), allow_fit)
end

# Target and reference on one grid: (target, reference, grid, representation, fit).
function _depth_curves(target, reference, grid, fpca, allow_fit)
    if reference isa FPCA || !_depth_complete(target) || !_depth_complete(reference)
        target isa FPCA && throw(ArgumentError("an FPCA fit can be the reference but not the target"))
        reference isa FPCA || allow_fit || throw(ArgumentError(
            "irregular or incomplete curves need an FPCA reconstruction; fit FPCA and pass the fit"))
        f = reference isa FPCA ? reference : fit(FPCA, reference; _depth_fpca_options(fpca)...)
        reference isa FPCA && !isempty(fpca) &&
            throw(ArgumentError("fpca options cannot be applied to an existing FPCA fit"))
        g = grid === nothing ? f.grid : grid
        ref = reconstruct(f; grid=g)
        new = reconstruct(f, project_scores(f, target); grid=g)
        return (new.values, ref.values, ref.grid, :fpca_reconstruction, f)
    end
    isempty(fpca) || throw(ArgumentError("fpca options apply only to irregular or incomplete curves"))
    target isa FPCA && throw(ArgumentError("an FPCA fit can be the reference but not the target"))
    shared = reference isa GridFunctionalData ? reference.grid :
        target isa GridFunctionalData ? target.grid : nothing
    if shared !== nothing
        grid === nothing || throw(ArgumentError(
            "grid data are evaluated on their own grid; convert with to_grid before choosing another"))
        for fd in (target, reference)
            fd isa GridFunctionalData && fd.grid != shared && throw(ArgumentError(
                "target and reference grid data must share a grid; convert with to_grid"))
        end
        g = shared
    else
        grid === nothing && throw(ArgumentError("basis data need an explicit evaluation grid (grid keyword)"))
        g = grid
    end
    x, gx, rx, _ = _depth_curves(target, target isa GridFunctionalData ? nothing : g, (;), allow_fit)
    y, _, ry, _ = _depth_curves(reference, reference isa GridFunctionalData ? nothing : g, (;), allow_fit)
    T = promote_type(eltype(x), eltype(y))
    representation = rx === ry ? rx : :basis_evaluated
    (Matrix{T}(x), Matrix{T}(y), Vector{T}(gx), representation, nothing)
end

# One prepared component: target and reference matrices on one grid.
function _depth_prepare_component(fd, ::Nothing, grid, fpca, allow_fit)
    x, g, representation, f = _depth_curves(_depth_input(fd), _depth_grid_keyword(grid), fpca, allow_fit)
    (; target=x, reference=x, grid=g, representation, fpca=f)
end
function _depth_prepare_component(fd, reference, grid, fpca, allow_fit)
    x, y, g, representation, f = _depth_curves(_depth_input(fd), _depth_input(reference),
        _depth_grid_keyword(grid), fpca, allow_fit)
    size(x, 2) == size(y, 2) || throw(DimensionMismatch("target and reference grids differ"))
    (; target=x, reference=y, grid=g, representation, fpca=f)
end

function _depth_component_grids(grid, p)
    grid === nothing && return fill(nothing, p)
    grid isa AbstractVector{<:Real} && return fill(grid, p)
    grid isa AbstractVector && length(grid) == p && return collect(grid)
    throw(DimensionMismatch("grid must be nothing, one real vector, or one grid per component ($p)"))
end

function _depth_prepare(fd::MultivariateFunctionalData, reference, grid, fpca, allow_fit)
    p = ncomponents(fd)
    reference === nothing || reference isa MultivariateFunctionalData || throw(ArgumentError(
        "the reference for multivariate data must be MultivariateFunctionalData"))
    reference === nothing || ncomponents(reference) == p || throw(DimensionMismatch(
        "target and reference have different numbers of components"))
    grids = _depth_component_grids(grid, p)
    references = [reference === nothing ? nothing : component(reference, j) for j in 1:p]
    # Shared FPCA options go to the components that need a reconstruction.
    fitted = [!_depth_complete(component(fd, j)) ||
        (references[j] !== nothing && !_depth_complete(references[j])) for j in 1:p]
    isempty(fpca) || any(fitted) ||
        throw(ArgumentError("fpca options apply only to irregular or incomplete curves"))
    [_depth_prepare_component(component(fd, j), references[j], grids[j], fitted[j] ? fpca : (;), allow_fit)
        for j in 1:p]
end
function _depth_prepare(fd, reference, grid, fpca, allow_fit)
    reference isa MultivariateFunctionalData &&
        throw(ArgumentError("multivariate references need MultivariateFunctionalData targets"))
    _depth_prepare_component(fd, reference, grid, fpca, allow_fit)
end

function _depth_weights(grid::Vector{T}, integration) where {T}
    integration === :uniform && return fill(one(T) / length(grid), length(grid))
    w = quadrature_weights(grid)
    w ./ sum(w)
end

function _depth_component_weights(weights, p)
    weights === nothing && return fill(1.0 / p, p)
    weights isa AbstractVector{<:Real} || throw(ArgumentError("weights must be a real vector"))
    length(weights) == p || throw(DimensionMismatch("weights need one entry per component ($p)"))
    w = Float64.(collect(weights))
    all(x -> isfinite(x) && x >= 0, w) || throw(ArgumentError("weights must be finite and nonnegative"))
    abs(sum(w) - 1) <= sqrt(eps(Float64)) || throw(ArgumentError("weights must sum to one"))
    w
end

function _depth_check_reference(n0)
    n0 >= 2 || throw(ArgumentError("depth needs at least two reference curves"))
end

# --- Rank-based pointwise depths --------------------------------------------

# Reference values strictly below and strictly above x in a sorted column.
@inline _depth_below(sorted, x) = searchsortedfirst(sorted, x) - 1
@inline _depth_above(sorted, x) = length(sorted) - searchsortedlast(sorted, x)

# Accumulate w[t] * f(sorted reference column, x(t)) over the grid.
function _depth_rank_sum(f, x::Matrix{T}, y::Matrix{T}, w) where {T}
    n, m = size(x)
    out = zeros(T, n)
    column = Vector{T}(undef, size(y, 1))
    for t in 1:m
        # Adding zero maps -0.0 to 0.0: sorting and searching use isless,
        # which would otherwise separate the two equal values.
        column .= view(y, :, t) .+ zero(T)
        sort!(column)
        for i in 1:n
            out[i] += w[t] * f(column, x[i, t] + zero(T))
        end
    end
    out
end

_pairs(k) = k * (k - 1) / 2

function _mbd(x::Matrix{T}, y::Matrix{T}, w, ties, self) where {T}
    n0 = size(y, 1)
    total = T(_pairs(n0))
    if ties === :exact
        out = _depth_rank_sum(x, y, w) do column, v
            T(total - _pairs(_depth_above(column, v)) - _pairs(_depth_below(column, v)))
        end
        return out ./ total
    end
    # roahd 1.4.3: average ranks within the sample, or of the target among
    # the target and reference curves for MBD_relative.
    out = _depth_rank_sum(x, y, w) do column, v
        below, above = _depth_below(column, v), _depth_above(column, v)
        equal = n0 - below - above
        rank = self ? below + (equal + 1) / 2 : below + (equal + 2) / 2
        T(self ? (n0 - rank) * (rank - 1) : (n0 + 1 - rank) * (rank - 1))
    end
    self && (out .+= n0 - 1)
    out ./ total
end

function _fraiman_muniz(x::Matrix{T}, y::Matrix{T}, w, scale) where {T}
    n0 = size(y, 1)
    _depth_rank_sum(x, y, w) do column, v
        d = 1 - abs(T(0.5) - T(searchsortedlast(column, v)) / n0)
        scale ? 2d - 1 : d
    end
end

function _band_depth_roahd(x::Matrix{T}, y::Matrix{T}, self) where {T}
    n, m = size(x)
    n0 = size(y, 1)
    out = Vector{T}(undef, n)
    for i in 1:n
        above = count(j -> all(t -> y[j, t] > x[i, t], 1:m), 1:n0)
        below = count(j -> all(t -> y[j, t] < x[i, t], 1:m), 1:n0)
        out[i] = (above * below + (self ? n0 - 1 : 0)) / T(_pairs(n0))
    end
    out
end

# A pair of reference curves encloses x unless both lie strictly above, or
# both strictly below, x at some grid point. Signs are packed into bit masks.
function _band_depth_graph(x::Matrix{T}, y::Matrix{T}) where {T}
    n, m = size(x)
    n0 = size(y, 1)
    chunks = cld(m, 64)
    up = zeros(UInt64, chunks, n0)
    down = zeros(UInt64, chunks, n0)
    out = Vector{T}(undef, n)
    for i in 1:n
        fill!(up, 0)
        fill!(down, 0)
        for j in 1:n0, t in 1:m
            chunk, bit = divrem(t - 1, 64)
            if y[j, t] > x[i, t]
                up[chunk + 1, j] |= UInt64(1) << bit
            elseif y[j, t] < x[i, t]
                down[chunk + 1, j] |= UInt64(1) << bit
            end
        end
        enclosing = 0
        for j in 1:n0-1, k in j+1:n0
            encloses = true
            for c in 1:chunks
                if (up[c, j] & up[c, k]) != 0 || (down[c, j] & down[c, k]) != 0
                    encloses = false
                    break
                end
            end
            enclosing += encloses
        end
        out[i] = enclosing / T(_pairs(n0))
    end
    out
end

# Tukey depth min(F(v), 1 - F(v-)) of the columns of xp among those of yp,
# averaged over columns.
function _projection_tukey(xp::Matrix{T}, yp::Matrix{T}) where {T}
    n0 = size(yp, 1)
    w = fill(one(T) / size(xp, 2), size(xp, 2))
    _depth_rank_sum(xp, yp, w) do column, v
        T(min(searchsortedlast(column, v), n0 - _depth_below(column, v))) / n0
    end
end

# Square root R (R R' = C) of the exponential covariance C(s, t) = exp(-|s - t|/θ),
# θ = 0.2 × domain length, used to draw the default projections.
function _projection_root(grid::Vector{T}) where {T}
    theta = T(0.2) * (last(grid) - first(grid))
    covariance = [exp(-abs(s - t) / theta) for s in grid, t in grid]
    system = eigen(Symmetric(covariance))
    system.vectors * Diagonal(sqrt.(max.(system.values, zero(T))))
end

function _depth_projections(method::RandomProjectionDepth, grid::Vector{T}) where {T}
    if method.projections !== nothing
        size(method.projections, 2) == length(grid) || throw(DimensionMismatch(
            "projections need one column per evaluation grid point ($(length(grid)))"))
        return Matrix{T}(method.projections)
    end
    z = randn(method.rng, T, method.nproj, length(grid)) * transpose(_projection_root(grid))
    w = quadrature_weights(grid)
    for j in axes(z, 1)
        scale = sqrt(sum(w .* abs2.(view(z, j, :))))
        scale > 0 || throw(ArgumentError("a generated projection has zero norm"))
        z[j, :] ./= scale
    end
    z
end

function _depth_distances(x::Matrix{T}, y::Matrix{T}, grid) where {T}
    w = quadrature_weights(grid)
    [sqrt(sum(w[t] * abs2(x[i, t] - y[j, t]) for t in eachindex(w))) for i in axes(x, 1), j in axes(y, 1)]
end

function _modal_depth(distances::Matrix{T}, reference_distances, method, default_quantile) where {T}
    q = method.quantile === nothing ? default_quantile : method.quantile
    h = if method.bandwidth === nothing
        T(_pfr_quantile(sort(vec(reference_distances)), q))
    else
        T(method.bandwidth)
    end
    h > 0 || throw(ArgumentError(
        "the bandwidth quantile of the reference distances is zero; give a positive bandwidth"))
    values = vec(sum(method.kernel.(distances ./ h); dims=2))
    (values, (; bandwidth=h, quantile=method.bandwidth === nothing ? q : nothing,
        kernel=method.kernel, distance=:L2_trapezoidal))
end

# --- Univariate depths --------------------------------------------------------

function _depth_values(method::FraimanMuniz, c, self)
    scale = method.scale === nothing ? true : method.scale
    w = _depth_weights(c.grid, method.integration)
    (_fraiman_muniz(c.target, c.reference, w, scale),
        (; scale, integration=method.integration, pointwise=:fraiman_muniz))
end

function _depth_values(method::BandDepth, c, self)
    values = method.definition === :graph ? _band_depth_graph(c.target, c.reference) :
        _band_depth_roahd(c.target, c.reference, self)
    (values, (; definition=method.definition))
end

function _depth_values(method::ModifiedBandDepth, c, self)
    w = _depth_weights(c.grid, method.integration)
    (_mbd(c.target, c.reference, w, method.ties, self), (; ties=method.ties, integration=method.integration))
end

function _depth_values(method::RandomProjectionDepth, c, self)
    z = _depth_projections(method, c.grid)
    scale = method.scale === nothing ? false : method.scale
    projector = quadrature_weights(c.grid) .* transpose(z)
    values = _projection_tukey(c.target * projector, c.reference * projector)
    scale && (values .*= 2)
    (values, (; nproj=size(z, 1), projections=z, generated=method.projections === nothing,
        scale, univariate=:tukey, inner_product=:trapezoidal))
end

function _depth_values(method::ModalDepth, c, self)
    reference = _depth_distances(c.reference, c.reference, c.grid)
    distances = self ? reference : _depth_distances(c.target, c.reference, c.grid)
    _modal_depth(distances, reference, method, 0.15)
end

# --- Multivariate depths ------------------------------------------------------

function _depth_common_grid(components, name)
    grid = first(components).grid
    all(c -> c.grid == grid, components) || throw(ArgumentError(
        "multivariate $name needs every component on the same evaluation grid"))
    grid
end

function _depth_weighted(method::Union{BandDepth,ModifiedBandDepth}, components, self, weights)
    w = _depth_component_weights(weights, length(components))
    parts = [_depth_values(method, c, self) for c in components]
    values = sum(w[j] .* first(parts[j]) for j in eachindex(parts))
    (values, merge(last(first(parts)), (; component_weights=w)))
end
_depth_values(method::Union{BandDepth,ModifiedBandDepth}, components::Vector, self, weights) =
    _depth_weighted(method, components, self, weights)

function _depth_values(method::FraimanMuniz, components::Vector, self, weights)
    weights === nothing || throw(ArgumentError("component weights apply only to BandDepth and ModifiedBandDepth"))
    method.scale === nothing || throw(ArgumentError(
        "multivariate FraimanMuniz uses the Mahalanobis pointwise depth; leave scale at nothing"))
    grid = _depth_common_grid(components, "FraimanMuniz")
    w = _depth_weights(grid, method.integration)
    T = eltype(first(components).target)
    n, n0, p = size(first(components).target, 1), size(first(components).reference, 1), length(components)
    n0 > p || throw(ArgumentError("multivariate FraimanMuniz needs more reference curves than components"))
    values = zeros(T, n)
    for t in eachindex(grid)
        y = reduce(hcat, [view(c.reference, :, t) for c in components])
        x = reduce(hcat, [view(c.target, :, t) for c in components])
        mu = vec(sum(y; dims=1)) ./ n0
        centered = y .- transpose(mu)
        factor = cholesky(Symmetric(transpose(centered) * centered ./ (n0 - 1)); check=false)
        issuccess(factor) || throw(ArgumentError(
            "the reference covariance of the components is singular at grid point $t"))
        distance = vec(sum(abs2, (x .- transpose(mu)) / factor.U; dims=2))
        values .+= w[t] ./ (1 .+ distance)
    end
    (values, (; scale=nothing, integration=method.integration, pointwise=:mahalanobis))
end

function _depth_values(method::RandomProjectionDepth, components::Vector, self, weights)
    weights === nothing || throw(ArgumentError("component weights apply only to BandDepth and ModifiedBandDepth"))
    grid = _depth_common_grid(components, "RandomProjectionDepth")
    z = _depth_projections(method, grid)
    scale = method.scale === nothing ? true : method.scale
    projector = quadrature_weights(grid) .* transpose(z)
    values = sum(_projection_tukey(c.target * projector, c.reference * projector) for c in components) ./
        length(components)
    scale && (values .*= 2)
    (values, (; nproj=size(z, 1), projections=z, generated=method.projections === nothing,
        scale, univariate=:tukey, inner_product=:trapezoidal))
end

function _depth_values(method::ModalDepth, components::Vector, self, weights)
    weights === nothing || throw(ArgumentError("component weights apply only to BandDepth and ModifiedBandDepth"))
    reference = sqrt.(sum(abs2.(_depth_distances(c.reference, c.reference, c.grid)) for c in components))
    distances = self ? reference :
        sqrt.(sum(abs2.(_depth_distances(c.target, c.reference, c.grid)) for c in components))
    values, settings = _modal_depth(distances, reference, method, 0.10)
    (values, merge(settings, (; distance=:euclidean_L2_trapezoidal)))
end

_depth_values(method::_DepthMethod, c::NamedTuple, self, weights) =
    weights === nothing ? _depth_values(method, c, self) :
        throw(ArgumentError("component weights apply only to multivariate data"))

# --- Public API -----------------------------------------------------------------

_depth_median_curve(c::NamedTuple, k) = GridFunctionalData(c.target[k:k, :], c.grid)
_depth_median_curve(components::Vector, k) =
    MultivariateFunctionalData([GridFunctionalData(c.target[k:k, :], c.grid) for c in components])
_depth_field(c::NamedTuple, name) = getproperty(c, name)
_depth_field(components::Vector, name) = [getproperty(c, name) for c in components]
_depth_sizes(c::NamedTuple) = (size(c.target, 1), size(c.reference, 1))
_depth_sizes(components::Vector) = _depth_sizes(first(components))

function _depth_result(method, prepared, self, ids, weights)
    n, n0 = _depth_sizes(prepared)
    _depth_check_reference(n0)
    values, settings = _depth_values(method, prepared, self, weights)
    all(isfinite, values) || throw(ArgumentError("depth values must be finite; rescale the curves"))
    ordering = sortperm(values; rev=true)
    k = first(ordering)
    settings = merge(settings, (; grid=_depth_field(prepared, :grid)))
    FunctionalDepth(values, ordering, k, _depth_median_curve(prepared, k), method,
        self ? :sample : :reference, n0, _depth_field(prepared, :representation),
        _depth_field(prepared, :fpca), ids, settings)
end

"""
    depth(fd, method; weights=nothing, grid=nothing, fpca=(;))
    depth(fd, reference, method; weights=nothing, grid=nothing, fpca=(;))

Functional depth of the curves in `fd` with respect to their own sample, or of
new curves `fd` with respect to the curves in `reference`. `method` is one of
[`FraimanMuniz`](@ref), [`BandDepth`](@ref), [`ModifiedBandDepth`](@ref),
[`RandomProjectionDepth`](@ref) and [`ModalDepth`](@ref). Returns a
[`FunctionalDepth`](@ref) with the depths, the ordering, the deepest curve and
the settings used. For every method except roahd's compatibility options
(`BandDepth(definition=:roahd)`, `ModifiedBandDepth(ties=:average)`), the depth
of a sample equals its depth with respect to an identical reference (for
random projections, with the same projections).

Depths compare curves on a common grid:
- complete `GridFunctionalData` uses its grid; target and reference must share it;
- `BasisFunctionalData` is evaluated exactly on `grid`, which is then required
  (or taken from grid data on the other side);
- `IrregularFunctionalData`, or grid data with missing values, is represented by
  its FPCA reconstruction on the work grid (or `grid`). FPCA is fitted to the
  reference curves with `fit(FPCA, reference; method=:pace, fpca...)`, so PACE
  is the default and `fpca` passes any other FPCA keywords; target curves are
  reconstructed from `project_scores`. When either side needs FPCA, both sides
  use the reconstruction. An existing `FPCA` fit can be passed as `fd` or as
  `reference` instead.
The representation and the FPCA fit are recorded on the result.

For `MultivariateFunctionalData` each component is brought to a grid in the same
way (`grid` may be one grid or one per component). `weights` (one nonnegative
weight per component, summing to one; uniform by default as in roahd) apply to
`BandDepth` and `ModifiedBandDepth`, which average component depths; the other
methods follow fda.usc's multivariate versions described in their docstrings.
"""
function depth(fd, method::_DepthMethod; weights=nothing, grid=nothing, fpca::NamedTuple=(;))
    prepared = _depth_prepare(fd, nothing, grid, fpca, true)
    _depth_result(method, prepared, true, ids(fd), weights)
end

function depth(fd, reference, method::_DepthMethod; weights=nothing, grid=nothing, fpca::NamedTuple=(;))
    prepared = _depth_prepare(fd, reference, grid, fpca, true)
    _depth_result(method, prepared, false, fd isa FPCA ? nothing : ids(fd), weights)
end

# Modified epigraph (count >= x) and hypograph (count <= x) indices.
function _graph_index(fd, reference, kind, integration, grid, weights)
    _depth_integration(integration)
    prepared = _depth_prepare(fd, reference, grid, (;), false)
    components = prepared isa Vector ? prepared : [prepared]
    w = prepared isa Vector ? _depth_component_weights(weights, length(components)) :
        weights === nothing ? [1.0] : throw(ArgumentError("component weights apply only to multivariate data"))
    total = nothing
    for (j, c) in enumerate(components)
        n0 = size(c.reference, 1)
        part = _depth_rank_sum(c.target, c.reference, _depth_weights(c.grid, integration)) do column, v
            kind === :epigraph ? (n0 - _depth_below(column, v)) / n0 : searchsortedlast(column, v) / n0
        end
        total = total === nothing ? w[j] .* part : total .+ w[j] .* part
    end
    total
end

"""
    modified_epigraph_index(fd; integration=:quadrature, grid=nothing, weights=nothing)
    modified_epigraph_index(fd, reference; integration=:quadrature, grid=nothing, weights=nothing)

Modified epigraph index (López-Pintado and Romo, 2011): the average over the
grid of the proportion of reference curves lying on or above each curve,
`MEI(x) = Σ_j λ{t : x_j(t) >= x(t)} / (n λ(I))`. For a sample the curve itself
counts. Low values mark curves high in the sample. It equals roahd 1.4.3 `MEI`
with `integration=:uniform`; the default uses the stored trapezoidal weights,
as in [`ModifiedBandDepth`](@ref). Multivariate data give the weighted
average of component indices (roahd `multiMEI`; uniform weights by default).

Curves must be complete grid data, basis data on `grid`, or an `FPCA` fit
(its reconstruction); irregular or incomplete curves throw, since an index
needs an explicit common-grid representation. Returns a vector.
"""
modified_epigraph_index(fd; integration::Symbol=:quadrature, grid=nothing, weights=nothing) =
    _graph_index(fd, nothing, :epigraph, integration, grid, weights)
modified_epigraph_index(fd, reference; integration::Symbol=:quadrature, grid=nothing, weights=nothing) =
    _graph_index(fd, reference, :epigraph, integration, grid, weights)

"""
    modified_hypograph_index(fd; integration=:quadrature, grid=nothing, weights=nothing)
    modified_hypograph_index(fd, reference; integration=:quadrature, grid=nothing, weights=nothing)

Modified hypograph index (López-Pintado and Romo, 2009): the average over the
grid of the proportion of reference curves lying on or below each curve,
`MHI(x) = Σ_j λ{t : x_j(t) <= x(t)} / (n λ(I))`. It equals roahd 1.4.3 `MHI`
with `integration=:uniform`. Inputs, weights and defaults are as in
[`modified_epigraph_index`](@ref).
"""
modified_hypograph_index(fd; integration::Symbol=:quadrature, grid=nothing, weights=nothing) =
    _graph_index(fd, nothing, :hypograph, integration, grid, weights)
modified_hypograph_index(fd, reference; integration::Symbol=:quadrature, grid=nothing, weights=nothing) =
    _graph_index(fd, reference, :hypograph, integration, grid, weights)
