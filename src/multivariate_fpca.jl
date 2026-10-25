"""
    MultivariateFPCA
    fit(FPCA, mfd::MultivariateFunctionalData; basis=:marginal, kwargs...)

Multivariate FPCA with one fit per node (`basis=:marginal`, the package's
default for nodewise analysis) or one shared basis (`basis=:pooled`). Marginal
fits delegate to univariate [`FPCA`](@ref), including its dense/PACE/FACE routing
and all defaults. Each keyword can be shared or an `AbstractVector` of one
value per node. A flat real `grid` vector is a shared work grid; use a vector
of grids (or `nothing`) for per-node grids. Supplied means and covariances use
the same fitted-object or named-tuple forms as univariate FPCA. Wrong vector
lengths throw `DimensionMismatch`.

Pooled fits retain copied original observations in `training_data` for nodewise
derivative covariance and score estimation. Marginal fits retain observations
in their individual univariate fits.

Pooled FPCA requires complete grid data on the same grid; pooled PACE is not
implemented. It averages node sample covariances with denominator `n-1`, then
reuses the weighted univariate eigenstep. Each node is centered by its own
mean. Node eigenvalues are score sample variances with denominator `n-1`,
ordered by the shared pooled spectrum, rather than necessarily decreasing.
The pooled spectrum, selection, dropped spectrum, sign rule, and settings
are stored in `f.pooled`. The sign reference is the average node mean;
the univariate largest-entry fallback resolves near-zero alignment.

Pooled defaults `ncomponents=FVE(0.99)`, `maxK=nothing`, `eigen_tol=1e-10`,
and `sign_tol=1e-12` follow the dense univariate defaults and have the
sources given in [`FPCA`](@ref). The pooled covariance averages `p` node
covariances of rank at most `nobs-1`, so its rank is at most `p*(nobs-1)`;
`maxK` therefore resolves to `max(1,min(grid_length-2,p*(nobs-1)-1))`, the
dense rule with that rank, with explicit cap `min(grid_length,p*(nobs-1))`. Only shared values of these four keywords and
`method=:auto`/`:dense` are supported for pooled fits. PACE tuning and
`presmooth` throw instead of being ignored. Inputs and settings are copied.

The covariance averaging follows partial separability in Zapata, Oh and
Petersen (2022), doi:10.1093/biomet/asab046. [`score_array`](@ref) supplies
FunctionalGraphicalModels.jl's subject × node × component ordering. That
package's reference covariance divisor is `n`, so its dense eigenvalues
must be multiplied by `n/(n-1)` for comparison.
"""
struct MultivariateFPCA{T<:AbstractFloat}
    basis::Symbol
    fits::Vector{FPCA}
    pooled::Union{Nothing,NamedTuple}
    scores::Vector{Matrix{T}}
    score_covariances::Vector{Union{Nothing,Array{T,3}}}
    eigenvalues::Vector{Vector{T}}
    bases::Vector{FPCBasis{T}}
    means::Vector{GridFunctionalData{T}}
    ids::Union{Nothing,Vector{Any}}
    settings::NamedTuple
    training_data::Union{Nothing,MultivariateFunctionalData{T}}
end

function _multivariate_keyword(value, name, p)
    if value isa AbstractVector && !(name === :grid && value isa AbstractVector{<:Real})
        Base.require_one_based_indexing(value)
        length(value) == p || throw(DimensionMismatch(
            "$name must be a shared value or a vector with one entry per node ($p)"))
        return true
    end
    false
end

function _multivariate_node_fit(fd::Union{GridFunctionalData,IrregularFunctionalData}; kwargs...)
    fit(FPCA, fd; kwargs...)
end
_multivariate_node_fit(fd; kwargs...) = throw(ArgumentError(
    "marginal FPCA requires grid or irregular nodes; convert basis data explicitly first"))

_multivariate_uncertainty(::Type{T}, ::Nothing) where {T} = nothing
_multivariate_uncertainty(::Type{T}, covariance::AbstractArray{<:Real,3}) where {T} =
    convert(Array{T,3}, covariance)

function _multivariate_marginal(mfd, options)
    p = ncomponents(mfd)
    per_node = map((name, value) -> _multivariate_keyword(value, name, p),
        keys(options), values(options))
    fitted = map(1:p) do j
        node_options = NamedTuple{keys(options)}(map((value, split) ->
            split ? value[j] : value, values(options), per_node))
        _multivariate_node_fit(component(mfd, j); node_options...)
    end
    T = mapreduce(f -> eltype(f.scores), promote_type, fitted)
    projected = Matrix{T}[scores(f) for f in fitted]
    uncertainties = Union{Nothing,Array{T,3}}[
        _multivariate_uncertainty(T, score_covariances(f)) for f in fitted]
    eigenvalues = Vector{T}[eigvals(f) for f in fitted]
    bases = FPCBasis{T}[f.basis isa FPCBasis{T} ? f.basis :
        FPCBasis(Vector{T}(f.grid), Matrix{T}(f.basis.values);
            weights=Vector{T}(f.weights)) for f in fitted]
    means = GridFunctionalData{T}[f.mean isa GridFunctionalData{T} ? f.mean :
        GridFunctionalData(Matrix{T}(f.mean.values), Vector{T}(f.grid)) for f in fitted]
    settings = (; requested=deepcopy(options), node_settings=[deepcopy(f.settings) for f in fitted])
    MultivariateFPCA(:marginal, FPCA[f for f in fitted], nothing, projected, uncertainties,
        eigenvalues, bases, means, ids(mfd), settings, nothing)
end

function _multivariate_pooled(mfd::MultivariateFunctionalData{T};
        ncomponents=FVE(0.99), maxK=nothing, eigen_tol=1e-10, sign_tol=1e-12,
        method=:auto, kwargs...) where {T}
    isempty(kwargs) || throw(ArgumentError(
        "pooled FPCA supports only ncomponents, maxK, eigen_tol, sign_tol and method; PACE tuning and presmooth are unsupported"))
    method in (:auto, :dense) || throw(ArgumentError(
        "pooled FPCA requires method=:auto or :dense; pooled PACE is not implemented"))
    for (name, value) in pairs((; ncomponents, maxK, eigen_tol, sign_tol))
        value isa AbstractVector && throw(ArgumentError("pooled FPCA $name must be one shared value"))
    end
    all(c -> c isa GridFunctionalData && c.observed === nothing, mfd.components) ||
        throw(ArgumentError("pooled FPCA requires complete GridFunctionalData nodes; pooled PACE is not implemented"))
    firstnode = first(mfd.components)::GridFunctionalData{T}
    all(c -> c.grid == firstnode.grid, mfd.components) || throw(ArgumentError(
        "pooled FPCA requires the same grid for every node; pooled PACE is not implemented"))
    all(c -> c.weights == firstnode.weights, mfd.components) || throw(ArgumentError(
        "pooled FPCA requires the same stored quadrature weights for every node"))
    n, g = size(firstnode.values)
    n >= 2 || throw(ArgumentError("pooled FPCA requires at least two curves"))
    p = length(mfd.components)
    # The pooled covariance averages p rank-(n-1) node covariances, so its rank
    # is at most p(n-1); the defaults mirror the dense rule with that rank.
    rank = p * (n - 1)
    requested_maxK = maxK
    maxK = maxK === nothing ? max(1, min(g-2, rank-1)) : maxK
    maxK isa Integer && 1 <= maxK <= min(g, rank) || throw(ArgumentError(
        "pooled FPCA maxK must be between 1 and min(grid length, nodes*(nobs-1))"))
    grid, weights = copy(firstnode.grid), copy(firstnode.weights)
    means = GridFunctionalData{T}[]
    residuals = Matrix{T}[]
    covariance = zeros(T, g, g)
    for node in mfd.components
        values = (node::GridFunctionalData{T}).values
        mu = sum(values; dims=1) ./ n
        centered = values .- mu
        push!(means, GridFunctionalData(mu, grid))
        push!(residuals, centered)
        covariance .+= (transpose(centered) * centered / (n-1)) / p
    end
    reference_mean = vec(sum(mu.values ./ p for mu in means))
    system = _fpca_eigen(covariance, weights, reference_mean; maxK, eigen_tol, sign_tol)
    selection = _fpca_selection(ncomponents, system.fve, Int(maxK))
    k = selection.chosen
    shared = FPCBasis(grid, system.eigenfunctions[:, 1:k]; weights)
    projected = Matrix{T}[centered * (weights .* shared.values) for centered in residuals]
    all(s -> all(isfinite, s), projected) || throw(ArgumentError(
        "pooled FPCA scores must be finite; rescale the observations"))
    eigenvalues = Vector{T}[]
    for s in projected
        centered = s .- sum(s; dims=1) ./ n
        variance = vec(sum(abs2, centered; dims=1) ./ (n-1))
        all(isfinite, variance) || throw(ArgumentError(
            "pooled FPCA node score variances must be finite; rescale the observations"))
        push!(eigenvalues, variance)
    end
    sign_rule = (; method=:mean_alignment, reference=:average_node_mean,
        fallback=:largest_absolute_entry, tolerance=sign_tol)
    pooled = (; grid, weights, covariance, eigenvalues=system.eigenvalues[1:k],
        all_eigenvalues=system.eigenvalues, fve=system.fve, basis=shared,
        selection, dropped=system.dropped, sign_rule, reference_mean)
    settings = (; requested_method=method, ncomponents=deepcopy(ncomponents),
        requested_maxK, maxK=Int(maxK), eigen_tol, sign_tol,
        grid_source=:data, quadrature=:stored_trapezoidal,
        covariance_denominator=n-1, score_variance_denominator=n-1,
        covariance_pooling=:equal_nodes, sign_reference=:average_node_mean,
        presmooth=nothing, observations=:copied_original)
    MultivariateFPCA(:pooled, FPCA[], pooled, projected,
        Union{Nothing,Array{T,3}}[nothing for _ in 1:p], eigenvalues,
        fill(shared, p), means, ids(mfd), settings,
        MultivariateFunctionalData{T}(mfd.components))
end

Base.@constprop :aggressive function fit(::Type{FPCA}, mfd::MultivariateFunctionalData;
        basis=:marginal, kwargs...)
    basis === :marginal && return _multivariate_marginal(mfd, (; kwargs...))
    basis === :pooled && return _multivariate_pooled(mfd; kwargs...)
    throw(ArgumentError("multivariate FPCA basis must be :marginal or :pooled"))
end

"""
    ncomponents(f::MultivariateFPCA)

Return the vector of selected component counts in node order.
"""
ncomponents(f::MultivariateFPCA) = Int[size(s, 2) for s in f.scores]
nobs(f::MultivariateFPCA) = size(first(f.scores), 1)
ids(f::MultivariateFPCA) = f.ids === nothing ? nothing : deepcopy(f.ids)

"""
    component(f::MultivariateFPCA, j)

Return node `j`'s stored univariate fit for marginal FPCA. Pooled fits throw:
their shared basis generally does not diagonalize individual node covariance.
"""
function component(f::MultivariateFPCA, j::Integer)
    f.basis === :marginal || throw(ArgumentError(
        "component(f, j) is only available for marginal FPCA; pooled fits have a shared basis"))
    f.fits[j]
end

"""
    scores(f::MultivariateFPCA)
    scores(f::MultivariateFPCA, j)

Return a vector of training score matrices, or node `j`'s matrix. Each has
subjects in rows and components in columns. Arrays are stored by the fit;
use `deepcopy` for independent storage.
"""
scores(f::MultivariateFPCA) = f.scores
scores(f::MultivariateFPCA, j::Integer) = f.scores[j]

"""
    score_array(f::MultivariateFPCA)

Return a copied `n × p × M` training score array in the same subject/node order
as [`to_array`](@ref), for FunctionalGraphicalModels.jl. Every node must have
the same component count. Unequal counts throw; use integer `ncomponents`
when a common count is needed. Scores are never padded or truncated.
"""
function score_array(f::MultivariateFPCA{T}) where {T}
    counts = ncomponents(f)
    all(==(first(counts)), counts) || throw(ArgumentError(
        "score_array requires equal component counts; fit with an integer ncomponents shared by all nodes"))
    out = Array{T}(undef, nobs(f), length(counts), first(counts))
    for j in eachindex(f.scores)
        out[:, j, :] = f.scores[j]
    end
    out
end

"""
    score_covariances(f::MultivariateFPCA)

Return conditional score covariances per node: `K_j × K_j × n` for PACE,
`nothing` for dense nodes (including pooled fits). These are stored arrays.
"""
score_covariances(f::MultivariateFPCA) = f.score_covariances

"""
    eigvals(f::MultivariateFPCA)

Return eigenvalue vectors in node order. For pooled fits these are sample
variances (`n-1` denominator) of each node's scores in pooled-spectrum order.
The pooled eigenvalues themselves are stored in `f.pooled.eigenvalues`.
"""
LinearAlgebra.eigvals(f::MultivariateFPCA) = f.eigenvalues

"""
    eigenfunctions(f::MultivariateFPCA)

Return a vector of selected [`FPCBasis`](@ref) objects, in node order.
Pooled fits return the same shared basis for every node.
"""
eigenfunctions(f::MultivariateFPCA) = f.bases

const _MultivariatePrediction{T} = Union{Matrix{T},
    NamedTuple{(:scores, :covariances),Tuple{Matrix{T},Array{T,3}}}}

_multivariate_prediction(::Type{T}, projected::AbstractMatrix) where {T} =
    convert(Matrix{T}, projected)
_multivariate_prediction(::Type{T}, predicted::NamedTuple) where {T} =
    (; scores=convert(Matrix{T}, predicted.scores),
        covariances=convert(Array{T,3}, predicted.covariances))

"""
    predict(f::MultivariateFPCA, newmfd::MultivariateFunctionalData)

Predict node by node with the training means and bases. Return a vector:
dense nodes give score matrices and PACE/FACE nodes give `(scores, covariances)`
named tuples, as in univariate FPCA. Node counts must match. Dense nodes
require their training grids and complete observations.
"""
function predict(f::MultivariateFPCA{T}, mfd::MultivariateFunctionalData{S}) where {T,S}
    ncomponents(mfd) == length(f.scores) || throw(DimensionMismatch(
        "prediction node count must match the multivariate FPCA fit"))
    R = promote_type(T, S)
    out = _MultivariatePrediction{R}[]
    for j in eachindex(f.scores)
        node = component(mfd, j)
        if f.basis === :marginal
            node isa Union{GridFunctionalData,IrregularFunctionalData} || throw(ArgumentError(
                "multivariate FPCA prediction requires grid or irregular nodes"))
            push!(out, _multivariate_prediction(R, predict(f.fits[j], node)))
        else
            node isa GridFunctionalData && node.observed === nothing || throw(ArgumentError(
                "pooled FPCA prediction requires complete grid nodes; pooled PACE is not implemented"))
            b = f.bases[j]
            length(node.grid) == length(b.grid) || throw(DimensionMismatch(
                "prediction grid length must match the pooled FPCA training grid"))
            node.grid == b.grid || throw(ArgumentError(
                "prediction grid must exactly match the pooled FPCA training grid"))
            projected = (node.values .- f.means[j].values) * (b.weights .* b.values)
            all(isfinite, projected) || throw(ArgumentError("pooled FPCA prediction scores must be finite"))
            push!(out, projected)
        end
    end
    out
end

function _multivariate_grids(f, grid)
    p = length(f.scores)
    grid === nothing && return [b.grid for b in f.bases]
    if grid isa AbstractVector{<:Real}
        return fill(grid, p)
    end
    grid isa AbstractVector || throw(ArgumentError(
        "reconstruction grid must be one real vector or a vector of grids"))
    Base.require_one_based_indexing(grid)
    length(grid) == p || throw(DimensionMismatch("reconstruction grids must have one entry per node"))
    all(g -> g isa AbstractVector{<:Real}, grid) || throw(ArgumentError(
        "each reconstruction grid must be a real vector"))
    grid
end

_multivariate_score_type(::Type{T}, ::AbstractVector{<:AbstractMatrix{S}}) where {T,S<:Real} =
    promote_type(T, _floattype(S))
_multivariate_score_type(::Type{T}, projected) where {T} =
    mapreduce(s -> _floattype(eltype(s)), promote_type, projected; init=T)
_multivariate_grid_type(::Type{T}, ::Nothing) where {T} = T
_multivariate_grid_type(::Type{T}, ::AbstractVector{S}) where {T,S<:Real} =
    promote_type(T, _floattype(S))
_multivariate_grid_type(::Type{T}, ::AbstractVector{<:AbstractVector{S}}) where {T,S<:Real} =
    promote_type(T, _floattype(S))
_multivariate_grid_type(::Type{T}, grids) where {T} =
    mapreduce(g -> _floattype(eltype(g)), promote_type, grids; init=T)

"""
    reconstruct(f::MultivariateFPCA, scores=scores(f); grid=nothing)

Reconstruct [`MultivariateFunctionalData`](@ref) from one score matrix per
node. A real grid vector applies to every node; a vector of grids applies
nodewise. The default uses each training grid. Every grid must be increasing
and inside its training domain. Score row counts must agree. Training IDs
are retained only if every score matrix is the stored training matrix;
copied or new score matrices have no inferred IDs.
"""
function reconstruct(f::MultivariateFPCA{T}, projected::AbstractVector=scores(f);
        grid=nothing) where {T}
    Base.require_one_based_indexing(projected)
    p = length(f.scores)
    length(projected) == p || throw(DimensionMismatch("reconstruction scores must have one matrix per node"))
    all(s -> s isa AbstractMatrix{<:Real}, projected) || throw(ArgumentError(
        "reconstruction scores must be real matrices"))
    n = size(first(projected), 1)
    n >= 1 || throw(ArgumentError("reconstruction requires at least one score row"))
    all(s -> size(s, 1) == n, projected) || throw(DimensionMismatch(
        "reconstruction score matrices must have the same number of rows"))
    grids = _multivariate_grids(f, grid)
    R = promote_type(_multivariate_score_type(T, projected), _multivariate_grid_type(T, grid))
    training = all(j -> projected[j] === f.scores[j], 1:p)
    nodes = AbstractFunctionalData{R}[]
    for j in 1:p
        s = projected[j]
        Base.require_one_based_indexing(s)
        size(s, 2) == size(f.scores[j], 2) || throw(DimensionMismatch(
            "reconstruction score columns must equal ncomponents at node $j"))
        all(isfinite, s) || throw(ArgumentError("reconstruction scores at node $j must be finite"))
        points = grids[j]
        quadrature_weights(points)
        functions = basismatrix(f.bases[j], points)
        mu = evaluate(f.means[j], points, PiecewiseLinear())
        values = s * transpose(functions) .+ mu
        push!(nodes, GridFunctionalData(Matrix{R}(values), Vector{R}(points);
            ids=training ? f.ids : nothing))
    end
    MultivariateFunctionalData{R}(nodes)
end

function Base.show(io::IO, f::MultivariateFPCA)
    print(io, "MultivariateFPCA(basis=", f.basis, ", n=", nobs(f),
        ", nodes=", length(f.scores), ", ncomponents=", ncomponents(f),
        ", settings=", f.settings)
    if f.basis === :marginal
        print(io, ", fits=", f.fits)
    else
        print(io, ", pooled=", f.pooled)
    end
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::MultivariateFPCA) = show(io, f)
