# Conservative storage checks include nested data, bases, means, settings and
# metadata. Numeric arrays are checked as storage, without visiting elements.
function _buffer_mightalias(out, source, seen)::Bool
    if source isa AbstractArray
        Base.mightalias(out, source) && return true
        eltype(source) <: Number && return false
        isbitstype(eltype(source)) && return false
    elseif isbits(source) || source isa Union{Number,AbstractString,Symbol,Type,Function,Module}
        return false
    end
    haskey(seen, source) && return false
    seen[source] = nothing
    if source isa Union{AbstractArray,Tuple,NamedTuple}
        return any(x -> _buffer_mightalias(out, x, seen), source)
    elseif source isa AbstractDict
        return any(pair -> _buffer_mightalias(out, first(pair), seen) ||
            _buffer_mightalias(out, last(pair), seen), source)
    end
    any(i -> isdefined(source, i) && _buffer_mightalias(out, getfield(source, i), seen),
        1:fieldcount(typeof(source)))
end

function _check_buffer(out, shape, ::Type{T}, sources...) where {T}
    Base.require_one_based_indexing(out)
    size(out) == shape || throw(DimensionMismatch("output buffer must have size $shape"))
    eltype(out) === T || throw(ArgumentError("output buffer element type must be $T, matching the allocating result"))
    seen = IdDict{Any,Nothing}()
    any(source -> _buffer_mightalias(out, source, seen), sources) &&
        throw(ArgumentError("output buffer must not alias input or model storage"))
    out
end

function _check_node_buffers(out::AbstractVector, p)
    Base.require_one_based_indexing(out)
    length(out) == p || throw(DimensionMismatch("output buffers must have one matrix per node"))
    all(x -> x isa AbstractMatrix, out) || throw(ArgumentError("node output buffers must be matrices"))
    for i in 1:p, j in 1:i-1
        Base.mightalias(out[i], out[j]) && throw(ArgumentError("node output buffers must not alias each other"))
    end
    nothing
end

# Dense column-contiguous storage uses the allocating multiplication's mul!
# path. Other layouts use a product workspace, retaining exact arithmetic.
_buffer_direct(out) = out isa StridedMatrix && stride(out, 1) == 1
function _buffer_product!(out, left, right)
    if _buffer_direct(out)
        mul!(out, left, right)
    else
        copyto!(out, left * right)
    end
    out
end

"""
    evaluate!(out, fd, points[, evaluator]; derivative=0)

Write the values of [`evaluate`](@ref) into `out` and return that same buffer.
For univariate data, `out` is an `nobs(fd) × length(points)` matrix. Raw grid
or irregular data require `PiecewiseLinear()` or `NoInterpolation()`; basis
data support the `derivative` keyword. Point order and repeated points are
preserved, including the allocating evaluator's missing/extrapolation rules.

For multivariate data, common points require an `nobs × ncomponents × npoints`
array. A vector of point vectors requires a vector of matrices in node order.
An evaluator is required whenever a component is raw data, and ignored for
basis components. These routes support value evaluation only.

Buffers must use one-based indexing and the exact floating element type of
the allocating result, including its point-type promotion. Shape/type errors
and conservative `Base.mightalias` checks reject overlap with input data,
points or model storage, and overlap between node buffers, before writing.
Evaluation errors can leave partial output. Inputs are never modified.
Internal point/basis and noncontiguous-product workspaces can still allocate;
this API reuses the caller's output storage.
"""
function evaluate!(out::AbstractMatrix, fd::Union{GridFunctionalData,IrregularFunctionalData},
        t::AbstractVector{<:Real}, evaluator::Union{PiecewiseLinear,NoInterpolation})
    points = _evaluation_points(fd, t)
    _check_buffer(out, (nobs(fd), length(points)), eltype(points), fd, t)
    for i in 1:nobs(fd)
        times, values = fd[i]
        _evaluate_curve!(view(out, i, :), times, values, points, evaluator)
    end
    out
end

evaluate!(::AbstractMatrix, ::Union{GridFunctionalData,IrregularFunctionalData}, ::AbstractVector{<:Real}) =
    throw(ArgumentError("raw data evaluation requires PiecewiseLinear() or NoInterpolation()"))

function evaluate!(out::AbstractMatrix, fd::BasisFunctionalData, t::AbstractVector{<:Real}; derivative=0)
    functions = basis_matrix(fd.basis, t; derivative)
    T = promote_type(eltype(fd.coefs), eltype(functions))
    _check_buffer(out, (nobs(fd), length(t)), T, fd, t, functions)
    _buffer_product!(out, fd.coefs, transpose(functions))
end

_component_evaluate!(out, fd::BasisFunctionalData, points, evaluator) = evaluate!(out, fd, points)
_component_evaluate!(out, fd::Union{GridFunctionalData,IrregularFunctionalData}, points, ::Nothing) =
    throw(ArgumentError("raw multivariate components require PiecewiseLinear() or NoInterpolation()"))
_component_evaluate!(out, fd, points, evaluator) = evaluate!(out, fd, points, evaluator)

function evaluate!(out::AbstractArray{T,3}, fd::MultivariateFunctionalData,
        t::AbstractVector{<:Real}, evaluator::Union{Nothing,PiecewiseLinear,NoInterpolation}=nothing) where {T}
    points = _evaluation_points(fd, t)
    _check_buffer(out, (nobs(fd), ncomponents(fd), length(points)), eltype(points), fd, t)
    for c in fd.components
        _component_domain(c, points)
    end
    for j in 1:ncomponents(fd)
        _component_evaluate!(view(out, :, j, :), component(fd, j), points, evaluator)
    end
    out
end

function evaluate!(out::AbstractVector, fd::MultivariateFunctionalData{T},
        ts::AbstractVector{<:AbstractVector{S}},
        evaluator::Union{Nothing,PiecewiseLinear,NoInterpolation}=nothing) where {T,S<:Real}
    Base.require_one_based_indexing(ts)
    p = ncomponents(fd)
    length(ts) == p || throw(DimensionMismatch("point vectors must match the number of components"))
    _check_node_buffers(out, p)
    R = promote_type(T, _floattype(S))
    for j in 1:p
        _component_domain(component(fd, j), ts[j])
        _check_buffer(out[j], (nobs(fd), length(ts[j])), R, fd, ts)
    end
    for j in 1:p
        _component_evaluate!(out[j], component(fd, j), ts[j], evaluator)
    end
    out
end

function _buffer_reconstruction_parts(projected, basis, mean, grid)
    _validate_strict_grid(grid)
    functions = basis_matrix(basis, grid)
    mu = _fpca_mean_values(mean, grid)
    (; projected, functions, mu)
end
_buffer_reconstruction_type(parts) =
    promote_type(eltype(parts.projected), eltype(parts.functions), eltype(parts.mu))

function _write_reconstruction!(out, parts)
    product_type = promote_type(eltype(parts.projected), eltype(parts.functions))
    if _buffer_direct(out) && eltype(out) === product_type === _buffer_reconstruction_type(parts)
        mul!(out, parts.projected, transpose(parts.functions))
        out .+= parts.mu
    else
        copyto!(out, parts.projected * transpose(parts.functions) .+ parts.mu)
    end
    all(isfinite, out) || throw(ArgumentError("reconstructed values must be finite; rescale scores"))
    out
end

"""
    reconstruct!(out, model, projected=scores(model); grid=model.grid)
    reconstruct!(node_buffers, model, projected=scores(model); grid=nothing)
    reconstruct!(node_buffers, d::MultivariateFPCADerivative; grid=nothing)

Write the curve values of [`reconstruct`](@ref) into caller-owned buffers and
return those buffers. This extends `StatsAPI.reconstruct!`, the in-place
counterpart of the `StatsAPI.reconstruct` generic used by this package.

FPCA and FPCADerivative use a `size(projected,1) × length(grid)` matrix.
MultivariateFPCA, MFPCA and MultivariateFPCADerivative use a vector of matrices,
one per node, with the same grids/score conventions as their allocating
methods. Values, arithmetic promotion and evaluation rules are identical.
Buffers contain only numeric values; IDs and derivative metadata remain on
the allocating functional-data results and are not attached to the buffers.

Each buffer must use one-based indexing and exactly the floating element
type of its allocating result. Shape/type errors and conservative
`Base.mightalias` checks reject overlap with scores, grids, model storage or
other node buffers before writing. Computation errors can leave partial
output. Inputs are never modified. Dense column-contiguous products write
directly into the buffer; basis/mean preparation and other layouts can still
allocate workspaces. The allocating methods keep their existing ownership.
"""
function reconstruct!(out::AbstractMatrix, f::Union{FPCA,FPCADerivative},
        projected::AbstractMatrix{<:Real}=scores(f); grid::AbstractVector{<:Real}=f.grid)
    Base.require_one_based_indexing(projected, grid)
    _validate_reconstruction_scores(projected, ncomponents(f))
    parts = _buffer_reconstruction_parts(projected, f.basis, f.mean, grid)
    _check_buffer(out, (size(projected,1), length(grid)), _buffer_reconstruction_type(parts), f, projected, grid)
    _write_reconstruction!(out, parts)
end

function _write_node_reconstructions!(out, parts, ::Type{T}, sources...) where {T}
    _check_node_buffers(out, length(parts))
    for j in eachindex(parts)
        item = parts[j]
        _check_buffer(out[j], (size(item.projected,1), size(item.functions,1)), T, sources...)
    end
    for j in eachindex(parts)
        _write_reconstruction!(out[j], parts[j])
    end
    out
end

function reconstruct!(out::AbstractVector, f::MultivariateFPCA{T},
        projected::AbstractVector=scores(f); grid=nothing) where {T}
    Base.require_one_based_indexing(projected)
    p = length(f.scores)
    length(projected) == p || throw(DimensionMismatch("reconstruction scores must have one matrix per node"))
    all(s -> s isa AbstractMatrix{<:Real}, projected) || throw(ArgumentError("reconstruction scores must be real matrices"))
    n = size(first(projected),1)
    all(s -> size(s,1) == n, projected) || throw(DimensionMismatch("reconstruction score matrices must have the same number of rows"))
    grids = _multivariate_grids(f, grid)
    R = promote_type(_multivariate_score_type(T, projected), _multivariate_grid_type(T, grid))
    parts = map(1:p) do j
        Base.require_one_based_indexing(projected[j])
        _validate_reconstruction_scores(projected[j], size(f.scores[j],2))
        _buffer_reconstruction_parts(projected[j], f.bases[j], f.means[j], grids[j])
    end
    _write_node_reconstructions!(out, parts, R, f, projected, grids)
end

function reconstruct!(out::AbstractVector, f::MFPCA{T}, projected::AbstractMatrix{S}=scores(f);
        grid=nothing) where {T,S<:Real}
    Base.require_one_based_indexing(projected)
    _validate_reconstruction_scores(projected, ncomponents(f))
    grids = _multivariate_grids(f.univariate, grid)
    R = promote_type(T, _floattype(S), _multivariate_grid_type(T, grid))
    parts = [_buffer_reconstruction_parts(projected, f.eigenfunctions[j],
        f.univariate.means[j], grids[j]) for j in eachindex(f.eigenfunctions)]
    _write_node_reconstructions!(out, parts, R, f, projected, grids)
end

function reconstruct!(out::AbstractVector, d::MultivariateFPCADerivative; grid=nothing)
    grids = _multivariate_grids(d, grid)
    parts = map(d.nodes, grids) do node, points
        _validate_reconstruction_scores(scores(node), ncomponents(node))
        _buffer_reconstruction_parts(scores(node), node.basis, node.mean, points)
    end
    R = mapreduce(_buffer_reconstruction_type, promote_type, parts)
    _write_node_reconstructions!(out, parts, R, d, grids)
end
