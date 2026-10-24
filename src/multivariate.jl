"""
    MultivariateFunctionalData(components)

Copied univariate components (grid, irregular, or basis data) for the same
subjects. Components may have different designs and domains. Indexing
selects subjects; `component(mfd, j)` selects node `j`. Scalar indexing
returns each component's own curve representation, including exact basis
objects. `domain`, `sampling_design`, and `nobs_per_curve` return one result
per component, without aggregating different designs.
"""
struct MultivariateFunctionalData{T<:AbstractFloat} <: AbstractFunctionalData{T}
    components::Vector{AbstractFunctionalData{T}}
    function MultivariateFunctionalData{T}(components) where {T<:AbstractFloat}
        isempty(components) && throw(ArgumentError("at least one component is required"))
        all(c -> c isa Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData}, components) ||
            throw(ArgumentError("components must be univariate grid, irregular or basis data"))
        all(c -> nobs(c) == nobs(first(components)), components) ||
            throw(DimensionMismatch("all components must have the same number of subjects"))
        _validate_component_ids(components)
        copied = AbstractFunctionalData{T}[_convert_component(T, c) for c in components]
        new{T}(copied)
    end
end

MultivariateFunctionalData(c::AbstractVector{<:AbstractFunctionalData{T}}) where {T} =
    MultivariateFunctionalData{T}(c)
function MultivariateFunctionalData(c::AbstractVector{<:AbstractFunctionalData})
    isempty(c) && throw(ArgumentError("at least one component is required"))
    T = mapreduce(eltype, promote_type, c)
    MultivariateFunctionalData{T}(c)
end

function _convert_component(::Type{T}, fd::GridFunctionalData) where {T}
    GridFunctionalData(Matrix{Union{Missing,T}}(_grid_matrix(fd)), Vector{T}(fd.grid);
        ids=fd.ids, metadata=fd.metadata)
end
_convert_component(::Type{T}, fd::IrregularFunctionalData) where {T} =
    IrregularFunctionalData(Vector{T}(fd.times), Vector{T}(fd.values), fd.offsets;
        domain=(T(domain(fd)[1]), T(domain(fd)[2])), ids=fd.ids)
_convert_component(::Type{T}, fd::BasisFunctionalData) where {T} =
    BasisFunctionalData(Matrix{T}(fd.coefs), fd.basis; ids=fd.ids, metadata=fd.metadata)

"""
    ncomponents(mfd)

Number of univariate components (nodes) in multivariate functional data.
"""
ncomponents(fd::MultivariateFunctionalData) = length(fd.components)

"""
    component(mfd, j)

Return component (node) `j`. Subject indexing never selects nodes. The
returned component is the stored object; use `deepcopy` for an independent
copy before mutation.
"""
component(fd::MultivariateFunctionalData, j::Integer) = fd.components[j]

nobs(fd::MultivariateFunctionalData) = nobs(first(fd.components))::Int
domain(fd::MultivariateFunctionalData{T}) where {T} =
    Tuple{T,T}[domain(c) for c in fd.components]
nobs_per_curve(fd::MultivariateFunctionalData) =
    Vector{Int}[nobs_per_curve(c) for c in fd.components]
sampling_design(fd::MultivariateFunctionalData; threshold=20) =
    Symbol[sampling_design(c; threshold) for c in fd.components]
Base.getindex(fd::MultivariateFunctionalData, i::Integer) =
    Any[c[_subject_index(fd, i)] for c in fd.components]
_subset(fd::MultivariateFunctionalData{T}, idx) where {T} =
    MultivariateFunctionalData{T}(AbstractFunctionalData{T}[c[idx] for c in fd.components])

function _component_evaluate(fd::BasisFunctionalData, t, evaluator)
    evaluate(fd, t)
end
_component_evaluate(fd::Union{GridFunctionalData,IrregularFunctionalData}, t, ::Nothing) =
    throw(ArgumentError("raw multivariate components require PiecewiseLinear() or NoInterpolation()"))
_component_evaluate(fd::Union{GridFunctionalData,IrregularFunctionalData}, t, evaluator) =
    evaluate(fd, t, evaluator)

function _component_domain(c, t)
    a, b = domain(c)
    all(x -> isfinite(x) && a <= x <= b, t) ||
        throw(ArgumentError("multivariate evaluation points must lie inside each component domain"))
end

function _evaluate_multivariate(fd::MultivariateFunctionalData, t::AbstractVector{<:Real}, evaluator)
    points = _evaluation_points(fd, t)
    out = Array{eltype(points)}(undef, nobs(fd), ncomponents(fd), length(points))
    for j in 1:ncomponents(fd)
        c = component(fd, j)
        _component_domain(c, points)
        out[:, j, :] = _component_evaluate(c, points, evaluator)
    end
    out
end
evaluate(fd::MultivariateFunctionalData, t::AbstractVector{<:Real}) =
    _evaluate_multivariate(fd, t, nothing)
evaluate(fd::MultivariateFunctionalData, t::AbstractVector{<:Real}, e::Union{PiecewiseLinear,NoInterpolation}) =
    _evaluate_multivariate(fd, t, e)

function _evaluate_multivariate(fd::MultivariateFunctionalData{T},
        ts::AbstractVector{<:AbstractVector{S}}, evaluator) where {T,S<:Real}
    length(ts) == ncomponents(fd) ||
        throw(DimensionMismatch("point vectors must match the number of components"))
    R = promote_type(T, _floattype(S))
    out = Vector{Matrix{R}}(undef, ncomponents(fd))
    for j in 1:ncomponents(fd)
        c = component(fd, j)
        _component_domain(c, ts[j])
        out[j] = _component_evaluate(c, ts[j], evaluator)
    end
    out
end
evaluate(fd::MultivariateFunctionalData, ts::AbstractVector{<:AbstractVector{<:Real}}) =
    _evaluate_multivariate(fd, ts, nothing)
evaluate(fd::MultivariateFunctionalData, ts::AbstractVector{<:AbstractVector{<:Real}},
        e::Union{PiecewiseLinear,NoInterpolation}) = _evaluate_multivariate(fd, ts, e)

"""
    to_array(mfd)

Return a copied `n × p × m` array in FunctionalGraphicalModels.jl order:
subjects, nodes, grid points. Every component must be complete grid data
on the same grid; other representations or missing observations throw.
"""
function to_array(fd::MultivariateFunctionalData{T}) where {T}
    all(c -> c isa GridFunctionalData && c.observed === nothing, fd.components) ||
        throw(ArgumentError("to_array requires grid components without missing observations"))
    grid = (first(fd.components)::GridFunctionalData).grid
    all(c -> c.grid == grid, fd.components) ||
        throw(ArgumentError("to_array requires the same grid for every component"))
    out = Array{T}(undef, nobs(fd), ncomponents(fd), length(grid))
    for j in 1:ncomponents(fd)
        out[:, j, :] = (component(fd, j)::GridFunctionalData).values
    end
    out
end

function Base.vcat(a::MultivariateFunctionalData, b::MultivariateFunctionalData)
    ncomponents(a) == ncomponents(b) ||
        throw(DimensionMismatch("multivariate vcat requires the same number of components"))
    for j in 1:ncomponents(a)
        left, right = component(a, j), component(b, j)
        ((left isa GridFunctionalData && right isa GridFunctionalData) ||
            (left isa IrregularFunctionalData && right isa IrregularFunctionalData) ||
            (left isa BasisFunctionalData && right isa BasisFunctionalData)) ||
            throw(ArgumentError("multivariate vcat requires matching representations at component $j"))
    end
    T = promote_type(eltype(a), eltype(b))
    MultivariateFunctionalData{T}(AbstractFunctionalData{T}[
        _mfd_concat_component(a, b, j) for j in 1:ncomponents(a)])
end
function Base.show(io::IO, fd::MultivariateFunctionalData{T}) where {T}
    print(io, "MultivariateFunctionalData{", T, "}(n=", nobs(fd),
        ", components=", ncomponents(fd), ")")
end
