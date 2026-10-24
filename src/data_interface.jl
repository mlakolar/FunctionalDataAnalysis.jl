Base.eltype(::Type{<:AbstractFunctionalData{T}}) where {T} = T
Base.eltype(fd::AbstractFunctionalData) = eltype(typeof(fd))
Base.IteratorEltype(::Type{<:AbstractFunctionalData}) = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:AbstractFunctionalData}) = Base.HasLength()
nobs(fd::GridFunctionalData) = size(fd.values, 1)
nobs(fd::IrregularFunctionalData) = length(fd.offsets) - 1
nobs(fd::BasisFunctionalData) = size(fd.coefs, 1)
Base.length(fd::AbstractFunctionalData) = nobs(fd)
domain(fd::GridFunctionalData) = (first(fd.grid), last(fd.grid))
domain(fd::IrregularFunctionalData) = fd.domain
domain(fd::BasisFunctionalData) = domain(fd.basis)

function _subject_index(fd, i::Integer)
    1 <= i <= nobs(fd) || throw(BoundsError(fd, i))
    Int(i)
end

function _subject_indices(fd, idx::AbstractVector{Bool})
    length(idx) == nobs(fd) || throw(DimensionMismatch("subject mask length must equal nobs"))
    index = findall(idx)
    isempty(index) && throw(ArgumentError("subject subset must contain at least one curve"))
    index
end

function _subject_indices(fd, idx::AbstractVector{<:Integer})
    isempty(idx) && throw(ArgumentError("subject subset must contain at least one curve"))
    [_subject_index(fd, i) for i in idx]
end

"""
    fd[i]
    fd[idx]

For grid and irregular data, scalar indexing returns copied `(t, y)` vectors
of actual observations, omitting only masked grid entries. For basis data,
it returns a one-curve `BasisFunctionalData`, preserving the exact basis
representation without selecting a grid. Multivariate scalar indexing
returns a vector of these component-specific results.

A range, integer vector, or Boolean subject mask returns the same data type,
with copied arrays. Empty subsets throw. Indexing always selects subjects;
use `component(mfd, j)` to select a multivariate node.
"""
function Base.getindex(fd::GridFunctionalData, i::Integer)
    i = _subject_index(fd, i)
    selected = fd.observed === nothing ? trues(length(fd.grid)) : fd.observed[i, :]
    (fd.grid[selected], fd.values[i, selected])
end

function Base.getindex(fd::IrregularFunctionalData, i::Integer)
    i = _subject_index(fd, i)
    index = fd.offsets[i]:fd.offsets[i + 1]-1
    (fd.times[index], fd.values[index])
end

Base.getindex(fd::BasisFunctionalData, i::Integer) =
    BasisFunctionalData(fd.coefs[_subject_index(fd, i):i, :], fd.basis;
        ids=fd.ids === nothing ? nothing : fd.ids[i:i])

Base.getindex(fd::AbstractFunctionalData, idx::AbstractVector{<:Integer}) =
    _subset(fd, _subject_indices(fd, idx))

function _grid_matrix(fd::GridFunctionalData{T}) where {T}
    out = Matrix{Union{Missing,T}}(fd.values)
    if fd.observed !== nothing
        out[.!fd.observed] .= missing
    end
    out
end

_subset(fd::GridFunctionalData, idx) = GridFunctionalData(_grid_matrix(fd)[idx, :], fd.grid;
    ids=fd.ids === nothing ? nothing : fd.ids[idx])
_subset(fd::BasisFunctionalData, idx) = BasisFunctionalData(fd.coefs[idx, :], fd.basis;
    ids=fd.ids === nothing ? nothing : fd.ids[idx])
_subset(fd::IrregularFunctionalData{T}, idx) where {T} =
    IrregularFunctionalData(Vector{T}[fd[i][1] for i in idx],
        Vector{T}[fd[i][2] for i in idx]; domain=domain(fd),
        ids=fd.ids === nothing ? nothing : fd.ids[idx])

"""
    iterate(fd[, state])

Iterate over subjects. Grid and irregular data yield copied observation
pairs. Basis data yield exact one-curve basis objects. Multivariate data
yield a vector containing each component's subject result. No grid or
interpolation is chosen for basis data.
"""
function Base.iterate(fd::AbstractFunctionalData, state::Int=1)
    state > nobs(fd) && return nothing
    (fd[state], state + 1)
end

"""
    nobs_per_curve(fd)

Return a `Vector{Int}` of actual observation counts for raw univariate data.
Basis data return the coefficient count per curve, since they have no
observation grid. Multivariate data return a `Vector{Vector{Int}}`, one
count vector per component, without aggregating different designs.
"""
function nobs_per_curve(fd::GridFunctionalData)
    fd.observed === nothing && return fill(length(fd.grid), nobs(fd))
    [count(@view fd.observed[i, :]) for i in 1:nobs(fd)]
end
nobs_per_curve(fd::IrregularFunctionalData) = diff(fd.offsets)
nobs_per_curve(fd::BasisFunctionalData) = fill(nbasis(fd.basis), nobs(fd))

function _threshold(threshold)
    threshold isa Real && isfinite(threshold) && threshold > 0 ||
        throw(ArgumentError("sampling threshold must be finite and positive"))
    threshold
end

function _count_design(counts, threshold)
    _threshold(threshold)
    x = sort(counts)
    n = length(x)
    median = isodd(n) ? x[div(n + 1, 2)] :
        (x[div(n, 2)] + x[div(n, 2) + 1]) / 2
    median >= threshold ? :dense_irregular : :sparse
end

function sampling_design(fd::GridFunctionalData; threshold=20)
    _threshold(threshold)
    fd.observed === nothing ? :regular : _count_design(nobs_per_curve(fd), threshold)
end
sampling_design(fd::IrregularFunctionalData; threshold=20) =
    _count_design(nobs_per_curve(fd), threshold)
function sampling_design(fd::BasisFunctionalData; threshold=20)
    _threshold(threshold)
    :regular
end

function Base.show(io::IO, fd::GridFunctionalData{T}) where {T}
    print(io, "GridFunctionalData{", T, "}(n=", nobs(fd), ", grid=",
        length(fd.grid), " points, missing=", fd.observed === nothing ? 0 : count(!, fd.observed), ")")
end
function Base.show(io::IO, fd::IrregularFunctionalData{T}) where {T}
    counts = nobs_per_curve(fd)
    print(io, "IrregularFunctionalData{", T, "}(n=", nobs(fd), ", points per curve=",
        minimum(counts), ":", maximum(counts), ")")
end
function Base.show(io::IO, fd::BasisFunctionalData{T}) where {T}
    print(io, "BasisFunctionalData{", T, "}(n=", nobs(fd), ", nbasis=", nbasis(fd.basis), ")")
end
Base.show(io::IO, ::MIME"text/plain", fd::AbstractFunctionalData) = show(io, fd)
