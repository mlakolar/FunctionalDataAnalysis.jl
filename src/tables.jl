function _validate_ids(labels, n)
    labels === nothing && return nothing
    length(labels) == n || throw(DimensionMismatch("ids must contain one label per curve"))
    any(ismissing, labels) && throw(ArgumentError("subject ids must not be missing"))
    length(unique(labels)) == n || throw(ArgumentError("subject ids must be unique"))
    deepcopy(Any[labels...])
end

"""
    ids(fd)

Return copied subject IDs in curve order. Table
constructors preserve the order of first appearance. Return `nothing` when
IDs are absent. Optional `ids` constructor metadata must be unique, nonmissing,
and have one label per curve. Labels are stored without restricting their
type. Exact duplicate subject subsets of labeled data therefore throw.
Multivariate data return the IDs shared by labeled components, or `nothing`
if no component is labeled. Labeled components must agree exactly in order.
"""
ids(fd::Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData}) =
    fd.ids === nothing ? nothing : deepcopy(fd.ids)

_stored_ids(fd::Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData}) = fd.ids

function _validate_component_ids(components)
    reference = nothing
    for c in components
        labels = _stored_ids(c)
        if labels !== nothing
            reference === nothing || isequal(reference, labels) ||
                throw(ArgumentError("labeled multivariate components must have identical ids in subject order"))
            reference = labels
        end
    end
    nothing
end

function ids(fd::MultivariateFunctionalData)
    for c in fd.components
        c.ids === nothing || return deepcopy(c.ids)::Vector{Any}
    end
    nothing
end

function _table_ids(fd)
    labels = ids(fd)
    labels === nothing ? Any[i for i in 1:nobs(fd)] : labels
end

function _concatenated_ids(a, b)
    a.ids === nothing && b.ids === nothing && return nothing
    a.ids !== nothing && b.ids !== nothing ||
        throw(ArgumentError("vcat requires both inputs to have ids or both to be unlabeled"))
    vcat(a.ids, b.ids)
end

function _column(columns, name::Symbol)
    haskey(columns, name) || throw(ArgumentError("table is missing column $name"))
    getproperty(columns, name)
end

function _table_columns(table, id, time, value)
    Tables.istable(table) || throw(ArgumentError("input must support the Tables.jl interface"))
    columns = Tables.columntable(table)
    labels, times, values = _column(columns, id), _column(columns, time), _column(columns, value)
    length(labels) == length(times) == length(values) ||
        throw(DimensionMismatch("id, time and value columns must have equal lengths"))
    isempty(labels) && throw(ArgumentError("table must contain at least one observation"))
    any(ismissing, labels) && throw(ArgumentError("subject ids must not be missing"))
    times isa AbstractVector{<:Union{Missing,Real}} ||
        throw(ArgumentError("table times must be finite nonmissing real values"))
    any(ismissing, times) && throw(ArgumentError("table times must not be missing"))
    values isa AbstractVector{<:Union{Missing,Real}} ||
        throw(ArgumentError("table values must be real or missing"))
    all(isfinite, times) || throw(ArgumentError("table times must be finite"))
    labels, Vector{Base.nonmissingtype(eltype(times))}(times), values
end

function _table_groups(labels)
    lookup = Dict{Any,Int}()
    ids = Any[]
    rows = Vector{Int}[]
    for (row, id) in enumerate(labels)
        group = get(lookup, id, 0)
        if group == 0
            push!(ids, id)
            push!(rows, Int[])
            group = length(ids)
            lookup[id] = group
        end
        push!(rows[group], row)
    end
    ids, rows
end

"""
    IrregularFunctionalData(table; id, time, value, domain=nothing, ties=:error)

Read a Tables.jl source using the named columns. Curve order is the order of
first appearance of each ID; each curve is sorted by time. Missing values
or times throw rather than being dropped. Duplicate times follow `ties`.
Subject labels are copied and available through `ids`.
"""
function IrregularFunctionalData(table; id::Symbol, time::Symbol, value::Symbol,
        domain=nothing, ties=:error)
    labels, times, values = _table_columns(table, id, time, value)
    _irregular_from_columns(labels, times, values; domain, ties)
end

function _irregular_from_columns(labels, times::AbstractVector{S}, values::AbstractVector{V};
        domain=nothing, ties=:error) where {S<:Real,V<:Union{Missing,Real}}
    any(ismissing, values) && throw(ArgumentError("irregular table values must not be missing"))
    labels, rows = _table_groups(labels)
    T = promote_type(_floattype(S), _floattype(Base.nonmissingtype(V)))
    t, y = Vector{T}(times), Vector{T}(values)
    IrregularFunctionalData(Vector{T}[t[r] for r in rows], Vector{T}[y[r] for r in rows];
        domain, ties, ids=labels)
end

"""
    GridFunctionalData(table; id, time, value)

Read a Tables.jl source onto the sorted union of its time points as one
common grid. Absent `(id, time)` rows and explicit missing values become
masked observations. Every `(id, time)` pair must be unique. Curve order is
first appearance of each ID. At least two distinct grid points are required.
Signed zero times are normalized to positive zero before grouping. No
interpolation is performed. Subject labels are copied and available through
`ids`.
"""
function GridFunctionalData(table; id::Symbol, time::Symbol, value::Symbol)
    labels, times, values = _table_columns(table, id, time, value)
    _grid_from_columns(labels, times, values)
end

function _grid_from_columns(labels, times::AbstractVector{S}, values::AbstractVector{V}) where
        {S<:Real,V<:Union{Missing,Real}}
    labels, rows = _table_groups(labels)
    T = promote_type(_floattype(S), _floattype(Base.nonmissingtype(V)))
    t = Vector{T}(times) .+ zero(T)
    grid = sort!(unique(t))
    quadrature_weights(grid)
    out = Matrix{Union{Missing,T}}(missing, length(labels), length(grid))
    seen = falses(size(out))
    for i in eachindex(rows), row in rows[i]
        j = searchsortedfirst(grid, t[row])
        seen[i, j] && throw(ArgumentError("table must have unique (id, time) pairs"))
        seen[i, j] = true
        out[i, j] = ismissing(values[row]) ? missing : T(values[row])
    end
    GridFunctionalData(out, grid; ids=labels)
end

_with_ids(fd::GridFunctionalData, labels) =
    GridFunctionalData(_grid_matrix(fd), fd.grid; ids=labels, metadata=fd.metadata)
_with_ids(fd::IrregularFunctionalData, labels) =
    IrregularFunctionalData(fd.times, fd.values, fd.offsets; domain=domain(fd), ids=labels)
_with_ids(fd::BasisFunctionalData, labels) =
    BasisFunctionalData(fd.coefs, fd.basis; ids=labels, metadata=fd.metadata)

function _mfd_concat_component(a, b, j)
    left, right = ids(a), ids(b)
    (left === nothing) == (right === nothing) ||
        throw(ArgumentError("multivariate vcat requires both inputs to have ids or both to be unlabeled"))
    if left === nothing
        vcat(component(a, j), component(b, j))
    else
        vcat(_with_ids(component(a, j), left), _with_ids(component(b, j), right))
    end
end

Tables.istable(::Type{<:GridFunctionalData}) = true
Tables.istable(::Type{<:IrregularFunctionalData}) = true
Tables.columnaccess(::Type{<:GridFunctionalData}) = true
Tables.columnaccess(::Type{<:IrregularFunctionalData}) = true

function Tables.columns(fd::GridFunctionalData{T}) where {T}
    labels = _table_ids(fd)
    (id=Any[labels[i] for i in 1:nobs(fd) for _ in fd.grid],
        time=repeat(fd.grid, nobs(fd)), value=vec(permutedims(_grid_matrix(fd))))
end
function Tables.columns(fd::IrregularFunctionalData)
    labels = _table_ids(fd)
    (id=Any[labels[i] for i in 1:nobs(fd) for _ in fd.offsets[i]:fd.offsets[i + 1]-1],
        time=copy(fd.times), value=copy(fd.values))
end
Tables.schema(fd::GridFunctionalData{T}) where {T} =
    Tables.Schema((:id, :time, :value), (Any, T, Union{Missing,T}))
Tables.schema(fd::IrregularFunctionalData{T}) where {T} =
    Tables.Schema((:id, :time, :value), (Any, T, T))

"""
    Tables.columntable(fd::BasisFunctionalData; grid)

Sample basis data exactly on an explicit grid and return long-format
`(id, time, value)` columns, preserving subject IDs. No default grid is
chosen. For unlabeled data, table output uses positional IDs `1:nobs(fd)`
without altering the stored metadata.
"""
function Tables.columntable(fd::BasisFunctionalData; grid=nothing)
    grid === nothing && throw(ArgumentError("basis table output requires an explicit grid"))
    Tables.columntable(to_grid(fd, grid))
end
