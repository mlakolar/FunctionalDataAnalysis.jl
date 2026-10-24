"""
    GridFunctionalData(values, grid)

Curves in the rows of a copied `n × m` matrix on a common, finite, strictly
increasing grid with at least two points. At least one curve is required.
Missing entries are represented by an `observed::Union{Nothing,BitMatrix}`
mask (true means observed); their stored numerical values are zero.
Every nonmissing value must be finite. Integer input becomes `Float64`.
"""
struct GridFunctionalData{T<:AbstractFloat} <: AbstractFunctionalData{T}
    values::Matrix{T}
    grid::Vector{T}
    weights::Vector{T}
    observed::Union{Nothing,BitMatrix}
    function GridFunctionalData{T}(::Val{:validated}, values, grid, weights, observed) where {T<:AbstractFloat}
        new{T}(values, grid, weights, observed)
    end
end

function GridFunctionalData(values::AbstractMatrix{V}, grid::AbstractVector{G}) where
        {V<:Union{Missing,Real},G<:Real}
    Base.require_one_based_indexing(values, grid)
    size(values, 1) >= 1 || throw(ArgumentError("at least one curve is required"))
    size(values, 2) == length(grid) ||
        throw(DimensionMismatch("values columns must match the grid length"))
    T = promote_type(_floattype(Base.nonmissingtype(V)), _floattype(G))
    x = Vector{T}(grid)
    w = quadrature_weights(x)
    y = zeros(T, size(values))
    observed = trues(size(values))
    for k in eachindex(values)
        if ismissing(values[k])
            observed[k] = false
        else
            value = T(values[k])
            isfinite(value) || throw(ArgumentError("nonmissing curve values must be finite"))
            y[k] = value
        end
    end
    GridFunctionalData{T}(Val(:validated), y, x, w, all(observed) ? nothing : observed)
end

"""
    IrregularFunctionalData(times, values; domain=nothing, ties=:error)
    IrregularFunctionalData(times, values, offsets; domain=nothing, ties=:error)

Copied observations on curve-specific grids. The first form takes vectors
of vectors; the second takes concatenated arrays with curve `i` at
`offsets[i]:offsets[i+1]-1`, starting at 1 and ending at `length(times)+1`.
Every curve has at least one finite observation. Times are sorted within
curves. Repeated times throw unless `ties=:mean` averages their values.

The domain defaults to the range of all times. A supplied finite interval
must contain every time; a singleton domain is allowed for point observations.
"""
struct IrregularFunctionalData{T<:AbstractFloat} <: AbstractFunctionalData{T}
    times::Vector{T}
    values::Vector{T}
    offsets::Vector{Int}
    domain::Tuple{T,T}
    function IrregularFunctionalData{T}(::Val{:validated}, times, values, offsets, domain) where {T<:AbstractFloat}
        new{T}(times, values, offsets, domain)
    end
end

_domaintype(::Nothing, ::Type{T}) where {T} = T
_domaintype(d::Tuple{<:Real,<:Real}, ::Type{T}) where {T} =
    promote_type(T, _floattype(promote_type(typeof(d[1]), typeof(d[2]))))

function _data_domain(bounds, times, ::Type{T}) where {T}
    a, b = bounds === nothing ? extrema(times) : (T(bounds[1]), T(bounds[2]))
    isfinite(a) && isfinite(b) && a <= b ||
        throw(ArgumentError("domain endpoints must be finite and nondecreasing"))
    all(t -> a <= t <= b, times) ||
        throw(ArgumentError("domain must contain all observation times"))
    (T(a), T(b))
end

function IrregularFunctionalData(times::AbstractVector{S}, values::AbstractVector{V},
        offsets::AbstractVector{<:Integer}; domain=nothing, ties=:error) where {S<:Real,V<:Real}
    Base.require_one_based_indexing(times, values, offsets)
    length(times) == length(values) ||
        throw(DimensionMismatch("observation times and values must have equal length"))
    length(offsets) >= 2 || throw(ArgumentError("offsets must describe at least one curve"))
    first(offsets) == 1 && last(offsets) == length(times) + 1 ||
        throw(ArgumentError("offsets must start at 1 and end at observation count plus 1"))
    all(i -> 1 <= offsets[i] < offsets[i + 1] <= length(times) + 1,
        1:length(offsets)-1) ||
        throw(ArgumentError("offsets must increase and every curve needs at least one observation"))
    ties in (:error, :mean) || throw(ArgumentError("ties must be :error or :mean"))
    T = _domaintype(domain, promote_type(_floattype(S), _floattype(V)))
    t, y = Vector{T}(times), Vector{T}(values)
    all(isfinite, t) && all(isfinite, y) ||
        throw(ArgumentError("observation times and values must be finite"))
    bounds = _data_domain(domain, t, T)
    outtimes, outvalues, outoffsets = T[], T[], Int[1]
    for i in 1:length(offsets)-1
        index = Int(offsets[i]):Int(offsets[i + 1])-1
        permutation = sortperm(t[index])
        ti, yi = t[index][permutation], y[index][permutation]
        k = 1
        while k <= length(ti)
            lasttie = k
            while lasttie < length(ti) && ti[lasttie + 1] == ti[k]
                lasttie += 1
            end
            lasttie > k && ties == :error &&
                throw(ArgumentError("observation times must be distinct within each curve"))
            value = lasttie == k ? yi[k] : sum(@view yi[k:lasttie]) / (lasttie - k + 1)
            isfinite(value) || throw(ArgumentError("mean of tied values must be finite"))
            push!(outtimes, ti[k])
            push!(outvalues, value)
            k = lasttie + 1
        end
        push!(outoffsets, length(outtimes) + 1)
    end
    IrregularFunctionalData{T}(Val(:validated), outtimes, outvalues, outoffsets, bounds)
end

function IrregularFunctionalData(times::AbstractVector{<:AbstractVector{S}},
        values::AbstractVector{<:AbstractVector{V}}; domain=nothing, ties=:error) where
        {S<:Real,V<:Real}
    Base.require_one_based_indexing(times, values)
    length(times) == length(values) ||
        throw(DimensionMismatch("times and values must describe the same number of curves"))
    isempty(times) && throw(ArgumentError("at least one curve is required"))
    t, y, offsets = S[], V[], Int[1]
    for i in eachindex(times)
        length(times[i]) == length(values[i]) ||
            throw(DimensionMismatch("observation times and values differ for curve $i"))
        isempty(times[i]) && throw(ArgumentError("every curve needs at least one observation"))
        append!(t, times[i])
        append!(y, values[i])
        push!(offsets, length(t) + 1)
    end
    IrregularFunctionalData(t, y, offsets; domain, ties)
end

"""
    BasisFunctionalData(coefs, basis)

Exact functional representations with one coefficient row per curve and
`nbasis(basis)` columns. Coefficients must be finite and at least one curve
is required. Coefficients and the basis are copied. Scalar indexing and
iteration retain exact one-curve basis objects; they never choose a grid.
The object is callable: `fd(t) == evaluate(fd, t)`.
"""
struct BasisFunctionalData{T<:AbstractFloat,B<:AbstractBasis} <: AbstractFunctionalData{T}
    coefs::Matrix{T}
    basis::B
    function BasisFunctionalData(coefs::AbstractMatrix{S}, basis::B) where {S<:Real,B<:AbstractBasis}
        Base.require_one_based_indexing(coefs)
        size(coefs, 1) >= 1 || throw(ArgumentError("at least one curve is required"))
        size(coefs, 2) == nbasis(basis) ||
            throw(DimensionMismatch("coefficient columns must match the number of basis functions"))
        T = promote_type(_floattype(S), _basis_eltype(basis))
        c = Matrix{T}(coefs)
        all(isfinite, c) || throw(ArgumentError("basis coefficients must be finite"))
        new{T,B}(c, deepcopy(basis))
    end
end

_basis_eltype(b::AbstractBasis) =
    _floattype(promote_type(typeof(domain(b)[1]), typeof(domain(b)[2])))
