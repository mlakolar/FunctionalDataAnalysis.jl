"""
    FPCBasis(grid, values; weights=quadrature_weights(grid))

Principal component functions stored in the columns of a `length(grid) × K`
matrix. The grid, function values and quadrature weights are copied. The grid
must be finite and strictly increasing, with at least two points. At least
one function is required; values must be finite and weights must be finite,
positive and have the same length as the grid.

[`basis_matrix`](@ref) evaluates the piecewise-linear interpolants inside the
closed grid range and throws outside it. Only derivative zero is supported;
[`penalty`](@ref) likewise supports only order zero. [`gram`](@ref) gives
exact integrals of products of the piecewise-linear interpolants.

Functions produced by `fit(FPCA, ...)` are orthonormal under the stored
work-grid trapezoidal weights. Their exact interpolant Gram matrix is close
to, but generally not exactly, the identity on a sufficiently fine grid.
The constructor does not normalize arbitrary supplied function values.
"""
struct FPCBasis{T<:AbstractFloat} <: AbstractBasis
    grid::Vector{T}
    weights::Vector{T}
    values::Matrix{T}
    function FPCBasis{T}(::Val{:validated}, grid, weights, values) where {T<:AbstractFloat}
        new{T}(grid, weights, values)
    end
end

function FPCBasis(grid::AbstractVector{G}, values::AbstractMatrix{V};
        weights::AbstractVector{W}=quadrature_weights(grid)) where {G<:Real,V<:Real,W<:Real}
    Base.require_one_based_indexing(grid, values, weights)
    size(values, 1) == length(grid) ||
        throw(DimensionMismatch("FPCBasis values rows must match the grid length"))
    size(values, 2) >= 1 ||
        throw(ArgumentError("FPCBasis requires at least one function"))
    length(weights) == length(grid) ||
        throw(DimensionMismatch("FPCBasis weights must match the grid length"))
    T = promote_type(_floattype(G), _floattype(V), _floattype(W))
    x, w, y = Vector{T}(grid), Vector{T}(weights), Matrix{T}(values)
    _validate_strict_grid(x)
    _interval((first(x), last(x)))
    all(v -> isfinite(v) && v > 0, w) ||
        throw(ArgumentError("FPCBasis weights must be finite and positive"))
    all(isfinite, y) || throw(ArgumentError("FPCBasis values must be finite"))
    FPCBasis{T}(Val(:validated), x, w, y)
end

Base.eltype(::Type{<:FPCBasis{T}}) where {T} = T
nbasis(b::FPCBasis) = size(b.values, 2)
domain(b::FPCBasis) = (first(b.grid), last(b.grid))

Base.:(==)(a::FPCBasis, b::FPCBasis) =
    a.grid == b.grid && a.weights == b.weights && a.values == b.values
Base.isequal(a::FPCBasis, b::FPCBasis) =
    isequal(a.grid, b.grid) && isequal(a.weights, b.weights) && isequal(a.values, b.values)
Base.hash(b::FPCBasis, h::UInt) =
    hash(b.values, hash(b.weights, hash(b.grid, hash(:FPCBasis, h))))

function Base.show(io::IO, b::FPCBasis)
    a, z = domain(b)
    print(io, "FPCBasis(", nbasis(b), " functions on [", a, ", ", z,
        "], ", length(b.grid), " grid points)")
end

Base.show(io::IO, ::MIME"text/plain", b::FPCBasis) = show(io, b)

function basis_matrix(b::FPCBasis, t::AbstractVector{<:Real}; derivative=0)
    _derivative_order(derivative) == 0 ||
        throw(ArgumentError("FPCBasis evaluation supports derivative=0 only"))
    points = _basispoints(b, t)
    out = Matrix{eltype(points)}(undef, length(points), nbasis(b))
    for k in 1:nbasis(b)
        _evaluate_curve!(@view(out[:, k]), b.grid, @view(b.values[:, k]),
            points, PiecewiseLinear())
    end
    out
end

function gram(b::FPCBasis{T}) where {T}
    out = zeros(T, nbasis(b), nbasis(b))
    inverse_sqrt_three = inv(sqrt(T(3)))
    for j in 1:nbasis(b), k in 1:j
        value = zero(T)
        for i in 1:length(b.grid)-1
            scale = sqrt(b.grid[i + 1] - b.grid[i])
            a, z = scale * b.values[i, j], scale * b.values[i + 1, j]
            c, d = scale * b.values[i, k], scale * b.values[i + 1, k]
            # Integrate midpoint and centered linear parts separately. Scaling
            # before products avoids overflow for a narrow grid and tall basis.
            midpoint_j, midpoint_k = (a + z) / 2, (c + d) / 2
            linear_j = ((z - a) / 2) * inverse_sqrt_three
            linear_k = ((d - c) / 2) * inverse_sqrt_three
            value += midpoint_j * midpoint_k + linear_j * linear_k
        end
        out[j, k] = out[k, j] = value
    end
    out
end

function penalty(b::FPCBasis, order)
    _derivative_order(order) == 0 ||
        throw(ArgumentError("FPCBasis derivative penalties of order >= 1 are not supported"))
    gram(b)
end
