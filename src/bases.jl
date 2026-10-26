"""
    nbasis(b)

Number of functions in the basis `b`.
"""
function nbasis end

"""
    basis_matrix(b, t; derivative=0)

Evaluate a basis, or its nonnegative integer derivative, at `t`. Return a
`length(t) × nbasis(b)` matrix. All points must be finite and in `domain(b)`.
"""
function basis_matrix end

"""
    gram(b)

Return the matrix of integrals of pairwise products of basis functions over
`domain(b)`.
"""
function gram end

"""
    penalty(b, order)
    penalty(b::FourierBasis, :harmonic_acceleration)

Return the matrix of integrals of pairwise products of derivatives of the
nonnegative integer `order`. Order zero is the Gram matrix.
For a Fourier basis, `:harmonic_acceleration` integrates products of
`Lφ = D³φ + (2π/domain_width)² Dφ`, the default operator in fda 6.3.0
`fdPar`. Its null space contains the constant and first sine/cosine pair.
"""
function penalty end

_floattype(::Type{T}) where {T<:Real} = T <: Integer ? Float64 : float(T)

function _interval(interval::Tuple{<:Real,<:Real})
    T = _floattype(promote_type(typeof(interval[1]), typeof(interval[2])))
    a, b = T(interval[1]), T(interval[2])
    isfinite(a) && isfinite(b) && a < b && isfinite(b - a) ||
        throw(ArgumentError("domain must have finite increasing endpoints and finite width"))
    (a, b)
end

function _derivative_order(order)
    _integer_in(order, 0) ||
        throw(ArgumentError("derivative order must be a nonnegative integer representable as Int"))
    Int(order)
end

function _derivative_sum(a, b)
    nu = _derivative_order(b)
    a <= typemax(Int) - nu || throw(ArgumentError("total derivative order exceeds Int"))
    a + nu
end

function _basispoints(b, t::AbstractVector{<:Real})
    Base.require_one_based_indexing(t)
    a, z = domain(b)
    all(x -> isfinite(x) && a <= x <= z, t) ||
        throw(ArgumentError("evaluation points must be finite and inside the basis domain"))
    R = promote_type(eltype(b), _floattype(eltype(t)))
    Vector{R}(t)
end

"""
    BSplineBasis(order, breakpoints)
    BSplineBasis(order, domain, nbasis)

A clamped B-spline basis wrapping `BSplineKit.BSplineBasis`. The positive
integer `order` is the polynomial degree plus one. Breakpoints are copied
and must be finite and strictly increasing. The second form uses equally
spaced breakpoints and requires `nbasis >= order`. Endpoint multiplicity is
`order`; derivatives at knots follow BSplineKit's one-sided convention.
"""
struct BSplineBasis{T<:AbstractFloat,B<:BSplineKit.BSplineBasis} <: AbstractBasis
    backend::B
    breakpoints::Vector{T}
    order::Int
    function BSplineBasis{T}(backend::B, breakpoints, order) where
            {T<:AbstractFloat,B<:BSplineKit.BSplineBasis}
        new{T,B}(backend, breakpoints, order)
    end
end

Base.@constprop :aggressive function BSplineBasis(order::Integer, breakpoints::AbstractVector{<:Real})
    _integer_in(order, 1) || throw(ArgumentError("B-spline order must be positive"))
    Base.require_one_based_indexing(breakpoints)
    T = _floattype(eltype(breakpoints))
    x = Vector{T}(breakpoints)
    _validate_strict_grid(x)
    _interval((first(x), last(x)))
    backend = BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(Int(order)), copy(x))
    BSplineBasis{T}(backend, x, Int(order))
end

Base.@constprop :aggressive function BSplineBasis(order::Integer, interval::Tuple{<:Real,<:Real}, n::Integer)
    _integer_in(order, 1) || throw(ArgumentError("B-spline order must be positive"))
    _integer_in(n, order) || throw(ArgumentError("number of B-splines must be at least the order"))
    a, b = _interval(interval)
    BSplineBasis(order, collect(range(a, b; length=n - order + 2)))
end

"""
    FourierBasis(domain, nbasis)

An orthonormal Fourier basis on the finite interval `domain`. The positive
odd `nbasis` counts the constant followed by sine/cosine pairs of increasing
frequency. The constant is `1/sqrt(width)`; each pair has scale
`sqrt(2/width)` and argument `2πk * (t - lower)/width`.
"""
struct FourierBasis{T<:AbstractFloat} <: AbstractBasis
    interval::Tuple{T,T}
    n::Int
    function FourierBasis(interval::Tuple{<:Real,<:Real}, n::Integer)
        _integer_in(n, 1) && isodd(n) || throw(ArgumentError("Fourier nbasis must be positive and odd"))
        bounds = _interval(interval)
        new{typeof(bounds[1])}(bounds, Int(n))
    end
end

"""
    MonomialBasis(domain, degree)

The functions `1, t, ..., t^degree` on the finite interval `domain`.
The polynomial degree must be a nonnegative integer.
Raw monomials are ill-conditioned at high degrees or on domains far from
zero. Consider [`BSplineBasis`](@ref) for those settings.
"""
struct MonomialBasis{T<:AbstractFloat} <: AbstractBasis
    interval::Tuple{T,T}
    degree::Int
    function MonomialBasis(interval::Tuple{<:Real,<:Real}, degree::Integer)
        _integer_in(degree, 0, typemax(Int)-1) || throw(ArgumentError("monomial degree must be nonnegative"))
        bounds = _interval(interval)
        new{typeof(bounds[1])}(bounds, Int(degree))
    end
end

Base.eltype(::Type{<:BSplineBasis{T}}) where {T} = T
Base.eltype(::Type{<:FourierBasis{T}}) where {T} = T
Base.eltype(::Type{<:MonomialBasis{T}}) where {T} = T
Base.eltype(b::AbstractBasis) = eltype(typeof(b))
nbasis(b::BSplineBasis) = length(b.breakpoints) + b.order - 2
nbasis(b::FourierBasis) = b.n
nbasis(b::MonomialBasis) = b.degree + 1
domain(b::BSplineBasis) = (first(b.breakpoints), last(b.breakpoints))
domain(b::Union{FourierBasis,MonomialBasis}) = b.interval

Base.:(==)(a::BSplineBasis, b::BSplineBasis) =
    a.order == b.order && a.breakpoints == b.breakpoints
Base.:(==)(a::FourierBasis, b::FourierBasis) = a.interval == b.interval && a.n == b.n
Base.:(==)(a::MonomialBasis, b::MonomialBasis) =
    a.interval == b.interval && a.degree == b.degree

Base.hash(b::BSplineBasis, h::UInt) =
    hash(b.breakpoints, hash(b.order, hash(:BSplineBasis, h)))
Base.hash(b::FourierBasis, h::UInt) =
    hash(b.interval, hash(b.n, hash(:FourierBasis, h)))
Base.hash(b::MonomialBasis, h::UInt) =
    hash(b.interval, hash(b.degree, hash(:MonomialBasis, h)))

function Base.show(io::IO, b::BSplineBasis)
    a, z = domain(b)
    print(io, "BSplineBasis(order ", b.order, ", ", nbasis(b),
        " functions on [", a, ", ", z, "])")
end

function Base.show(io::IO, b::FourierBasis)
    a, z = domain(b)
    print(io, "FourierBasis(", nbasis(b), " functions on [", a, ", ", z, "])")
end

function Base.show(io::IO, b::MonomialBasis)
    a, z = domain(b)
    print(io, "MonomialBasis(degree ", b.degree, ", ", nbasis(b),
        " functions on [", a, ", ", z, "])")
end

Base.show(io::IO, ::MIME"text/plain", b::Union{BSplineBasis,FourierBasis,MonomialBasis}) =
    show(io, b)

function basis_matrix(b::BSplineBasis, t::AbstractVector{<:Real}; derivative=0)
    m = _derivative_order(derivative)
    x = _basispoints(b, t)
    out = zeros(eltype(x), length(x), nbasis(b))
    if m < b.order && !isempty(x)
        BSplineKit.collocation_matrix!(out, b.backend, x, BSplineKit.Derivative(m);
            clip_threshold=0)
    end
    out
end

function basis_matrix(b::FourierBasis, t::AbstractVector{<:Real}; derivative=0)
    m = _derivative_order(derivative)
    x = _basispoints(b, t)
    T = eltype(x)
    a, z = domain(b)
    width = T(z - a)
    out = zeros(T, length(x), nbasis(b))
    m == 0 && (out[:, 1] .= inv(sqrt(width)))
    for k in 1:div(b.n - 1, 2)
        frequency = T(2) * T(pi) * k / width
        scale = sqrt(T(2) / width) * frequency^m
        for i in eachindex(x)
            phase = T(2k) * (x[i] - a) / width + T(m) / 2
            out[i, 2k] = scale * sinpi(phase)
            out[i, 2k + 1] = scale * cospi(phase)
        end
    end
    out
end

function _falling(p, m, ::Type{T}) where {T}
    value = one(T)
    for k in (p - m + 1):p
        value *= k
    end
    value
end

function basis_matrix(b::MonomialBasis, t::AbstractVector{<:Real}; derivative=0)
    m = _derivative_order(derivative)
    x = _basispoints(b, t)
    T = eltype(x)
    out = zeros(T, length(x), nbasis(b))
    for p in m:b.degree
        out[:, p + 1] .= _falling(p, m, T) .* x .^ (p - m)
    end
    out
end

gram(b::AbstractBasis) = penalty(b, 0)

function penalty(b::BSplineBasis{T}, order) where {T}
    m = _derivative_order(order)
    m >= b.order && return zeros(T, nbasis(b), nbasis(b))
    d = BSplineKit.Derivative(m)
    out = zeros(T, nbasis(b), nbasis(b))
    BSplineKit.galerkin_matrix!(out, b.backend, (d, d))
    (out + transpose(out)) / 2
end

function penalty(b::FourierBasis{T}, order) where {T}
    if order === :harmonic_acceleration
        omega = T(2) * T(pi) / (b.interval[2] - b.interval[1])
        diagonal = zeros(T, nbasis(b))
        for k in 2:div(b.n - 1, 2)
            value = ((k * omega) * (omega^2 - (k * omega)^2))^2
            diagonal[2k] = diagonal[2k + 1] = value
        end
        return Matrix(Diagonal(diagonal))
    end
    m = _derivative_order(order)
    diagonal = zeros(T, nbasis(b))
    diagonal[1] = m == 0 ? one(T) : zero(T)
    a, z = domain(b)
    for k in 1:div(b.n - 1, 2)
        value = (T(2) * T(pi) * k / (z - a))^(2m)
        diagonal[2k] = diagonal[2k + 1] = value
    end
    Matrix(Diagonal(diagonal))
end

function penalty(b::MonomialBasis{T}, order) where {T}
    m = _derivative_order(order)
    a, z = domain(b)
    out = zeros(T, nbasis(b), nbasis(b))
    for p in m:b.degree, q in m:b.degree
        power = p + q - 2m + 1
        out[p + 1, q + 1] = _falling(p, m, T) * _falling(q, m, T) *
            (z^power - a^power) / power
    end
    out
end

# ---------------------------------------------------------------------------
# mgcv P-splines.

"""
    PSplineBasis(domain, nbasis; degree=3, difference_order=2)

The B-spline basis of mgcv's P-spline smooth (`bs="ps"`, mgcv 1.9-3
`smooth.construct.ps.smooth.spec`) on the finite interval `domain`. The
`nbasis` functions of polynomial `degree` have simple, equally spaced knots
that extend `degree` intervals beyond a range widened by 0.1% on each side,
exactly as mgcv places them when it is given no knots (or only the range):
with `[a, b] = domain`, `r = b - a`, `nk = nbasis - degree + 1`,
`xl = a - r/1000`, `xu = b + r/1000` and `dx = (xu - xl)/(nk - 1)`, the
`nbasis + degree + 1` knots run from `xl - degree*dx` to `xu + degree*dx`.
mgcv's `m = c(m1, m2)` corresponds to `degree = m1 + 1` and
`difference_order = m2`; the defaults are mgcv's `m = c(2, 2)`. mgcv takes the
range from the data, so pass the range of the observation times to reproduce
it. Requires `nbasis >= degree + 1` and `0 <= difference_order < nbasis`.

[`difference_penalty`](@ref) gives the P-spline penalty `D'D` (mgcv's `S`
before its `S.scale` normalization). [`penalty`](@ref) and [`gram`](@ref)
integrate products of derivatives over `domain`, as for every other basis.
Evaluation outside `domain` is an error, although the functions do not vanish
at its ends.
"""
struct PSplineBasis{T<:AbstractFloat,B<:BSplineKit.BSplineBasis} <: AbstractBasis
    backend::B
    interval::Tuple{T,T}
    knots::Vector{T}
    degree::Int
    difference_order::Int
end

function PSplineBasis(interval::Tuple{<:Real,<:Real}, n::Integer; degree::Integer=3, difference_order::Integer=2)
    _integer_in(degree, 0) || throw(ArgumentError("P-spline degree must be a nonnegative integer"))
    _pspline_basis(interval, n, Val(Int(degree)), difference_order)
end

# A degree known to the compiler gives a concrete backend type.
function _pspline_basis(interval::Tuple{<:Real,<:Real}, n::Integer, ::Val{p}, difference_order::Integer) where {p}
    _integer_in(n, p + 1) || throw(ArgumentError(
        "P-spline nbasis must be at least degree + 1 (mgcv needs two knots inside the range)"))
    _integer_in(difference_order, 0, n - 1) || throw(ArgumentError(
        "P-spline difference_order must be in 0:nbasis-1"))
    a, b = _interval(interval)
    T = typeof(a)
    k = Int(n)
    nk = k - p + 1
    r = b - a
    xl, xu = a - r*T(0.001), b + r*T(0.001)
    dx = (xu - xl)/(nk - 1)
    # R's seq(from, to, length.out): exact ends, from + i*by in between.
    from, to = xl - dx*p, xu + dx*p
    count = nk + 2p
    by = (to - from)/(count - 1)
    knots = T[i == 0 ? from : i == count - 1 ? to : from + i*by for i in 0:count-1]
    all(isfinite, knots) && issorted(knots; lt=<=) || throw(ArgumentError(
        "P-spline knots must be finite and strictly increasing; use a smaller nbasis"))
    backend = BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(p + 1), copy(knots); augment=Val(false))
    PSplineBasis{T,typeof(backend)}(backend, (a, b), knots, p, Int(difference_order))
end

Base.eltype(::Type{<:PSplineBasis{T}}) where {T} = T
nbasis(b::PSplineBasis) = length(b.knots) - b.degree - 1
domain(b::PSplineBasis) = b.interval
Base.:(==)(a::PSplineBasis, b::PSplineBasis) = a.interval == b.interval &&
    a.knots == b.knots && a.degree == b.degree && a.difference_order == b.difference_order
Base.hash(b::PSplineBasis, h::UInt) = hash(b.interval, hash(b.knots,
    hash(b.degree, hash(b.difference_order, hash(:PSplineBasis, h)))))

function Base.show(io::IO, b::PSplineBasis)
    a, z = domain(b)
    print(io, "PSplineBasis(degree ", b.degree, ", ", nbasis(b), " functions on [", a, ", ", z,
        "], difference order ", b.difference_order, ")")
end

Base.show(io::IO, ::MIME"text/plain", b::PSplineBasis) = show(io, b)

function basis_matrix(b::PSplineBasis, t::AbstractVector{<:Real}; derivative=0)
    m = _derivative_order(derivative)
    x = _basispoints(b, t)
    out = zeros(eltype(x), length(x), nbasis(b))
    if m <= b.degree && !isempty(x)
        BSplineKit.collocation_matrix!(out, b.backend, x, BSplineKit.Derivative(m);
            clip_threshold=0)
    end
    out
end

# On its domain a P-spline basis spans the clamped spline space with the knots
# inside the domain as breakpoints. Integrate exactly in that basis and map
# back through the collocation change of basis at the Greville abscissae.
function penalty(b::PSplineBasis{T}, order) where {T}
    m = _derivative_order(order)
    k = nbasis(b)
    m > b.degree && return zeros(T, k, k)
    lo, hi = domain(b)
    clamped = BSplineBasis(b.degree + 1, vcat(lo, filter(x -> lo < x < hi, b.knots), hi))
    nbasis(clamped) == k || throw(ArgumentError(
        "P-spline basis has more functions than its domain supports; use a smaller nbasis"))
    sequence = vcat(fill(lo, b.degree), clamped.breakpoints, fill(hi, b.degree))
    points = [sum(view(sequence, j+1:j+b.degree))/max(b.degree, 1) for j in 1:k]
    b.degree == 0 && (points = (clamped.breakpoints[1:end-1] + clamped.breakpoints[2:end])/2)
    change = basis_matrix(clamped, points) \ basis_matrix(b, points)
    out = transpose(change)*penalty(clamped, m)*change
    (out + transpose(out))/2
end

"""
    difference_penalty(b::PSplineBasis, order=b.difference_order)

Return the P-spline penalty `D'D`, where `D` takes coefficient differences of
the nonnegative integer `order` (the identity for order zero). This is mgcv
1.9-3's `ps` penalty `S` before mgcv rescales it by `S.scale`.
"""
difference_penalty(b::PSplineBasis{T}, order=b.difference_order) where {T} =
    _difference_penalty(nbasis(b), order, T)
