# One-dimensional thin plate regression splines with second-derivative
# penalty, after mgcv 1.9-3 smooth.construct.tp.smooth.spec and tprs.c
# (Wood 2003, JRSSB 65:95-114).

"""
    ThinPlateRegressionBasis(points, k=10)

The rank-`k` thin plate regression spline basis of mgcv 1.9-3 `s(x, bs="tp")`
in one dimension, with penalty order `m = 2` (`∫f''(t)²dt`). It is refund's
default `lf()` basis, where `k=10` is mgcv's default dimension.

The finite `points` (usually a covariate grid; ties are merged) are the knots
of the full spline. Their mean is subtracted for stability (mgcv `shift`).
With `η(r) = |r|³/12` and `E[i,j] = η(xᵢ-xⱼ)` over the `u` distinct points,
the basis keeps the eigenvectors `U` of the `k` eigenvalues of `E` of largest
magnitude (ordered by decreasing value), imposes `Tᵀδ = 0` for the null-space
polynomials `T = [1, x]` by mgcv's Householder factorization, and appends `T`.
Functions are `f(t) = Σⱼ η(t-xⱼ)δⱼ + α₀ + α₁t` with `[δ; α] = UZ * γ`. As in
mgcv, every column is scaled to unit root mean square over the supplied
points. When `u == k` no eigen truncation is applied. mgcv uses a Lanczos
iteration with relative tolerance `eps^0.7`; this basis uses LAPACK's
symmetric eigensolver, so it agrees with mgcv up to column signs and the
Lanczos tolerance. On points symmetric about their mean, some Householder
signs in mgcv's construction are decided by rounding noise; this basis uses
the exact-arithmetic choice, so its columns can then differ from mgcv's by a
rotation of the same function space and penalty.

The domain is the range of `points`. `penalty(b, 2)` integrates `f''²` exactly
over the domain (the functions are natural cubic splines, so this equals
mgcv's penalty `γᵀSγ`, available unscaled as `b.penalty`). `basis_matrix`
supports derivatives of any order (zero beyond the third).
"""
struct ThinPlateRegressionBasis{T<:AbstractFloat} <: AbstractBasis
    knots::Vector{T}
    shift::T
    transform::Matrix{T}
    penalty::Matrix{T}
    eigenvalues::Vector{T}
    interval::Tuple{T,T}
    k::Int
    function ThinPlateRegressionBasis{T}(::Val{:validated}, knots, shift, transform, penalty,
            eigenvalues, interval, k) where {T<:AbstractFloat}
        new{T}(knots, shift, transform, penalty, eigenvalues, interval, k)
    end
end

const _TPRS_NULL_DIMENSION = 2

_tprs_eta(r) = abs(r)^3 / 12
function _tprs_eta(r, m)
    m == 0 && return abs(r)^3 / 12
    m == 1 && return r * abs(r) / 4
    m == 2 && return abs(r) / 2
    m == 3 && return sign(r) / 2
    zero(r)
end

# mgcv tprs.c QT(Q, A, fullQ=1): A*Q = [0, B] with B reverse lower triangular.
# Row i is rotated by a Householder reflection into column size(A, 2)-i+1.
function _mgcv_qt(A::Matrix{T}) where {T}
    A = copy(A)
    r, c = size(A)
    Q = Matrix{T}(I, c, c)
    for i in 1:r
        len = c - i + 1
        p = A[i, 1:len]
        scale = maximum(abs, p)
        scale > 0 && (p ./= scale)
        lsq = sqrt(sum(abs2, p))
        p[len] < 0 && (lsq = -lsq)
        p[len] += lsq
        g = lsq != 0 ? 1 / (lsq * p[len]) : zero(T)
        for j in (i+1):r
            x = g * dot(p, view(A, j, 1:len))
            A[j, 1:len] .-= x .* p
        end
        for j in 1:c
            x = g * dot(p, view(Q, j, 1:len))
            Q[j, 1:len] .-= x .* p
        end
    end
    Q
end

# Radial and polynomial parts at centered points, before the transform.
function _tprs_raw(knots::Vector{T}, x::AbstractVector, derivative::Int) where {T}
    out = zeros(T, length(x), length(knots) + _TPRS_NULL_DIMENSION)
    for (i, t) in enumerate(x), (j, s) in enumerate(knots)
        out[i, j] = _tprs_eta(t - s, derivative)
    end
    u = length(knots)
    derivative == 0 && (out[:, u+1] .= one(T))
    derivative == 0 && (out[:, u+2] .= x)
    derivative == 1 && (out[:, u+2] .= one(T))
    out
end

function ThinPlateRegressionBasis(points::AbstractVector{<:Real}, k::Integer=10)
    Base.require_one_based_indexing(points)
    T = _floattype(eltype(points))
    x = Vector{T}(points)
    isempty(x) && throw(ArgumentError("thin plate regression spline points must be nonempty"))
    all(isfinite, x) || throw(ArgumentError("thin plate regression spline points must be finite"))
    M = _TPRS_NULL_DIMENSION
    _integer_in(k, M + 1) || throw(ArgumentError(
        "thin plate regression spline dimension k must be an integer of at least $(M + 1)"))
    k = Int(k)
    shift = sum(x) / length(x)
    knots = sort!(unique(x .- shift))
    u = length(knots)
    u >= k || throw(ArgumentError(
        "thin plate regression spline needs at least k=$k distinct points; got $u"))
    interval = _interval((minimum(x), maximum(x)))
    E = [_tprs_eta(a - b) for a in knots, b in knots]
    Tm = hcat(ones(T, u), knots)
    if u == k
        U = Matrix{T}(I, u, u)
        D = E
        values = T[]
    else
        decomposition = eigen(Symmetric(E))
        keep = sortperm(abs.(decomposition.values); rev=true)[1:k]
        keep = keep[sortperm(decomposition.values[keep]; rev=true)]
        U = decomposition.vectors[:, keep]
        # Eigenvector signs are arbitrary; fix them for reproducible columns.
        for j in axes(U, 2)
            U[argmax(abs.(view(U, :, j))), j] < 0 && (U[:, j] .*= -1)
        end
        values = decomposition.values[keep]
        D = Diagonal(values)
    end
    # On points symmetric about their mean, entries of TᵀU that vanish in exact
    # arithmetic decide Householder signs; set them to exact zeros.
    TU = Matrix(transpose(Tm) * U)
    for i in axes(TU, 1)
        tiny = 1000 * eps(T) * maximum(abs, view(TU, i, :))
        TU[i, :] .= ifelse.(abs.(view(TU, i, :)) .<= tiny, zero(T), view(TU, i, :))
    end
    Q = _mgcv_qt(TU)
    Zc = Q[:, 1:(k-M)]
    transform = zeros(T, u + M, k)
    transform[1:u, 1:(k-M)] .= U * Zc
    transform[(u+1):(u+M), (k-M+1):k] .= Matrix{T}(I, M, M)
    S = zeros(T, k, k)
    S[1:(k-M), 1:(k-M)] .= transpose(Zc) * (D * Zc)
    S = (S + transpose(S)) / 2
    # mgcv rescales each column to unit root mean square over the data.
    design = _tprs_raw(knots, x .- shift, 0) * transform
    w = vec(sqrt.(sum(abs2, design; dims=1) ./ size(design, 1)))
    all(v -> isfinite(v) && v > 0, w) || throw(ArgumentError(
        "thin plate regression spline columns must have positive finite scale"))
    transform ./= transpose(w)
    S = S ./ w ./ transpose(w)
    ThinPlateRegressionBasis{T}(Val(:validated), knots, shift, transform, S, values,
        interval, k)
end

Base.eltype(::Type{<:ThinPlateRegressionBasis{T}}) where {T} = T
nbasis(b::ThinPlateRegressionBasis) = b.k
domain(b::ThinPlateRegressionBasis) = b.interval

Base.:(==)(a::ThinPlateRegressionBasis, b::ThinPlateRegressionBasis) =
    a.k == b.k && a.shift == b.shift && a.knots == b.knots && a.transform == b.transform
Base.hash(b::ThinPlateRegressionBasis, h::UInt) =
    hash(b.transform, hash(b.knots, hash(b.k, hash(:ThinPlateRegressionBasis, h))))

function Base.show(io::IO, b::ThinPlateRegressionBasis)
    a, z = domain(b)
    print(io, "ThinPlateRegressionBasis(", nbasis(b), " functions, ", length(b.knots),
        " knots on [", a, ", ", z, "])")
end
Base.show(io::IO, ::MIME"text/plain", b::ThinPlateRegressionBasis) = show(io, b)

function basis_matrix(b::ThinPlateRegressionBasis, t::AbstractVector{<:Real}; derivative=0)
    m = _derivative_order(derivative)
    x = _basispoints(b, t)
    _tprs_raw(b.knots, x .- b.shift, m) * b.transform
end

# Products of two cubic pieces are exact under 4-point Gauss-Legendre rules.
const _GAUSS4_NODES = (-0.8611363115940526, -0.3399810435848563,
    0.3399810435848563, 0.8611363115940526)
const _GAUSS4_WEIGHTS = (0.3478548451374538, 0.6521451548625461,
    0.6521451548625461, 0.3478548451374538)

function penalty(b::ThinPlateRegressionBasis{T}, order) where {T}
    m = _derivative_order(order)
    breaks = unique(vcat(b.interval[1], b.knots .+ b.shift, b.interval[2]))
    sort!(breaks)
    out = zeros(T, nbasis(b), nbasis(b))
    for i in 1:(length(breaks)-1)
        a, z = breaks[i], breaks[i+1]
        half, mid = (z - a) / 2, (z + a) / 2
        nodes = T[mid + half * T(s) for s in _GAUSS4_NODES]
        weights = T[half * T(w) for w in _GAUSS4_WEIGHTS]
        B = basis_matrix(b, clamp.(nodes, b.interval...); derivative=m)
        out .+= transpose(B) * (weights .* B)
    end
    (out + transpose(out)) / 2
end
