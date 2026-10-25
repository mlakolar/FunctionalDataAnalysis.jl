# Internal P-splines. Unlike penalty(basis, m), this penalizes coefficient
# differences, not integrated derivatives. refund 0.1-40 pspline.setting.R.
function _difference_penalty(k::Integer, m::Integer, ::Type{T}=Float64) where {T}
    0 <= m < k || throw(ArgumentError("difference penalty order m must be in 0:nbasis-1"))
    D = zeros(T, k-m, k)
    for i in axes(D, 1), j in 0:m
        D[i, i+j] = T((-1)^j * binomial(big(m), j))
    end
    all(isfinite, D) || throw(ArgumentError("difference penalty coefficients overflow; reduce m"))
    P = transpose(D) * D
    all(isfinite, P) || throw(ArgumentError("difference penalty overflows; reduce m"))
    P
end

function _pspline_setting(x::AbstractVector{T}, knots, p, m) where {T<:AbstractFloat}
    knots isa Integer && !(knots isa Bool) && knots >= 1 || throw(ArgumentError("FACE knots must be a positive integer (number of equal intervals)"))
    p isa Integer && !(p isa Bool) && p >= 0 || throw(ArgumentError("FACE degree p must be a nonnegative integer"))
    knots < length(x) && p < length(x)-knots || throw(ArgumentError("FACE knots + p must be smaller than the grid length"))
    k = Int(knots+p)
    m isa Integer && !(m isa Bool) && 0 <= m < k || throw(ArgumentError("FACE difference penalty order m must be in 0:knots+p-1"))
    p, m, knots = Int(p), Int(m), Int(knots)
    # R uses simple knots extending p intervals beyond each endpoint, rather
    # than clamped endpoints. BSplineKit's augment=false preserves these knots.
    a, b = first(x), last(x)
    sequence = T[a + (b-a)*j/knots for j in -p:(knots+p)]
    backend = BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(p+1), sequence; augment=Val(false))
    basis = BSplineBasis{T}(backend, sequence[(p+1):(end-p)], p+1)
    # This internal wrapper has an unaugmented backend; its dimension is k.
    B = zeros(T, length(x), k)
    BSplineKit.collocation_matrix!(B, basis.backend, x; clip_threshold=0)
    P = _difference_penalty(k, m, T)
    gram_eigen = eigen(Symmetric(transpose(B)*B))
    minimum(gram_eigen.values) > T(1e-7) || throw(ArgumentError(
        "FACE B-spline design is rank deficient or ill-conditioned; reduce knots or degree p"))
    # Deliberate departure: refund silently adds 1e-6 to every eigenvalue of
    # an ill-conditioned design. Input validation replaces that hidden ridge.
    inverse_root = gram_eigen.vectors * Diagonal(inv.(sqrt.(gram_eigen.values))) * transpose(gram_eigen.vectors)
    eig = eigen(Symmetric(inverse_root * P * inverse_root))
    s = max.(eig.values, zero(T))
    m > 0 && (s[1:m] .= 0)
    A0 = inverse_root * eig.vectors
    (; basis, knots=sequence, B, P, s, A0, A=B*A0)
end

# Brent's bounded search, with the stopping constants used by R 4.5 stats
# smooth.spline (sbart.c). Return the LAST evaluated tuning value: that is the
# spline stored by R, even when a previous point has the best criterion.
function _spline_gcv_search(f)
    a, b = -1.5, 1.5
    golden = (3-sqrt(5.0))/2
    x = w = v = a + golden*(b-a)
    fx = fw = fv = f(x)
    e = d = 0.0
    last = x
    for _ in 1:500
        mid = (a+b)/2
        tol = 2e-8*abs(x) + 1e-4/3
        abs(x-mid) <= 2tol-(b-a)/2 && return last
        parabolic = false
        if abs(e) > tol && max(fx, fw, fv) < 1e100
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q - (x-w)*r
            q = 2(q-r)
            q > 0 && (p = -p)
            q = abs(q)
            r, e = e, d
            if q != 0 && abs(p) < abs(q*r/2) && q*(a-x) < p < q*(b-x)
                d = p/q
                u = x+d
                (u-a < 2tol || b-u < 2tol) && (d = copysign(tol, mid-x))
                parabolic = true
            end
        end
        if !parabolic
            e = x >= mid ? a-x : b-x
            d = golden*e
        end
        u = x + (abs(d) >= tol ? d : copysign(tol, d))
        last = u
        fu = f(u)
        if fu <= fx
            u >= x ? (a=x) : (b=x)
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else
            u < x ? (a=u) : (b=u)
            if fu <= fw || w == x
                v, fv, w, fw = w, fw, u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    throw(ArgumentError("smoothing-spline GCV search did not converge"))
end

function _spline_nknots(n)
    n < 50 && return n
    n < 200 && return floor(Int, 2^(log2(50)+(log2(100)-log2(50))*(n-50)/150))
    n < 800 && return floor(Int, 2^(log2(100)+(log2(140)-log2(100))*(n-200)/600))
    n < 3200 && return floor(Int, 2^(log2(140)+(log2(200)-log2(140))*(n-800)/2400))
    floor(Int, 200+(n-3200)^0.2)
end

# R's stats/src/sgram.f uses 0.3330, rather than 1/3, for the product of
# second-derivative slopes. Keep that legacy approximation local to missing
# data initialization. All ordinary PenalizedSpline fits keep exact integrals.
struct _RInitializationBasis{T,B<:BSplineBasis} <: AbstractBasis
    basis::B
    roughness::Matrix{T}
end
Base.eltype(::Type{<:_RInitializationBasis{T}}) where {T} = T
nbasis(b::_RInitializationBasis) = nbasis(b.basis)
domain(b::_RInitializationBasis) = domain(b.basis)
basismatrix(b::_RInitializationBasis, t::AbstractVector{<:Real}; derivative=0) =
    basismatrix(b.basis, t; derivative)
penalty(b::_RInitializationBasis, order) = order == 2 ? copy(b.roughness) : penalty(b.basis, order)

# Initialization only: reproduce stats::smooth.spline's default knot count and
# normalized coordinates, using the existing integrated-penalty smoother.
function _face_initial_curve(t, y, targets)
    length(t) >= 4 || throw(ArgumentError("FACE missing-value initialization needs at least four observations per incomplete curve"))
    a, b = first(t), last(t)
    x = (t .- a) ./ (b-a)
    nk = _spline_nknots(length(t))
    indices = floor.(Int, range(1, length(t); length=nk))
    cubic = BSplineBasis(4, x[indices])
    slopes = diff(basismatrix(cubic, cubic.breakpoints; derivative=2); dims=1)
    roughness = penalty(cubic, 2) - transpose(slopes)*(diff(cubic.breakpoints) .* slopes)/3000
    basis = _RInitializationBasis(cubic, roughness)
    X = basismatrix(basis, x)
    P = penalty(basis, 2)
    XtX = transpose(X)*X
    inner = 3:(nbasis(basis)-3)
    ratio = sum(diag(XtX)[inner])/sum(diag(P)[inner])
    raw = GridFunctionalData(reshape(y, 1, :), x)
    criterion(spar) = only(_gcv_selection(raw, basis, 2, [ratio*16.0^(6spar-2)]).scores)
    spar = _spline_gcv_search(criterion)
    lambda = ratio*16.0^(6spar-2)
    smoothed = smooth_curves(raw, PenalizedSpline(basis; lambda))
    out = fill(sum(y)/length(y), length(targets))
    inside = findall(z -> a <= z <= b, targets)
    out[inside] = vec(evaluate(smoothed, (targets[inside] .- a) ./ (b-a)))
    (; values=out, lambda, spar, knots=copy(cubic.breakpoints))
end
