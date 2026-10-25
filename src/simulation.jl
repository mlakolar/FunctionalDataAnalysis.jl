"""
    RegularSampling(grid)

Sample every simulated curve on a copied, finite, strictly increasing grid
of at least two points. `simulate_kl` returns grid data for this design.
"""
struct RegularSampling{T<:AbstractFloat}
    grid::Vector{T}
    function RegularSampling(grid::AbstractVector{S}) where {S<:Real}
        T = _floattype(S)
        points = Vector{T}(grid)
        quadrature_weights(points)
        new{T}(points)
    end
end

"""
    SparseSampling(domain, npoints)

Draw an observation count uniformly from the nonempty positive integer
range `npoints` for every curve, then draw that many distinct uniform times
on the finite interval `domain`. Times are sorted. `simulate_kl` returns
irregular data. If floating-point precision prevents distinct times, throw
instead of dropping observations.
"""
struct SparseSampling{T<:AbstractFloat}
    domain::Tuple{T,T}
    npoints::StepRange{Int,Int}
    function SparseSampling(domain::Tuple{<:Real,<:Real}, npoints::AbstractRange{<:Integer})
        bounds = _interval(domain)
        !isempty(npoints) && _integer_in(first(npoints), 1) && _integer_in(last(npoints), 1) &&
            !(step(npoints) isa Bool) && typemin(Int) <= step(npoints) <= typemax(Int) ||
            throw(ArgumentError("npoints must be a nonempty range of positive integers"))
        counts = Int(first(npoints)):Int(step(npoints)):Int(last(npoints))
        new{typeof(bounds[1])}(bounds, counts)
    end
end

struct _FourierEigenfunction{T<:AbstractFloat}
    domain::Tuple{T,T}
    index::Int
end
struct _LegendreEigenfunction{T<:AbstractFloat}
    domain::Tuple{T,T}
    degree::Int
end

function _eigenpoint(bounds, t)
    a, b = bounds
    t isa Real && isfinite(t) && a <= t <= b ||
        throw(ArgumentError("eigenfunction evaluation point must be finite and inside the domain"))
    a, b
end

function (f::_FourierEigenfunction{T})(t::Real) where {T}
    a, b = _eigenpoint(f.domain, t)
    R = promote_type(T, _floattype(typeof(t)))
    width = R(b - a)
    f.index == 1 && return inv(sqrt(width))
    k = div(f.index, 2)
    phase = R(2k) * (R(t) - a) / width
    sqrt(R(2) / width) * (iseven(f.index) ? sinpi(phase) : cospi(phase))
end

function (f::_LegendreEigenfunction{T})(t::Real) where {T}
    a, b = _eigenpoint(f.domain, t)
    R = promote_type(T, _floattype(typeof(t)))
    z = R(2) * (R(t) - a) / (b - a) - one(R)
    previous, current = one(R), z
    if f.degree == 0
        current = previous
    else
        for k in 2:f.degree
            previous, current = current, ((2k - 1) * z * current - (k - 1) * previous) / k
        end
    end
    sqrt(R(2 * f.degree + 1) / (b - a)) * current
end

"""
    fourier_eigenfunctions(domain, K)

Return `K` orthonormal callable Fourier functions on `domain`: a constant,
then sine/cosine pairs of increasing frequency, with the normalization of
`FourierBasis`. `K` is nonnegative and need not be odd. This deterministic
utility does not use an RNG. Evaluation outside the domain throws.
"""
function fourier_eigenfunctions(domain::Tuple{<:Real,<:Real}, K::Integer)
    _integer_in(K, 0) || throw(ArgumentError("number of eigenfunctions must be nonnegative"))
    bounds = _interval(domain)
    T = typeof(bounds[1])
    out = Vector{_FourierEigenfunction{T}}(undef, K)
    for k in 1:K
        out[k] = _FourierEigenfunction{T}(bounds, k)
    end
    out
end

"""
    legendre_eigenfunctions(domain, K)

Return `K` orthonormal callable shifted Legendre polynomials on `domain`,
starting with degree zero. Degree `k` has scale `sqrt((2k+1)/width)` and
argument `2(t-lower)/width-1`. `K` is nonnegative. This deterministic utility
does not use an RNG. Evaluation outside the domain throws.
"""
function legendre_eigenfunctions(domain::Tuple{<:Real,<:Real}, K::Integer)
    _integer_in(K, 0) || throw(ArgumentError("number of eigenfunctions must be nonnegative"))
    bounds = _interval(domain)
    T = typeof(bounds[1])
    out = Vector{_LegendreEigenfunction{T}}(undef, K)
    for k in 1:K
        out[k] = _LegendreEigenfunction{T}(bounds, k - 1)
    end
    out
end

function _simulation_parameters(n, eigenfunctions, eigenvalues, noise_sd, sampling)
    _integer_in(n, 1) || throw(ArgumentError("simulation requires at least one curve"))
    eigenvalues isa AbstractVector{<:Real} ||
        throw(ArgumentError("eigenvalues must be a real vector"))
    length(eigenfunctions) == length(eigenvalues) ||
        throw(DimensionMismatch("eigenfunctions and eigenvalues must have equal length"))
    all(x -> isfinite(x) && x >= 0, eigenvalues) ||
        throw(ArgumentError("eigenvalues must be finite and nonnegative"))
    noise_sd isa Real && isfinite(noise_sd) && noise_sd >= 0 ||
        throw(ArgumentError("noise_sd must be finite and nonnegative"))
    sampling isa Union{RegularSampling,SparseSampling} ||
        throw(ArgumentError("sampling must be RegularSampling or SparseSampling"))
    T = promote_type(_sampling_type(sampling), _floattype(eltype(eigenvalues)))
    T, Vector{T}(eigenvalues), T(noise_sd)
end
_sampling_type(::RegularSampling{T}) where {T} = T
_sampling_type(::SparseSampling{T}) where {T} = T

function _validate_simulation_callables(mean, eigenfunctions, t)
    applicable(mean, t) || throw(ArgumentError("mean must be callable at observation times"))
    for f in eigenfunctions
        applicable(f, t) || throw(ArgumentError("eigenfunctions must be callable at observation times"))
    end
    nothing
end

function _truth_value(mean, eigenfunctions, scores, i, t, ::Type{T}) where {T}
    mu = mean(t)
    mu isa Real && isfinite(mu) || throw(ArgumentError("mean must return finite real values"))
    value = T(mu)
    for k in eachindex(eigenfunctions)
        raw = eigenfunctions[k](t)
        raw isa Real && isfinite(raw) ||
            throw(ArgumentError("eigenfunctions must return finite real values"))
        phi = T(raw)
        value += scores[i, k] * phi
    end
    isfinite(value) || throw(ArgumentError("noise-free simulated values must be finite"))
    value
end

function _simulate_kl(rng, n, mean, eigenfunctions, values::Vector{T}, noise_sd,
        sampling::RegularSampling) where {T}
    grid = Vector{T}(sampling.grid)
    _validate_simulation_callables(mean, eigenfunctions, first(grid))
    scores = randn(rng, T, n, length(values)) .* transpose(sqrt.(values))
    truth = Matrix{T}(undef, n, length(grid))
    observed = similar(truth)
    for i in 1:n, j in eachindex(grid)
        truth[i, j] = _truth_value(mean, eigenfunctions, scores, i, grid[j], T)
        observed[i, j] = truth[i, j] + (noise_sd == 0 ? zero(T) : noise_sd * randn(rng, T))
    end
    (data=GridFunctionalData(observed, grid), scores=scores,
        truth=GridFunctionalData(truth, grid))
end

function _sample_times(rng, sampling::SparseSampling, count, ::Type{T}) where {T}
    a, b = T(sampling.domain[1]), T(sampling.domain[2])
    for attempt in 1:1000
        times = sort!(a .+ (b - a) .* rand(rng, T, count))
        all(>(0), diff(times)) && return times
    end
    throw(ArgumentError("sampling domain cannot provide the requested distinct times at this precision"))
end

function _simulate_kl(rng, n, mean, eigenfunctions, values::Vector{T}, noise_sd,
        sampling::SparseSampling) where {T}
    _validate_simulation_callables(mean, eigenfunctions, T(first(sampling.domain)))
    scores = randn(rng, T, n, length(values)) .* transpose(sqrt.(values))
    times = [_sample_times(rng, sampling, rand(rng, sampling.npoints), T) for _ in 1:n]
    truth = Vector{T}[Vector{T}(undef, length(t)) for t in times]
    observed = Vector{T}[similar(y) for y in truth]
    for i in 1:n, j in eachindex(times[i])
        truth[i][j] = _truth_value(mean, eigenfunctions, scores, i, times[i][j], T)
        observed[i][j] = truth[i][j] + (noise_sd == 0 ? zero(T) : noise_sd * randn(rng, T))
    end
    (data=IrregularFunctionalData(times, observed; domain=sampling.domain), scores=scores,
        truth=IrregularFunctionalData(times, truth; domain=sampling.domain))
end

"""
    simulate_kl(rng::AbstractRNG, n; mean, eigenfunctions, eigenvalues,
                noise_sd=0, sampling)

Draw independent curves `mean(t) + sum(scores[i,k]*eigenfunctions[k](t)) + ε`
with independent Gaussian scores of variances `eigenvalues` and independent
Gaussian measurement noise of standard deviation `noise_sd`. All randomness
uses `rng`. Eigenvalues and noise must be finite and nonnegative.

Return `(data, scores, truth)`. Regular sampling returns grid data; sparse
sampling returns irregular data. `truth` is an independent data object of
the same type on the same observed times, containing noise-free values.
`scores` is `n × length(eigenvalues)`. No smoothing or FPCA is performed.
"""
function simulate_kl(rng::AbstractRNG, n::Integer; mean, eigenfunctions,
        eigenvalues, noise_sd=0, sampling)
    _, values, noise = _simulation_parameters(n, eigenfunctions, eigenvalues, noise_sd, sampling)
    _simulate_kl(rng, Int(n), mean, eigenfunctions, values, noise, sampling)
end
