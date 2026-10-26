# Exponential-family responses for penalized IRLS (`PenalizedLeastSquares`)
# and FPC regression, after the R 4.5.1 `stats` family objects that mgcv 1.9-3
# `gam.fit3` and `glm` use.

struct _GLMFamily
    family::Symbol
    link::Symbol
end

const _GLM_LINKS = (gaussian=(:identity, :log, :inverse), binomial=(:logit, :probit, :cloglog),
    poisson=(:log, :identity, :sqrt))
const _GLM_CANONICAL = (gaussian=:identity, binomial=:logit, poisson=:log)

function _glm_family(family, link)
    family isa Symbol && family in (:gaussian, :binomial, :poisson) ||
        throw(ArgumentError("family must be :gaussian, :binomial or :poisson"))
    links = _GLM_LINKS[family]
    chosen = link === nothing ? _GLM_CANONICAL[family] : link
    chosen isa Symbol && chosen in links || throw(ArgumentError(
        "link for family=:$family must be nothing or one of " * join((":$l" for l in links), ", ")))
    _GLMFamily(family, chosen)
end

_glm_canonical(f::_GLMFamily) = f.link === _GLM_CANONICAL[f.family]
_glm_scale_known(f::_GLMFamily) = f.family !== :gaussian
_glm_identity(f::_GLMFamily) = f.family === :gaussian && f.link === :identity

# Inverse link μ = h(η), 1 - μ (accurate for the binomial links) and the first
# four derivatives of h.
function _glm_inverse_link(link::Symbol, eta::T) where {T<:AbstractFloat}
    if link === :identity
        return eta, 1 - eta, one(T), zero(T), zero(T), zero(T)
    elseif link === :log
        m = exp(eta)
        return m, 1 - m, m, m, m, m
    elseif link === :logit
        e = exp(-abs(eta))
        m, mc = eta >= 0 ? (1 / (1 + e), e / (1 + e)) : (e / (1 + e), 1 / (1 + e))
        h1 = e / (1 + e)^2
        s = mc - m
        return m, mc, h1, h1 * s, h1 * s^2 - 2h1^2, h1 * s^3 - 8h1^2 * s
    elseif link === :probit
        m = SpecialFunctions.erfc(-eta / sqrt(T(2))) / 2
        mc = SpecialFunctions.erfc(eta / sqrt(T(2))) / 2
        h1 = exp(-eta^2 / 2) / sqrt(2 * T(pi))
        return m, mc, h1, -eta * h1, (eta^2 - 1) * h1, (3eta - eta^3) * h1
    elseif link === :cloglog
        u = exp(eta)
        E = exp(-u)
        return -expm1(-u), E, u * E, (u - u^2) * E, (u - 3u^2 + u^3) * E,
            (u - 7u^2 + 6u^3 - u^4) * E
    elseif link === :inverse
        m = 1 / eta
        return m, 1 - m, -m^2, 2m^3, -6m^4, 24m^5
    end
    # :sqrt
    return eta^2, 1 - eta^2, 2eta, T(2), zero(T), zero(T)
end

function _glm_link(link::Symbol, mu::T) where {T<:AbstractFloat}
    link === :identity && return mu
    link === :log && return log(mu)
    link === :logit && return log(mu / (1 - mu))
    link === :probit && return -sqrt(T(2)) * SpecialFunctions.erfcinv(2mu)
    link === :cloglog && return log(-log1p(-mu))
    link === :inverse && return 1 / mu
    sqrt(mu)
end

# Valid linear predictors and means (R's valideta and validmu).
function _glm_valid(f::_GLMFamily, eta, mu, mc)
    isfinite(eta) && isfinite(mu) || return false
    f.link === :inverse && eta == 0 && return false
    f.link === :sqrt && eta <= 0 && return false
    f.family === :poisson ? mu > 0 : f.family === :binomial ? mu > 0 && mc > 0 : true
end

# Variance function V(μ) and its first two derivatives (V''' = 0 here).
function _glm_variance(f::_GLMFamily, mu::T, mc::T) where {T}
    f.family === :gaussian && return one(T), zero(T), zero(T)
    f.family === :poisson && return mu, one(T), zero(T)
    mu * mc, mc - mu, -T(2)
end

function _glm_deviance(f::_GLMFamily, y::T, w::T, mu::T, mc::T) where {T}
    f.family === :gaussian && return w * (y - mu)^2
    xlogy(a, b) = a == 0 ? zero(T) : a * log(a / b)
    f.family === :poisson && return 2w * (xlogy(y, mu) - (y - mu))
    2w * (xlogy(y, mu) + xlogy(1 - y, mc))
end

# Deviance contribution d(η) of one observation, its derivative, the Newton
# weight d''/2 (mgcv's alpha-weighted w) with two derivatives, and the Fisher
# weight w h'²/V with two derivatives, all in η.
function _glm_observation(f::_GLMFamily, y::T, w::T, eta::T) where {T}
    mu, mc, h1, h2, h3, h4 = _glm_inverse_link(f.link, eta)
    valid = _glm_valid(f, eta, mu, mc)
    V, V1, V2 = _glm_variance(f, mu, mc)
    r = y - mu
    d0 = valid ? _glm_deviance(f, y, w, mu, mc) : T(NaN)
    f0 = -2w * r / V
    f1 = 2w * (1 / V + r * V1 / V^2)
    f2 = 2w * (-2V1 / V^2 + r * V2 / V^2 - 2r * V1^2 / V^3)
    f3 = 2w * (-3V2 / V^2 + 6V1^2 / V^3 - 6r * V1 * V2 / V^3 + 6r * V1^3 / V^4)
    if _glm_canonical(f)
        # h' = V(μ): the Newton and Fisher weights are w h'(η), exactly.
        wn, wn1, wn2 = w * h1, w * h2, w * h3
        return (; mu, dev=d0, d1=-2w * r, wn, wn1, wn2, wf=wn, wf1=wn1, wf2=wn2,
            pearson=w * r^2 / V, valid)
    end
    d1 = f0 * h1
    wn = (f1 * h1^2 + f0 * h2) / 2
    wn1 = (f2 * h1^3 + 3 * f1 * h1 * h2 + f0 * h3) / 2
    wn2 = (f3 * h1^4 + 6 * f2 * h1^2 * h2 + f1 * (3h2^2 + 4h1 * h3) + f0 * h4) / 2
    wf = w * h1^2 / V
    wf1 = w * (2h1 * h2 / V - h1^3 * V1 / V^2)
    wf2 = w * (2h2^2 / V + 2h1 * h3 / V - 5h1^2 * h2 * V1 / V^2 - h1^4 * V2 / V^2 +
        2h1^4 * V1^2 / V^3)
    (; mu, dev=d0, d1, wn, wn1, wn2, wf, wf1, wf2, pearson=w * r^2 / V, valid)
end

# Saturated log-likelihood (mgcv fix.family.ls); for the Gaussian family the
# scale is profiled and only Σ log wᵢ enters.
function _glm_saturated(f::_GLMFamily, y::Vector{T}, w::Vector{T}) where {T}
    f.family === :gaussian && return sum(log, w)
    lgamma1(x) = SpecialFunctions.logabsgamma(x + 1)[1]
    if f.family === :poisson
        return sum(w[i] * ((y[i] == 0 ? zero(T) : y[i] * log(y[i])) - y[i] - lgamma1(y[i]))
            for i in eachindex(y))
    end
    total = zero(T)
    for i in eachindex(y)
        m = round(w[i])
        k = round(w[i] * y[i])
        total += lgamma1(m) - lgamma1(k) - lgamma1(m - k) +
            (k == 0 ? zero(T) : k * log(y[i])) + (m - k == 0 ? zero(T) : (m - k) * log1p(-y[i]))
    end
    total
end

# Response checks: R's family initialize plus the integer data the saturated
# likelihood needs (R would round or return -Inf silently).
function _glm_check_response(f::_GLMFamily, y::AbstractVector{T}, w::AbstractVector{T}) where {T}
    integral(x) = abs(x - round(x)) <= sqrt(eps(T)) * max(one(T), abs(x))
    if f.family === :binomial
        all(v -> 0 <= v <= 1, y) || throw(ArgumentError("binomial responses must lie in [0, 1]"))
        all(integral, w) && all(i -> integral(w[i] * y[i]), eachindex(y)) || throw(ArgumentError(
            "binomial responses need integer trial counts (weights) and integer successes weights .* y"))
    elseif f.family === :poisson
        all(v -> v >= 0 && integral(v), y) || throw(ArgumentError("poisson responses must be nonnegative integers"))
    elseif f.link === :log
        all(>(0), y) || throw(ArgumentError(
            "gaussian responses with link=:log must be positive for the starting values mgcv and glm use"))
    elseif f.link === :inverse
        all(!iszero, y) || throw(ArgumentError(
            "gaussian responses with link=:inverse must be nonzero for the starting values mgcv and glm use"))
    end
    nothing
end

# R family initialize: starting means.
function _glm_mustart(f::_GLMFamily, y::T, w::T) where {T}
    f.family === :binomial && return (w * y + T(0.5)) / (w + 1)
    f.family === :poisson && return y + T(0.1)
    y
end

function _glm_response_values(f::_GLMFamily, eta::AbstractArray{T}) where {T}
    map(e -> first(_glm_inverse_link(f.link, e)), eta)
end

# Deviance of the intercept-only model at the weighted mean (R's nulldev).
function _glm_null_deviance(f::_GLMFamily, y::Vector{T}, w::Vector{T}) where {T}
    m = sum(w .* y) / sum(w)
    sum(_glm_deviance(f, y[i], w[i], m, 1 - m) for i in eachindex(y))
end

# Map link-scale intervals to the response scale: the estimate and endpoints go
# through the (monotone) inverse link; `se` stays on the link scale.
function _glm_scale_intervals(f::_GLMFamily, out, type::Symbol)
    type === :link && return out
    a, b = _glm_response_values(f, out.lower), _glm_response_values(f, out.upper)
    merge(out, (; estimate=_glm_response_values(f, out.estimate), lower=min.(a, b), upper=max.(a, b),
        settings=merge(out.settings, (; scale=:response))))
end
