# --- Shared robust statistics -------------------------------------------------

# R's median, mad (constant 1.4826) and quantile(type = 7) of an unsorted vector.
function _r_median(x)
    s = sort(x)
    n = length(s)
    isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end
_r_mad(x, center=_r_median(x)) = 1.4826 * _r_median(abs.(x .- center))
_r_quantile(x, p) = _pfr_quantile(sort(x), p)

_normal_cdf(x) = (1 + SpecialFunctions.erf(x / sqrt(2))) / 2
_normal_quantile(p) = sqrt(2) * SpecialFunctions.erfinv(2p - 1)
_chisq_cdf(x, k) = first(SpecialFunctions.gamma_inc(k / 2, x / 2))
_chisq_quantile(p, k) = 2 * SpecialFunctions.gamma_inc_inv(k / 2, p, 1 - p)
function _f_quantile(p, d1, d2)
    x = first(SpecialFunctions.beta_inc_inv(d1 / 2, d2 / 2, p))
    d2 * x / (d1 * (1 - x))
end

function _outlier_factor(factor)
    factor isa Real && !(factor isa Bool) && isfinite(factor) && factor >= 0 ||
        throw(ArgumentError("factor must be a finite nonnegative number"))
    Float64(factor)
end

function _outlier_proportion(p, name)
    p isa Real && !(p isa Bool) && 0 < p < 1 || throw(ArgumentError("$name must lie in (0, 1)"))
    Float64(p)
end

_outlier_components(prepared) = prepared isa Vector ? prepared : [prepared]

function _outlier_univariate(fd, name)
    fd isa MultivariateFunctionalData && throw(ArgumentError("$name needs univariate functional data"))
    fd
end

# --- Functional boxplot -------------------------------------------------------

"""
    AdjustedFactor(; ntrials=20, trial_size=nothing, rate=2Φ(4Φ⁻¹(1/4)), rng=Random.default_rng())

The adjusted fence factor of Sun and Genton (2012) for [`functional_boxplot`](@ref),
following roahd 1.4.3 `fbplot(adjust=list(...))`. The curves' covariance is
estimated robustly by the orthogonalized Gnanadesikan–Kettenring estimator
(robustbase 0.99-7 `covOGK(sigmamu=s_Qn)`, two iterations, Qn scales and
τ-scale pairwise covariances, as roahd; the Qn scale is the exact order
statistic, which robustbase can truncate to single precision). Each of
`ntrials` trials draws
`trial_size` Gaussian curves (default `8n`, roahd's) with that covariance
around the median curve, and finds the factor at which the boxplot of the
simulated sample flags the proportion `rate` of its curves. The adjusted
factor is the average over trials. The default `rate`,
`2Φ(4Φ⁻¹(1/4)) ≈ 0.007`, is the share of Gaussian observations outside the
classical boxplot fences, as in roahd.

roahd finds each trial's factor with `uniroot` on `[0.5, 5]` with tolerance
`1e-3`. The flagged proportion is a step function of the factor: a simulated
curve is flagged exactly when the factor is below its own threshold
`max_t max((x - U)/(U - L), (L - x)/(U - L))` (central envelope `[L, U]`).
This package therefore takes each trial's factor exactly, as the
`(⌊rate · trial_size⌋ + 1)`-th largest threshold, the smallest factor
flagging at most the proportion `rate`; no bracket is needed. A `rate` at or
above the share of curves outside the central region would give a negative
factor (fences inside the central region); this throws an `ArgumentError`,
where roahd's bracket would make `uniroot` fail. Simulations
use `rng`; the trial factors are stored on the result. Univariate data only,
as in roahd.
"""
struct AdjustedFactor{R<:AbstractRNG}
    ntrials::Int
    trial_size::Union{Nothing,Int}
    rate::Float64
    rng::R
    function AdjustedFactor(; ntrials::Integer=20, trial_size::Union{Nothing,Integer}=nothing,
            rate::Real=2 * _normal_cdf(4 * _normal_quantile(0.25)), rng::AbstractRNG=Random.default_rng())
        ntrials >= 1 || throw(ArgumentError("ntrials must be a positive integer"))
        trial_size === nothing || trial_size >= 3 || throw(ArgumentError("trial_size must be at least 3"))
        0 < rate < 1 || throw(ArgumentError("rate must lie in (0, 1)"))
        new{typeof(rng)}(Int(ntrials), trial_size === nothing ? nothing : Int(trial_size), Float64(rate), rng)
    end
end

Base.show(io::IO, a::AdjustedFactor) = print(io, "AdjustedFactor(ntrials=", a.ntrials,
    ", trial_size=", a.trial_size === nothing ? "8n" : a.trial_size, ", rate=", a.rate, ")")

"""
    FunctionalBoxplot

Result of [`functional_boxplot`](@ref). `depth` is the [`FunctionalDepth`](@ref)
that orders the curves, `central` the indices of the curves forming the
central region (deepest first), `outliers` the sorted indices of the curves
outside the fences, `median` the deepest curve and `factor` the fence factor
used. `regions` holds one named tuple per component (one for univariate data)
with `grid`, the curves `values` on it, the central envelope `central_lower`
and `central_upper`, the fences `fence_lower` and `fence_upper`, and the
whiskers `whisker_lower` and `whisker_upper` (the envelope of the curves not
flagged). `settings` records the central proportion, the tie rule and how the
factor was chosen.
"""
struct FunctionalBoxplot{T<:AbstractFloat,D<:FunctionalDepth,R,S<:NamedTuple}
    depth::D
    central::Vector{Int}
    outliers::Vector{Int}
    median::Int
    factor::T
    regions::Vector{R}
    settings::S
end

nobs(b::FunctionalBoxplot) = nobs(b.depth)

function Base.show(io::IO, b::FunctionalBoxplot)
    print(io, "FunctionalBoxplot(n=", nobs(b), ", method=", b.depth.method, ", factor=", b.factor,
        ", central=", b.settings.central, " (", length(b.central), " curves, ties=:", b.settings.central_ties,
        "), outliers=", b.outliers, ", median=", b.median, ", factor_rule=:", b.settings.factor_rule, ")")
end
Base.show(io::IO, ::MIME"text/plain", b::FunctionalBoxplot) = show(io, b)

function _boxplot_ties(ties)
    ties in (:first, :all) || throw(ArgumentError("central_ties must be :first or :all"))
    ties
end

# Indices of the central region, deepest first.
function _boxplot_central(values, ordering, central, ties)
    ties === :first && return ordering[1:ceil(Int, central * length(values))]
    threshold = _r_quantile(values, 1 - central)
    filter(i -> values[i] >= threshold, ordering)
end

function _boxplot_envelope(x, rows)
    lower = vec(minimum(view(x, rows, :); dims=1))
    upper = vec(maximum(view(x, rows, :); dims=1))
    lower, upper
end

_boxplot_outside(x, lower, upper) =
    [any(t -> x[i, t] > upper[t] || x[i, t] < lower[t], axes(x, 2)) for i in axes(x, 1)]

# Smallest factor at which the boxplot of a sample flags at most the share `rate`.
function _boxplot_trial_factor(x::Matrix{T}, values, ordering, central, ties, rate) where {T}
    lower, upper = _boxplot_envelope(x, _boxplot_central(values, ordering, central, ties))
    thresholds = map(axes(x, 1)) do i
        maximum(axes(x, 2)) do t
            range = upper[t] - lower[t]
            range > 0 && return max((x[i, t] - upper[t]) / range, (lower[t] - x[i, t]) / range)
            x[i, t] == upper[t] ? T(-Inf) : T(Inf)
        end
    end
    factor = sort(thresholds; rev=true)[floor(Int, rate * size(x, 1)) + 1]
    isfinite(factor) || throw(ArgumentError(
        "the adjusted factor is undefined: simulated curves leave a degenerate central envelope"))
    factor >= 0 || throw(ArgumentError(string("the adjusted factor is negative (", factor, "): rate ", rate,
        " exceeds the share of simulated curves outside the central region; choose a smaller rate")))
    factor
end

function _boxplot_adjusted(adjust::AdjustedFactor, c, d, method, central, ties)
    x = c.target
    n, m = size(x)
    m >= 2 || throw(ArgumentError("the adjusted factor needs at least two grid points"))
    size_trial = adjust.trial_size === nothing ? 8n : adjust.trial_size
    covariance = _ogk_covariance(x)
    factor = cholesky(Symmetric(covariance); check=false)
    issuccess(factor) || throw(ArgumentError("the robust covariance of the curves is not positive definite"))
    center = x[d.median, :]
    trials = map(1:adjust.ntrials) do _
        sim = randn(adjust.rng, eltype(x), size_trial, m) * factor.U .+ transpose(center)
        values = first(_depth_values(method, (; target=sim, reference=sim, grid=c.grid), true))
        _boxplot_trial_factor(sim, values, sortperm(values; rev=true), central, ties, adjust.rate)
    end
    (sum(trials) / length(trials), (; trial_factors=trials, trial_size=size_trial, rate=adjust.rate,
        ntrials=adjust.ntrials, covariance=:ogk_qn))
end

"""
    functional_boxplot(fd; method=ModifiedBandDepth(), factor=1.5, central=0.5,
                       central_ties=:first, weights=nothing, grid=nothing, fpca=(;))

Functional boxplot (Sun and Genton, 2011). Curves are ordered by the depth
`method` (computed by [`depth`](@ref), with `weights`, `grid` and `fpca` as
there); the central region is the band spanned by the deepest proportion
`central` of the curves, and the fences inflate it by `factor` times its
pointwise range on each side. Curves crossing a fence anywhere are flagged
as magnitude outliers. Returns a [`FunctionalBoxplot`](@ref).

Defaults: depth MBD (fda 6.3.0 `fbplot` and roahd 1.4.3 `fbplot`), here with
the package's exact-tie, trapezoidal [`ModifiedBandDepth`](@ref); factor 1.5
and central proportion 0.5 (both references). `central_ties=:first` takes
exactly `⌈n · central⌉` curves, ties in depth broken by curve order, as the
definition and fda's `fbplot`; `:all` takes every curve whose depth is at
least the `1 - central` quantile, as roahd (which then may include more
curves). A curve is flagged when it lies strictly outside a fence, as roahd;
fda also flags curves touching a fence, which matters only on exact equality.
The median is the first deepest curve (roahd); fda averages tied deepest
curves.

roahd is reproduced by `method=ModifiedBandDepth(integration=:uniform)`
with `central_ties=:all`, fda by `method=ModifiedBandDepth(ties=:average,
integration=:uniform)`. `factor=AdjustedFactor()` selects the adjusted
factor of Sun and Genton (2012); see [`AdjustedFactor`](@ref).

Multivariate data use the multivariate depth (for MBD, roahd's `multiMBD`
weighted average) to order curves, one envelope and pair of fences per
component, and flag a curve outside the fences of any component, as roahd's
`fbplot` for `mfData`.
"""
function functional_boxplot(fd; method::_DepthMethod=ModifiedBandDepth(), factor=1.5, central::Real=0.5,
        central_ties::Symbol=:first, weights=nothing, grid=nothing, fpca::NamedTuple=(;))
    central = _outlier_proportion(central, "central")
    _boxplot_ties(central_ties)
    factor isa AdjustedFactor || _outlier_factor(factor)
    prepared = _depth_prepare(fd, nothing, grid, fpca, true)
    d = _depth_result(method, prepared, true, ids(fd), weights)
    components = _outlier_components(prepared)
    T = eltype(d.values)
    rule, details = if factor isa AdjustedFactor
        prepared isa Vector && throw(ArgumentError("the adjusted factor applies only to univariate data"))
        value, info = _boxplot_adjusted(factor, prepared, d, method, central, central_ties)
        (:adjusted, merge((; adjust=factor), info))
    else
        value = _outlier_factor(factor)
        (:fixed, (;))
    end
    f = T(value)
    rows = _boxplot_central(d.values, d.ordering, central, central_ties)
    flagged = falses(nobs(d))
    envelopes = map(components) do c
        lower, upper = _boxplot_envelope(c.target, rows)
        spread = upper .- lower
        fence_lower, fence_upper = lower .- f .* spread, upper .+ f .* spread
        flagged .|= _boxplot_outside(c.target, fence_lower, fence_upper)
        (; lower, upper, fence_lower, fence_upper)
    end
    outliers = findall(flagged)
    kept = findall(!, flagged)
    regions = map(components, envelopes) do c, e
        whisker_lower, whisker_upper = _boxplot_envelope(c.target, kept)
        (; grid=copy(c.grid), values=copy(c.target), central_lower=e.lower, central_upper=e.upper,
            fence_lower=e.fence_lower, fence_upper=e.fence_upper, whisker_lower, whisker_upper)
    end
    settings = merge((; central, central_ties, factor_rule=rule, comparison=:strict), details)
    FunctionalBoxplot(d, rows, outliers, d.median, f, regions, settings)
end

# --- Robust covariance for the adjusted factor (robustbase 0.99-7) ------------------

# Qn scale (Rousseeuw and Croux, 1993) with robustbase's constant and
# finite-sample corrections.
function _qn_scale(x::AbstractVector{T}) where {T}
    n = length(x)
    n == 1 && return zero(T)
    h = n ÷ 2 + 1
    k = h * (h - 1) ÷ 2
    differences = [abs(x[i] - x[j]) for i in 1:n-1 for j in i+1:n]
    r = T(2.21914) * partialsort!(differences, k)
    if n <= 12
        return r * T((0.399356, 0.99365, 0.51321, 0.84401, 0.6122, 0.85877, 0.66993, 0.87344,
            0.72014, 0.88906, 0.75743)[n - 1])
    end
    correction = isodd(n) ? 1.60188 + (-2.1284 - 5.172 / n) / n : 3.67561 + (1.9654 + (6.987 - 77 / n) / n) / n
    r / T(correction / n + 1)
end

# robustbase scaleTau2 with its defaults (c1 = 4.5, c2 = 3, one iteration).
function _tau_scale(x::AbstractVector{T}) where {T}
    c1, c2 = 4.5, 3.0
    mu0 = _r_median(x)
    deviations = abs.(x .- mu0)
    sigma0 = _r_median(deviations)
    sigma0 <= 0 && return zero(T)
    b = c2 * _normal_quantile(0.75)
    expected = 2 * ((1 - b^2) * _normal_cdf(b) - b * exp(-b^2 / 2) / sqrt(2pi) + b^2) - 1
    w = max.(0, 1 .- (deviations ./ (sigma0 * c1)) .^ 2) .^ 2
    mu = sum(x .* w) / sum(w)
    rho = min.(((x .- mu) ./ sigma0) .^ 2, c2^2)
    T(sigma0 * sqrt(sum(rho) / (length(x) * expected)))
end

_gk_covariance(x, y) = (_tau_scale(x .+ y)^2 - _tau_scale(x .- y)^2) / 4

# robustbase covOGK(X, n.iter = 2, sigmamu = s_Qn)$cov: the raw OGK estimate.
function _ogk_covariance(x::Matrix{T}) where {T}
    p = size(x, 2)
    z = copy(x)
    transforms = Matrix{T}[]
    for _ in 1:2
        d = [_qn_scale(view(z, :, j)) for j in 1:p]
        all(>(0), d) || throw(ArgumentError("a grid point has zero Qn scale; the robust covariance is undefined"))
        z ./= transpose(d)
        u = Matrix{T}(I, p, p)
        for i in 2:p, j in 1:i-1
            u[i, j] = u[j, i] = _gk_covariance(view(z, :, i), view(z, :, j))
        end
        vectors = eigen(Symmetric(u)).vectors
        push!(transforms, d .* vectors)
        z = z * vectors
    end
    covariance = Diagonal([_qn_scale(view(z, :, j))^2 for j in 1:p])
    covariance = transforms[2] * covariance * transpose(transforms[2])
    transforms[1] * covariance * transpose(transforms[1])
end

# --- Outliergram --------------------------------------------------------------

"""
    Outliergram

Result of [`outliergram`](@ref). `mei` and `mbd` are the modified epigraph
index and modified band depth of every curve; `distance` is
`a0 + a1·MEI + a2·MEI² - MBD`, the vertical distance below the parabola with
`coefficients = (a0, a1, a2) = (-2/(n(n-1)), 2(n+1)/(n-1), -2n/(n-1))`.
`threshold` is `Q3 + factor · (Q3 - Q1)` of the distances, with
`quartiles = (Q1, Q3)`. `outliers` lists the shape outliers (sorted):
`flagged` those with distance at least the threshold, `shifted` those found
by the shift check. `grid`, `values`, `representation` and `fpca` describe
the curves used; `settings` records every choice.
"""
struct Outliergram{T<:AbstractFloat,M<:ModifiedBandDepth,F,S<:NamedTuple}
    mei::Vector{T}
    mbd::Vector{T}
    distance::Vector{T}
    outliers::Vector{Int}
    flagged::Vector{Int}
    shifted::Vector{Int}
    quartiles::NTuple{2,T}
    threshold::T
    coefficients::NTuple{3,T}
    factor::T
    method::M
    grid::Vector{T}
    values::Matrix{T}
    representation::Symbol
    fpca::F
    settings::S
end

nobs(o::Outliergram) = length(o.mei)

function Base.show(io::IO, o::Outliergram)
    print(io, "Outliergram(n=", nobs(o), ", method=", o.method, ", factor=", o.factor,
        ", p_check=", o.settings.p_check, ", shift=", o.settings.shift, ", threshold=", o.threshold,
        ", outliers=", o.outliers, ", representation=:", o.representation, ")")
end
Base.show(io::IO, ::MIME"text/plain", o::Outliergram) = show(io, o)

_outliergram_parabola(coefficients, mei) = coefficients[1] + coefficients[2] * mei + coefficients[3] * mei^2

function _outliergram_point(x, row, i, w, method)
    y = copy(x)
    y[i, :] = row
    target = reshape(row, 1, :)
    (only(_graph_part(target, y, w, :epigraph)), only(_mbd(target, y, w, method.ties, true)))
end

# roahd's shift check: a candidate attaining the pointwise minimum (or maximum)
# somewhere is moved up (down) by the smallest gap to the second curve there
# and tested again within the sample.
function _outliergram_shift(x, candidates, w, method, coefficients, threshold, side)
    n, m = size(x)
    gaps = fill(side === :low ? typemax(eltype(x)) : typemin(eltype(x)), n)
    attains = falses(n)
    for t in 1:m
        column = view(x, :, t)
        k = side === :low ? argmin(column) : argmax(column)
        others = column[[1:k-1; k+1:n]]
        gap = side === :low ? minimum(others) - column[k] : maximum(others) - column[k]
        attains[k] = true
        gaps[k] = side === :low ? min(gaps[k], gap) : max(gaps[k], gap)
    end
    filter(candidates) do i
        attains[i] || return false
        mei, mbd = _outliergram_point(x, x[i, :] .+ gaps[i], i, w, method)
        _outliergram_parabola(coefficients, mei) - mbd >= threshold
    end
end

"""
    outliergram(fd; method=ModifiedBandDepth(), factor=1.5, p_check=0.05, shift=true,
                grid=nothing, fpca=(;))

Outliergram (Arribas-Gil and Romo, 2014): the modified epigraph index against
the modified band depth. For a curve that does not cross the others the point
`(MEI, MBD)` lies on the parabola `a0 + a1·MEI + a2·MEI²`
(`a0 = -2/(n(n-1))`, `a1 = 2(n+1)/(n-1)`, `a2 = n²a0`); shape outliers lie far
below it. Curves whose distance below the parabola is at least
`Q3 + factor · IQR` of the distances (R quantile type 7) are flagged.

With `shift=true`, curves not flagged whose MEI is at most the `p_check`
quantile (curves high in the sample) or at least the `1 - p_check` quantile
(low curves) are checked again after a vertical shift: a curve that is the
pointwise maximum (minimum) somewhere is moved down (up) by the smallest gap
to the second-largest (second-smallest) value at those points, its MEI and
MBD recomputed within the sample, and it is flagged if its distance then
reaches the threshold. Returns an [`Outliergram`](@ref).

Defaults follow roahd 1.4.3 `outliergram`: `factor=1.5`, `p_check=0.05`,
shift check on. `method` is the [`ModifiedBandDepth`](@ref) and its
`integration` also weights the MEI; the package default is exact ties and
trapezoidal weights. roahd's `outliergram` uses `MBD` with average ranks and
equal weights, which `method=ModifiedBandDepth(ties=:average,
integration=:uniform)` reproduces. Two departures from roahd, documented on
the "Depth and outliers" page: roahd skips the shift check when the first
stage flags no curve, and matches shift candidates with `grep`, which
confuses indices such as 1 and 12; this package checks every candidate.

Univariate data only. Curves reach a common grid as in [`depth`](@ref).
"""
function outliergram(fd; method::ModifiedBandDepth=ModifiedBandDepth(), factor::Real=1.5,
        p_check::Real=0.05, shift::Bool=true, grid=nothing, fpca::NamedTuple=(;))
    f = _outlier_factor(factor)
    p_check = _outlier_proportion(p_check, "p_check")
    c = _depth_prepare(_outlier_univariate(fd, "outliergram"), nothing, grid, fpca, true)
    x = c.target
    n = size(x, 1)
    n >= 3 || throw(ArgumentError("outliergram needs at least three curves"))
    T = eltype(x)
    w = _depth_weights(c.grid, method.integration)
    mei = _graph_part(x, x, w, :epigraph)
    mbd = _mbd(x, x, w, method.ties, true)
    a0 = T(-2 / (n * (n - 1)))
    coefficients = (a0, T(2 * (n + 1) / (n - 1)), T(n^2) * a0)
    distance = _outliergram_parabola.(Ref(coefficients), mei) .- mbd
    q1, q3 = _r_quantile(distance, 0.25), _r_quantile(distance, 0.75)
    threshold = q3 + T(f) * (q3 - q1)
    flagged = findall(>=(threshold), distance)
    shifted = Int[]
    if shift
        kept = setdiff(1:n, flagged)
        low, high = _r_quantile(mei, p_check), _r_quantile(mei, 1 - p_check)
        shifted = sort!(vcat(
            _outliergram_shift(x, filter(i -> mei[i] >= high, kept), w, method, coefficients, threshold, :low),
            _outliergram_shift(x, filter(i -> mei[i] <= low, kept), w, method, coefficients, threshold, :high)))
    end
    outliers = sort!(vcat(flagged, shifted))
    settings = (; p_check, shift, integration=method.integration, ties=method.ties,
        quantile_type=7, grid=copy(c.grid))
    Outliergram(mei, mbd, distance, outliers, flagged, shifted, (q1, q3), threshold, coefficients, T(f),
        method, copy(c.grid), copy(x), c.representation, c.fpca, settings)
end

# --- Magnitude-shape plot -----------------------------------------------------

"""
    MagnitudeShape

Result of [`magnitude_shape`](@ref). `outlyingness` holds the directional
outlyingness `O_i(t)` (`n × m × p`, one slice per component),
`mean_outlyingness` the mean directional outlyingness `MO` (`n × p`) and
`variation_outlyingness` the variation `VO`. `distance` is the squared robust
Mahalanobis distance of `(MO, VO)` from the minimum covariance determinant
estimate `center` and `covariance` (raw or reweighted, as
`settings.estimate` records); curves with distance above `cutoff` are
`outliers` (sorted), and `median` is the curve of smallest distance. `grid`,
`representation`, `fpca` and `settings` record the curves and choices used,
including the projection directions for multivariate data and the MCD
subset.
"""
struct MagnitudeShape{T<:AbstractFloat,R,F,S<:NamedTuple}
    outlyingness::Array{T,3}
    mean_outlyingness::Matrix{T}
    variation_outlyingness::Vector{T}
    distance::Vector{T}
    center::Vector{T}
    covariance::Matrix{T}
    cutoff::T
    outliers::Vector{Int}
    median::Int
    grid::Vector{T}
    representation::R
    fpca::F
    settings::S
end

nobs(s::MagnitudeShape) = length(s.distance)

function Base.show(io::IO, s::MagnitudeShape)
    print(io, "MagnitudeShape(n=", nobs(s), ", components=", size(s.mean_outlyingness, 2),
        ", definition=:", s.settings.definition, ", integration=:", s.settings.integration,
        ", cutoff=", s.cutoff, " (", s.settings.cutoff_rule, "), outliers=", s.outliers,
        ", median=", s.median, ", estimate=:", s.settings.estimate, ", mcd_subsets=", s.settings.mcd.nsamp,
        s.settings.mcd.exact ? " (all)" : " (sampled)", ")")
end
Base.show(io::IO, ::MIME"text/plain", s::MagnitudeShape) = show(io, s)

# Upper median and unscaled MAD, as fdaoutlier's C++ customMedian/customMAD.
_fdaoutlier_median(x) = partialsort(x, length(x) ÷ 2 + 1)

function _ms_directions(directions, nproj, p, rng)
    if directions === nothing
        nproj isa Integer && nproj >= 1 || throw(ArgumentError("nproj must be a positive integer"))
        u = randn(rng, Int(nproj), p)
    else
        directions isa AbstractMatrix{<:Real} || throw(ArgumentError("directions must be a real matrix"))
        Base.require_one_based_indexing(directions)
        size(directions, 2) == p || throw(DimensionMismatch(
            "directions need one column per component ($p)"))
        nproj === nothing || nproj == size(directions, 1) || throw(DimensionMismatch(
            "nproj ($nproj) differs from the number of supplied directions ($(size(directions, 1)))"))
        u = Matrix{Float64}(directions)
        size(u, 1) >= 1 && all(isfinite, u) || throw(ArgumentError("directions must be finite, with at least one row"))
    end
    norms = sqrt.(vec(sum(abs2, u; dims=2)))
    all(>(0), norms) || throw(ArgumentError("every direction must be nonzero"))
    u ./ norms
end

# Projection outlyingness of the rows of z (n × p) at one grid point.
function _ms_projection(z::Matrix{T}, u, definition) where {T}
    n = size(z, 1)
    out = fill(T(-Inf), n)
    for k in axes(u, 1)
        projection = z * view(u, k, :)
        if definition === :dai_genton
            center = _r_median(projection)
            scale = _r_mad(projection, center)
        else
            center = _fdaoutlier_median(projection)
            scale = _fdaoutlier_median(abs.(projection .- center))
        end
        scale > 0 || throw(ArgumentError(
            "a projection has zero median absolute deviation at some grid point; outlyingness is undefined"))
        for i in 1:n
            v = (projection[i] - center) / scale
            out[i] = max(out[i], definition === :dai_genton ? abs(v) : v)
        end
    end
    out
end

function _ms_outlyingness(curves::Vector{Matrix{T}}, u, definition) where {T}
    n, m = size(first(curves))
    p = length(curves)
    o = zeros(T, n, m, p)
    for t in 1:m
        if p == 1
            column = curves[1][:, t]
            center = _r_median(column)
            scale = _r_mad(column, center)
            scale > 0 || throw(ArgumentError(
                "the curves have zero median absolute deviation at grid point $t; outlyingness is undefined"))
            o[:, t, 1] .= (column .- center) ./ scale
            continue
        end
        z = reduce(hcat, [c[:, t] for c in curves])
        out = _ms_projection(z, u, definition)
        if definition === :fdaoutlier
            out = 1 ./ (1 ./ (1 .+ out)) .- 1
        end
        deviations = z .- transpose(z[argmin(out), :])
        lengths = sqrt.(vec(sum(abs2, deviations; dims=2)))
        for i in 1:n
            lengths[i] > 0 && (o[i, t, :] .= out[i] .* view(deviations, i, :) ./ lengths[i])
        end
    end
    o
end

# Squared Mahalanobis distances of the rows of x.
function _mahalanobis(x::Matrix{T}, center, covariance) where {T}
    factor = cholesky(Symmetric(covariance); check=false)
    issuccess(factor) || throw(ArgumentError("the robust covariance is singular"))
    vec(sum(abs2, (x .- transpose(center)) / factor.U; dims=2))
end

_column_means(x) = vec(sum(x; dims=1)) ./ size(x, 1)
function _sample_covariance(x)
    centered = x .- transpose(_column_means(x))
    transpose(centered) * centered ./ (size(x, 1) - 1)
end

# MASS do_one: subset mean and QR; returns (log|det R|, squared distances) or nothing if singular.
function _mcd_subset(x::Matrix{T}, rows) where {T}
    xr = x[rows, :]
    means = _column_means(xr)
    xr .-= transpose(means)
    norms = sqrt.(vec(sum(abs2, xr; dims=1)))
    r = qr(xr).R
    all(k -> abs(r[k, k]) >= 1e-7 * norms[k] && norms[k] > 0, axes(r, 1)) || return nothing
    det = sum(k -> log(abs(r[k, k])), axes(r, 1))
    distances = (length(rows) - 1) .* vec(sum(abs2, (x .- transpose(means)) / UpperTriangular(r); dims=2))
    (det, distances)
end

# MASS next_set: the next subset in lexicographic order (zero-based values).
function _mcd_next!(x, n, k)
    j = k
    tmp = x[j]
    x[j] += 1
    while j > 1 && x[j] >= n - (k - j)
        j -= 1
        x[j] += 1
        tmp = x[j]
    end
    for i in j+1:k
        tmp += 1
        x[i] = tmp
    end
    x
end

function _mcd_nsamp(nsamp, n, ps)
    exact = binomial(big(n), ps)
    if nsamp === :best
        return exact < 5000 ? (true, Int(exact)) : (false, min(500ps, 3000))
    elseif nsamp === :exact
        exact <= 10^7 || throw(ArgumentError("nsamp=:exact would try $exact subsets; give an integer"))
        return (true, Int(exact))
    end
    nsamp isa Integer && nsamp >= 1 || throw(ArgumentError("nsamp must be :best, :exact or a positive integer"))
    nsamp >= exact ? (true, Int(exact)) : (false, Int(nsamp))
end

# MASS 7.3-65 cov.rob(x, method = "mcd"): subsets of size p + 1 with up to
# four concentration steps each, then the reweighted estimate.
function _mass_mcd(x::Matrix{T}, nsamp, rng) where {T}
    n, p = size(x)
    quan = (n + p + 1) ÷ 2
    quan <= n - 1 || throw(ArgumentError("the MCD estimate needs at least $(p + 3) curves"))
    divisor = [_r_quantile(view(x, :, j), 0.75) - _r_quantile(view(x, :, j), 0.25) for j in 1:p]
    all(>(0), divisor) || throw(ArgumentError(
        "a magnitude-shape coordinate has zero interquartile range; the MCD estimate is undefined"))
    xs = x ./ transpose(divisor)
    ps = p + 1
    exact, trials = _mcd_nsamp(nsamp, n, ps)
    which = collect(0:ps-1)
    best = T(Inf)
    bestone = falses(n)
    singular = 0
    for trial in 1:trials
        if exact
            trial > 1 && _mcd_next!(which, n, ps)
        else
            which = randperm(rng, n)[1:ps] .- 1
        end
        subset = _mcd_subset(xs, which .+ 1)
        subset === nothing && (singular += 1; continue)
        det, d2 = subset
        lim = partialsort(d2, quan)
        crit = zero(T)
        for iter in 0:3
            iter > 0 && (lim = partialsort(d2, quan))
            rows = findall(<=(lim), d2)[1:quan]
            step = _mcd_subset(xs, rows)
            step === nothing || ((det, d2) = step)
            iter > 0 && 2det >= 0.999 * crit && break
            crit = 2det
        end
        if crit < best
            best = crit
            bestone = d2 .<= lim
        end
    end
    count(bestone) > p || throw(ArgumentError("the magnitude-shape coordinates are probably collinear"))
    chosen = xs[bestone, :]
    means = _column_means(chosen)
    raw = _sample_covariance(chosen)
    dist = _mahalanobis(xs, means, raw .* (1 + 15 / (n - p))^2)
    cut = _chisq_quantile(0.975, p) * _r_quantile(dist, quan / n) / _chisq_quantile(quan / n, p)
    kept = xs[dist .< cut, :]
    size(kept, 1) > p || throw(ArgumentError("too few curves remain after MCD reweighting"))
    (; raw=(means .* divisor, divisor .* raw .* transpose(divisor)),
        reweighted=(_column_means(kept) .* divisor, divisor .* _sample_covariance(kept) .* transpose(divisor)),
        subset=findall(bestone), settings=(; quantile_used=quan, exact, nsamp=trials, singular_subsets=singular))
end

# Hardin and Rocke (2005) F approximation for squared robust MCD distances:
# c(m - q + 1)/(q m) d² ~ F(q, m - q + 1), with m from the Croux and Haesbroeck
# asymptotics and Hardin and Rocke's small-sample correction.
function _hardin_rocke(n, q)
    h = (n + q + 1) ÷ 2
    alpha = (n - h) / n
    q_alpha = _chisq_quantile(1 - alpha, q)
    c_alpha = (1 - alpha) / _chisq_cdf(q_alpha, q + 2)
    c2 = -_chisq_cdf(q_alpha, q + 2) / 2
    c3 = -_chisq_cdf(q_alpha, q + 4) / 2
    c4 = 3c3
    b1 = c_alpha * (c3 - c4) / (1 - alpha)
    b2 = 0.5 + c_alpha / (1 - alpha) * (c3 - q_alpha * (c2 + (1 - alpha) / 2) / q)
    v1 = (1 - alpha) * b1^2 * (alpha * (c_alpha * q_alpha / q - 1)^2 - 1) -
        2 * c3 * c_alpha^2 * (3 * (b1 - q * b2)^2 + (q + 2) * b2 * (2 * b1 - q * b2))
    v2 = n * (b1 * (b1 - q * b2) * (1 - alpha))^2 * c_alpha^2
    m_asymptotic = 2 / (c_alpha^2 * v1 / v2)
    m = m_asymptotic * exp(0.725 - 0.00663 * q - 0.078 * log(n))
    m < q && (m = m_asymptotic)
    c = _chisq_cdf(_chisq_quantile(h / n, q), q + 2) / (h / n)
    scale = c * (m - q + 1) / (q * m)
    quantile = _f_quantile(0.993, q, m - q + 1)
    (; cutoff=quantile / scale, m, c, scale, quantile)
end

function _ms_definition(definition, integration)
    definition in (:dai_genton, :fdaoutlier) || throw(ArgumentError("definition must be :dai_genton or :fdaoutlier"))
    integration === nothing && return definition === :dai_genton ? :quadrature : :uniform
    _depth_integration(integration)
    definition === :fdaoutlier && integration !== :uniform &&
        throw(ArgumentError("definition=:fdaoutlier uses equal grid weights; leave integration at nothing"))
    integration
end

"""
    magnitude_shape(fd; definition=:dai_genton, integration=nothing, nproj=nothing,
                    directions=nothing, rng=Random.default_rng(), cutoff=:hardin_rocke,
                    nsamp=:best, grid=nothing, fpca=(;))

Magnitude–shape plot and its outlier rule (Dai and Genton, 2018), following
fdaoutlier 0.2.1 `msplot`. The directional outlyingness of curve `i` at `t`
is `O_i(t) = SDO(x_i(t)) · v_i(t)`, with the Stahel–Donoho outlyingness `SDO`
(the reciprocal projection depth minus one) of `x_i(t)` among the curves at
`t` and `v_i(t)` the unit vector from the pointwise median to `x_i(t)`. For
univariate data `O_i(t) = (x_i(t) - med(t)) / MAD(t)` (R's `mad`, constant
1.4826). The mean `MO_i = ∫ O_i` and variation `VO_i = ∫ ‖O_i - MO_i‖²` (with
`∫` the normalized grid average) are the magnitude and shape coordinates.
The minimum covariance determinant (MCD) subset of `(MO, VO)` is found by the
search of MASS 7.3-65 `cov.rob(method="mcd", nsamp="best")`, as fdaoutlier;
curves whose squared robust Mahalanobis distance exceeds `cutoff` are
flagged. Returns a [`MagnitudeShape`](@ref).

- `integration=nothing` resolves to `:quadrature` (stored trapezoidal weights,
  the package rule for integrals) for `definition=:dai_genton` and to
  `:uniform` (equal weights, fdaoutlier) for `:fdaoutlier`.
- Multivariate data (`p` components on one grid) use `SDO = max_u |u'x - med| / MAD`
  over projection directions `u`, with the pointwise median the curve value of
  smallest outlyingness, as fdaoutlier. `directions` (`nproj × p`, rows
  normalized here) are used as given; otherwise `nproj` (default 200, as
  fdaoutlier) directions are drawn uniformly on the sphere with `rng`.
  fdaoutlier draws them uniformly in the cube `[-1, 1]^p` and normalizes,
  which favours the cube's diagonals.
- `definition=:dai_genton` (default) measures distances from the raw MCD
  estimate, the mean and sample covariance of the subset, to which the
  Hardin–Rocke cutoff applies (Dai and Genton, 2018).
- `definition=:fdaoutlier` reproduces fdaoutlier's computations exactly:
  distances from MASS's reweighted estimate (`cov.rob`'s `center` and
  `cov`); for multivariate data its one-sided `max_u (u'x - med) / MAD` with
  upper medians and an unscaled MAD; for univariate data `VO` as the sample
  variance over grid points (divisor `m - 1`). Multivariate `VO` divides by
  `m` in both definitions.
- `cutoff=:hardin_rocke` (default) is the Hardin and Rocke (2005)
  approximation `c(m - q + 1)/(q m) d² ~ F(q, m - q + 1)` at level 0.993
  (Dai and Genton), `q = p + 1`, with `m` from the Croux–Haesbroeck
  asymptotics and Hardin and Rocke's small-sample correction, and the exact
  `c = P(χ²_{q+2} < χ²_{q, h/n}) / (h/n)`, `h = ⌊(n + q + 1)/2⌋`. fdaoutlier
  instead uses a table of simulated factors for `n < 1000` and `q ≤ 3` and
  otherwise estimates `c` by Monte Carlo from noncentral χ² draws (passing
  `h/n` as the noncentrality), about twice the exact value; its cutoff is
  therefore random and not reproduced. A positive number gives the cutoff
  for squared distances directly (for example fdaoutlier's).
- `nsamp` follows MASS: `:best` enumerates all `q + 1`-subsets when there
  are fewer than 5000 and otherwise samples `min(500(q + 1), 3000)` of them
  with `rng`; `:exact` or an integer may be given. The settings record which.

Curves reach a common grid as in [`depth`](@ref).
"""
function magnitude_shape(fd; definition::Symbol=:dai_genton, integration::Union{Nothing,Symbol}=nothing,
        nproj::Union{Nothing,Integer}=nothing, directions=nothing, rng::AbstractRNG=Random.default_rng(),
        cutoff=:hardin_rocke, nsamp=:best, grid=nothing, fpca::NamedTuple=(;))
    integration = _ms_definition(definition, integration)
    cutoff === :hardin_rocke || (cutoff isa Real && !(cutoff isa Bool) && isfinite(cutoff) && cutoff > 0) ||
        throw(ArgumentError("cutoff must be :hardin_rocke or a finite positive number"))
    prepared = _depth_prepare(fd, nothing, grid, fpca, true)
    components = _outlier_components(prepared)
    p = length(components)
    grid = p == 1 ? only(components).grid : _depth_common_grid(components, "magnitude_shape")
    curves = [c.target for c in components]
    T = eltype(first(curves))
    n, m = size(first(curves))
    n >= p + 4 || throw(ArgumentError("magnitude_shape needs at least $(p + 4) curves"))
    u = p == 1 ? nothing : _ms_directions(directions, nproj === nothing && directions === nothing ? 200 : nproj, p, rng)
    p == 1 && (directions === nothing && nproj === nothing ||
        throw(ArgumentError("directions apply only to multivariate data")))
    o = _ms_outlyingness(curves, u, definition)
    w = _depth_weights(grid, integration)
    mo = reduce(hcat, [view(o, :, :, j) * w for j in 1:p])
    vo = if definition === :fdaoutlier && p == 1
        vec(sum(abs2, view(o, :, :, 1) .- mo; dims=2)) ./ (m - 1)
    else
        [sum(w[t] * sum(abs2, o[i, t, j] - mo[i, j] for j in 1:p) for t in 1:m) for i in 1:n]
    end
    coordinates = hcat(mo, vo)
    mcd = _mass_mcd(coordinates, nsamp, rng)
    estimate = definition === :dai_genton ? :raw : :reweighted
    center, covariance = getproperty(mcd, estimate)
    distance = _mahalanobis(coordinates, center, covariance)
    rule, value, details = if cutoff === :hardin_rocke
        hr = _hardin_rocke(n, p + 1)
        (:hardin_rocke, hr.cutoff, (; hardin_rocke=hr))
    else
        (:given, Float64(cutoff), (;))
    end
    outliers = findall(>(value), distance)
    settings = merge((; definition, integration, nproj=u === nothing ? nothing : size(u, 1), directions=u,
        generated=directions === nothing && p > 1, cutoff_rule=rule, level=0.993, estimate,
        mcd=merge(mcd.settings, (; subset=mcd.subset))), details)
    MagnitudeShape(o, mo, vo, distance, center, covariance, T(value), outliers, argmin(distance), copy(grid),
        _depth_field(prepared, :representation), _depth_field(prepared, :fpca), settings)
end
