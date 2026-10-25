# Integer-degree chi-square probabilities. The half-integer gamma recurrence
# starts from the public Gaussian-kernel CDF; no additional dependency is needed.
# A positive lower-tail series avoids cancellation near zero.
function _chisq_probability(k::Int, x::Float64, upper::Bool)
    x == 0 && return upper ? 1.0 : 0.0
    z, a = x / 2, k / 2
    loggamma = isodd(k) ? log(sqrt(pi)) : 0.0
    for b in (isodd(k) ? 0.5 : 1.0):1.0:(a - 1)
        loggamma += log(b)
    end
    if z < a + 1
        term = total = 1.0
        for j in 1:10000
            term *= z / (a + j)
            total += term
            if term <= eps(Float64) * total
                p = exp(a * log(z) - z - loggamma - log(a)) * total
                return upper ? 1 - p : p
            end
        end
        throw(ArgumentError("chi-square probability series did not converge"))
    end
    b = isodd(k) ? 0.5 : 1.0
    q = isodd(k) ? 2 * KernelSmoothing.cdf(KernelSmoothing.GaussianKernel(), -sqrt(x)) : exp(-z)
    loggamma = isodd(k) ? log(sqrt(pi)) - log(2.0) : 0.0
    while b < a
        q += exp(b * log(z) - z - loggamma)
        loggamma += log(b + 1)
        b += 1
    end
    upper ? q : 1 - q
end

function _chisq_upper_quantile(k::Int, alpha::Float64)
    upper = alpha <= 0.5
    target = upper ? alpha : 1 - alpha
    lo, hi = 0.0, Float64(k)
    beyond(x) = upper ? _chisq_probability(k, x, true) < target :
        _chisq_probability(k, x, false) > target
    while !beyond(hi)
        hi *= 2
    end
    for _ in 1:1100
        mid = lo + (hi - lo) / 2
        (mid == lo || mid == hi) && return mid
        if beyond(mid)
            hi = mid
        else
            lo = mid
        end
    end
    throw(ArgumentError("chi-square quantile did not converge"))
end

function _band_alpha(alpha)
    alpha isa Real && !(alpha isa Bool) && isfinite(alpha) && 0 < alpha < 1 ||
        throw(ArgumentError("alpha must be finite and strictly between zero and one"))
    a = Float64(alpha)
    0 < a < 1 || throw(ArgumentError("alpha must remain strictly between zero and one in Float64"))
    a
end

"""
    FittedBands

Result of [`fitted_bands`](@ref). `fitted`, `lower`, `upper`, and `variance`
are curve-by-grid matrices. `curves` records training indices in row order.
The result stores copied coordinates, alpha, coverage, critical value, selected
component count, and score method. Variance concerns latent selected-K curves,
conditional on the fitted model; it excludes new measurement error.
"""
struct FittedBands{T<:AbstractFloat}
    grid::Vector{T}
    fitted::Matrix{T}
    lower::Matrix{T}
    upper::Matrix{T}
    variance::Matrix{T}
    curves::Vector{Int}
    alpha::Float64
    coverage::Symbol
    critical_value::Float64
    ncomponents::Int
    method::Symbol
    settings::NamedTuple{(:psd_relative_tolerance, :psd_scale_floor, :interpolation,
        :variance_source, :target, :measurement_error),Tuple{T,T,Symbol,Symbol,Symbol,Bool}}
end

"""
    fitted_bands(f::FPCA; alpha=0.05, coverage=:interval, grid=f.grid, curves=:all)
    fitted_bands(f::MultivariateFPCA; alpha=0.05, coverage=:interval, grid=nothing, curves=:all)

Conditional Gaussian prediction intervals for training latent curves, using
`phi(t)' * score_covariances(f)[:,:,i] * phi(t)`. `:interval` uses the
two-sided normal critical value; `:band` uses `sqrt(chisq(K, 1-alpha))`
(Yao, Müller and Wang, 2005, equations 7–8; Theorems 4–5). K is the selected rank.
These match fdapace 0.6.0 `fitted.FPCA`'s named methods and formulas; that
function defaults to no intervals (`alpha=NULL`) and, when requested, `band`.
This API defaults to alpha 0.05 and pointwise intervals per milestone 5.

The work grid and all curves are used by default, as in fdapace. Grid and
curve selection follow [`reconstruct`](@ref) and [`curve_data`](@ref).
Return [`FittedBands`](@ref), or `(; nodes)` for multivariate fits, with
separate coverage per curve and per node, not joint across subjects or nodes.
Dense integration scores have no conditional variance and throw; use
`method=:pace` when fitting. No uncertainty in estimated mean, covariance,
eigenfunctions or truncation is included. Omitted components are not covered.
Roundoff-negative conditional eigenvalues are clipped within
`100eps(T)*max(1, maximum(abs, eigenvalues))`, a package numerical tolerance
recorded in `settings`; materially indefinite covariances throw.

Common-grid FACE supports `coverage=:interval` with the posterior covariance consistent
with its shrinkage scores. For grid eigenfunctions `Phi`, eigenvalues `Lambda`
and positive noise `sigma2`, its curve variance is the diagonal of
`sigma2 * Phi * (Phi'Phi + sigma2 * inv(Lambda))^(-1) * Phi'`.
This corrects refund 0.1-40's use of its internal scaled eigenvalues in
place of the inverse score covariance; see the FPCA page for its scale.
For masked training data, filled observations are treated as complete, as in
refund; imputation uncertainty is excluded. Sparse FACE uses the same
`Phi * score_covariance * Phi'` projection with its BLUP covariance, which
conditions on the full positive spline covariance as in face 0.1-8.
Simultaneous FACE bands throw.
"""
function fitted_bands(f::FPCA; alpha=0.05, coverage=:interval, grid=f.grid, curves=:all)
    a = _band_alpha(alpha)
    coverage in (:interval, :band) || throw(ArgumentError("coverage must be :interval or :band"))
    f.method === :face && coverage !== :interval && throw(ArgumentError(
        "FACE fitted_bands supports pointwise coverage=:interval only; simultaneous bands are not implemented"))
    f.score_covariances === nothing && throw(ArgumentError(
        "integration scores carry no conditional variance; fit with method = :pace for fitted_bands"))
    grid isa AbstractVector{<:Real} || throw(ArgumentError("fitted bands grid must be a real vector"))
    index = _plot_curves(f, curves)
    recovered = reconstruct(f, f.scores[index, :]; grid)
    functions = basismatrix(f.basis, recovered.grid)
    variance = similar(recovered.values)
    for (row, i) in enumerate(index)
        omega = Symmetric(f.score_covariances[:, :, i])
        # As in fdapace, project roundoff-negative conditional eigenvalues to zero.
        eig = eigen(omega)
        scale = maximum(abs, eig.values; init=zero(eltype(eig.values)))
        minimum(eig.values) >= -100 * eps(eltype(eig.values)) * max(scale, one(scale)) ||
            throw(ArgumentError("conditional score covariance for curve $i is not positive semidefinite"))
        root = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0)))
        variance[row, :] = vec(sum(abs2, functions * root; dims=2))
    end
    critical = sqrt(_chisq_upper_quantile(coverage === :interval ? 1 : ncomponents(f), a))
    width = eltype(variance).(critical .* sqrt.(variance))
    all(isfinite, width) || throw(ArgumentError("fitted bands widths must be finite"))
    lower, upper = recovered.values - width, recovered.values + width
    all(isfinite, lower) && all(isfinite, upper) ||
        throw(ArgumentError("fitted bands endpoints must be finite; rescale the observations"))
    T = eltype(variance)
    settings = (; psd_relative_tolerance=T(100eps(eltype(f.score_covariances))),
        psd_scale_floor=one(T), interpolation=:piecewise_linear,
        variance_source=:stored_selected_score_covariance, target=:selected_latent_curve,
        measurement_error=false)
    FittedBands(copy(recovered.grid), copy(recovered.values), lower,
        upper, variance, index, a, coverage, critical, ncomponents(f), f.method, settings)
end

function fitted_bands(f::MultivariateFPCA{T}; alpha=0.05, coverage=:interval,
        grid=nothing, curves=:all) where {T}
    grids = _multivariate_grids(f, grid)
    R = _multivariate_grid_type(T, grid)
    nodes = FittedBands{R}[]
    for (j, node) in enumerate(f.fits)
        push!(nodes, fitted_bands(node; alpha, coverage, grid=Vector{R}(grids[j]), curves))
    end
    (; nodes)
end

const _BandEndpoints{T} = NamedTuple{(:lower, :upper),Tuple{Vector{T},Vector{T}}}
const _MeanBandSettings = NamedTuple{(:method, :bandwidth_selection, :quantile,
    :centering, :critical_value, :resampling, :variance_estimator, :inference, :minobs, :clusters),
    Tuple{Symbol,Symbol,Union{Nothing,Symbol},Symbol,Union{Nothing,Float64},
        Union{Nothing,Symbol},Union{Nothing,Symbol},Symbol,Int,Union{Nothing,Symbol}}}

"""
    MeanBands

Mean uncertainty result from [`mean_bands`](@ref). `fitted` is the original
mean; `interval` and `band` hold `(; lower, upper)` vectors, or `nothing` when
not requested. `standard_error` is the bootstrap sample standard deviation.
`bootstrap` stores B rows of refitted means. `rng_state` is a copy of the RNG
before drawing; `mean` is a copied `SmoothedMean`. Sandwich results instead
store cluster-robust standard errors, zero bootstrap rows, `B=0`, and
`rng_state=nothing`. All settings are retained.
"""
struct MeanBands{T<:AbstractFloat,M,R,S}
    grid::Vector{T}
    fitted::Vector{T}
    interval::Union{Nothing,_BandEndpoints{T}}
    band::Union{Nothing,_BandEndpoints{T}}
    standard_error::Vector{T}
    bootstrap::Matrix{T}
    alpha::Float64
    B::Int
    coverage::Symbol
    rng_state::R
    mean::M
    settings::S
end

# R's type-7 empirical quantile, also used by fdapace::GetMeanCI.
function _band_percentile(values, p)
    sorted = sort(collect(values))
    h = (length(sorted) - 1) * p
    lo = floor(Int, h) + 1
    hi = min(lo + 1, length(sorted))
    sorted[lo] + (h - floor(h)) * (sorted[hi] - sorted[lo])
end

"""
    mean_bands(fd, m::SmoothedMean; method=:bootstrap, alpha=0.05,
        B=nothing, coverage=(method === :sandwich ? :interval : :both), rng=nothing)

Resample whole independent curves with replacement and refit the pooled mean
with `m`'s stored grid, bandwidth, kernel, degree and weighting. Bandwidth
selection is not repeated. `fd` must be the data used to fit `m`; this is
checked by refitting once. Undefined bootstrap fits throw with the replicate
number; no draws or grid points are dropped. Grid and irregular data, including
masked grids, are supported. At least two curves and B >= 2 are required.

`:interval` gives pointwise percentile intervals (R type-7 quantiles, as in
fdapace 0.6.0 `GetMeanCI`). `:band` gives a gridwise simultaneous sup-t band:
the (1-alpha) quantile of `maximum(abs.(mstar-m.values)./se)` multiplies the
bootstrap standard deviation `se` at each grid point. `:both` returns both.
A zero `se` gives zero standardized deviation only when all replicates equal
the original estimate there; otherwise a requested sup-t band throws.

For `method=:sandwich`, only `coverage=:interval` is supported: use
KernelSmoothing's `ClusterVariance(1)` (CR1, its cluster default), with curves
as clusters, conventional standard errors, and normal critical values around
the stored mean. No robust bias correction is applied. Explicit `B` and `rng`
are bootstrap-only and throw for this method.

Return [`MeanBands`](@ref). Bootstrap `B=nothing` resolves to 1000 and
`rng=nothing` to `Random.default_rng()`. The RNG must support state copying
(`RandomDevice` is not supported). It is copied before use, leaves the caller's
state unchanged, and its initial state is stored for recreation. Alpha 0.05,
B=1000, `:both`, and the default RNG follow milestone 5's Gap Map. Curve-level
resampling preserves within-curve dependence. See Degras (2011), *Simultaneous
confidence bands for nonparametric regression with functional data*, for
functional mean-band theory; this curve bootstrap is not its Gaussian-process
calibration. Coverage concerns the supplied finite grid and requires negligible
smoothing bias. No continuum or arbitrary sparse-design guarantee is claimed.
Sandwich intervals also require negligible smoothing bias.
"""
Base.@constprop :aggressive function mean_bands(fd::Union{GridFunctionalData,IrregularFunctionalData}, m::SmoothedMean;
        method=:bootstrap, alpha=0.05, B=nothing,
        coverage=(method === :sandwich ? :interval : :both), rng=nothing)
    method in (:bootstrap, :sandwich) || throw(ArgumentError("method must be :bootstrap or :sandwich"))
    if method === :sandwich
        B === nothing || throw(ArgumentError("B is only available for method = :bootstrap"))
        rng === nothing || throw(ArgumentError("rng is only available for method = :bootstrap"))
        return _mean_sandwich(fd, m; alpha, coverage)
    end
    generator = rng === nothing ? Random.default_rng() : rng
    generator isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG or nothing"))
    generator isa Random.RandomDevice && throw(ArgumentError(
        "rng must support reproducible state copying; RandomDevice is not supported"))
    # copy(TaskLocalRNG) snapshots its task state into an independent Xoshiro.
    initial = deepcopy(copy(generator))
    try
        _mean_bands(fd, m, initial; alpha, B=B === nothing ? 1000 : B, coverage)
    finally
        # KernelSmoothing may spawn tasks even for deterministic computations.
        # Restore the default RNG's task-splitting state as well as its draws.
        generator isa Random.TaskLocalRNG && copy!(generator, initial)
    end
end

function _mean_band_inputs(fd, m)
    nobs(fd) >= 2 || throw(ArgumentError("mean bands require at least two curves"))
    method = LocalPolynomial(; bandwidth=m.bandwidth, kernel=m.kernel, degree=m.degree)
    t, y, w, subjects = _mean_pool(fd, m.weighting)
    check = _mean_predict(t, y, w, m.grid, method)
    isapprox(check, m.values; rtol=100eps(eltype(m.values)), atol=0) ||
        throw(ArgumentError("fd and m do not give the same fitted mean with the stored settings"))
    t, y, w, subjects, method
end

function _mean_sandwich(fd, m; alpha, coverage)
    a = _band_alpha(alpha)
    coverage === :interval || throw(ArgumentError("method = :sandwich requires coverage = :interval"))
    isdefined(KernelSmoothing, :ClusterVariance) || throw(ArgumentError(
        "method = :sandwich requires KernelSmoothing with ClusterVariance support"))
    t, y, w, subjects, _ = _mean_band_inputs(fd, m)
    regression = KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, t, y;
        bandwidth=m.bandwidth, kernel=m.kernel, degree=m.degree, weights=w,
        cluster=subjects, vce=KernelSmoothing.ClusterVariance(1), points=m.grid, minobs=0)
    T = eltype(m.grid)
    se = Vector{T}(KernelSmoothing.stderror(regression; method=KernelSmoothing.Conventional()))
    for (j, value) in enumerate(se)
        isfinite(value) && value >= 0 || throw(ArgumentError(
            "curve-cluster standard error is undefined at grid point $(m.grid[j]); increase bandwidth or provide more curves"))
    end
    critical = sqrt(_chisq_upper_quantile(1, a))
    interval = (; lower=Vector{T}(m.values - critical .* se), upper=Vector{T}(m.values + critical .* se))
    all(isfinite, interval.lower) && all(isfinite, interval.upper) ||
        throw(ArgumentError("sandwich endpoints must be finite; rescale the observations"))
    settings = _MeanBandSettings((:sandwich, :fixed, nothing, :original_mean,
        critical, nothing, :CR1, :conventional, 0, :curves))
    MeanBands{T,typeof(m),Nothing,_MeanBandSettings}(copy(m.grid), copy(m.values), interval, nothing, se,
        Matrix{T}(undef, 0, length(m.grid)), a, 0, :interval, nothing, deepcopy(m), settings)
end

function _mean_bands(fd, m, initial; alpha, B, coverage)
    a = _band_alpha(alpha)
    B isa Integer && !(B isa Bool) && 2 <= B <= typemax(Int) ||
        throw(ArgumentError("B must be an integer of at least two"))
    coverage in (:interval, :band, :both) ||
        throw(ArgumentError("coverage must be :interval, :band, or :both"))
    n = nobs(fd)
    t, y, w, subjects, method = _mean_band_inputs(fd, m)
    generator = deepcopy(initial)
    T = eltype(m.grid)
    bootstrap = Matrix{T}(undef, B, length(m.grid))
    # Reuse the mean smoother directly: a bootstrap sample's observed range may
    # shrink, but the original grid is retained. Undefined estimates still throw.
    rows = [findall(==(i), subjects) for i in 1:n]
    for b in 1:B
        index = reduce(vcat, rows[rand(generator, 1:n, n)])
        try
            bootstrap[b, :] = _mean_predict(t[index], y[index], w[index], m.grid, method)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("mean bootstrap replicate $b failed: $(err.msg)"))
        end
    end
    all(isfinite, bootstrap) || throw(ArgumentError("bootstrap means must remain finite after conversion"))
    average = sum(bootstrap; dims=1) ./ B
    se = vec(sqrt.(sum(abs2, bootstrap .- average; dims=1) ./ (B - 1)))
    all(isfinite, se) || throw(ArgumentError("bootstrap standard errors must be finite; rescale the observations"))
    interval = coverage in (:interval, :both) ?
        (; lower=T[_band_percentile(c, a/2) for c in eachcol(bootstrap)],
            upper=T[_band_percentile(c, 1-a/2) for c in eachcol(bootstrap)]) : nothing
    if interval !== nothing
        all(isfinite, interval.lower) && all(isfinite, interval.upper) ||
            throw(ArgumentError("percentile endpoints must be finite; rescale the observations"))
    end
    critical = nothing
    band = nothing
    if coverage in (:band, :both)
        maxima = zeros(T, B)
        for j in eachindex(se)
            deviation = abs.(bootstrap[:, j] .- m.values[j])
            if iszero(se[j])
                all(iszero, deviation) || throw(ArgumentError(
                    "sup-t band has zero bootstrap standard error and nonzero deviation at grid point $(m.grid[j]); increase B or inspect the data"))
            else
                maxima .= max.(maxima, deviation ./ se[j])
            end
        end
        critical = _band_percentile(maxima, 1-a)
        isfinite(critical) || throw(ArgumentError("sup-t critical value must be finite; rescale the observations"))
        width = critical .* se
        band = (; lower=Vector{T}(m.values - width), upper=Vector{T}(m.values + width))
        all(isfinite, band.lower) && all(isfinite, band.upper) ||
            throw(ArgumentError("sup-t endpoints must be finite; rescale the observations"))
    end
    settings = _MeanBandSettings((:bootstrap, :fixed, :type7, :original_mean,
        critical, :curves, nothing, :bootstrap, 0, nothing))
    MeanBands(copy(m.grid), copy(m.values), interval, band, se, bootstrap,
        a, Int(B), coverage, initial, deepcopy(m), settings)
end

function Base.show(io::IO, b::FittedBands)
    print(io, "FittedBands(alpha=", b.alpha, ", coverage=", b.coverage,
        ", grid=", b.grid, ", curves=", b.curves, ", ncomponents=", b.ncomponents,
        ", method=", b.method, ", critical_value=", b.critical_value, ", settings=")
    _show_band_settings(io, b.settings)
    print(io, ")")
end

function Base.show(io::IO, b::MeanBands)
    print(io, "MeanBands(alpha=", b.alpha, ", B=", b.B, ", coverage=", b.coverage,
        ", rng_state=", b.rng_state, ", mean=", b.mean, ", settings=")
    _show_band_settings(io, b.settings)
    print(io, ")")
end

function _show_band_settings(io, settings)
    print(io, "(")
    for (i, (name, value)) in enumerate(pairs(settings))
        i == 1 || print(io, ", ")
        print(io, name, " = ")
        show(io, value)
    end
    print(io, ")")
end
