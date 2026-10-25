"""
    ConcurrentRegression
    fit(ConcurrentRegression, x, y; mean_bandwidth, covariance_bandwidth,
        scalar_covariates=nothing, grid=nothing,
        kernel=KernelSmoothing.GaussianKernel(), measurement_error=true)

Concurrent functional regression `Y_i(t) = beta_0(t) + sum_j beta_j(t) X_ij(t) +
sum_k gamma_k(t) z_ik + e_i(t)` by fdapace 0.6.0 `FCReg` (the successor of
`ConcurReg`, which 0.6.0 no longer contains). `x` is one univariate grid or
irregular dataset, a tuple/vector of them, or `MultivariateFunctionalData`;
`y` is a grid or irregular response. Dense and sparse designs use the same
estimator, and covariates and response may be observed at different times.
Optional scalar covariates give varying coefficients `gamma_k(t)`.

Every functional variable is centered by a local linear mean (bandwidth
`mean_bandwidth`) evaluated at its own sorted unique observation times and
interpolated linearly, with constant extrapolation (`approxfun(rule=2)`).
Scalar covariates are centered by their sample means. All within-subject
products of centered values are smoothed by the existing bivariate local
linear smoother with `covariance_bandwidth` on both axes and unit weights; only
the diagonal `(t,t)` is needed and computed. A variable's own products drop
equal-time pairs when `measurement_error=true`, removing the noise variance from
the diagonal. Functional-scalar products are smoothed by the univariate local
linear smoother, and scalar-scalar covariances are sample covariances
(denominator `n-1`). Then `beta(t) = C_XX(t)^(-1) C_XY(t)`,
`R2(t) = C_YX(t) beta(t) / C_YY(t)` clipped to `[0,1]`, and
`beta_0(t) = mu_Y(t) - sum_j beta_j(t) mu_j(t)`.

Defaults follow `FCReg`: Gaussian kernel and `measurement_error=true`.
`FCReg` has no bandwidth defaults (`userBwMu` and `userBwCov` are required),
so both bandwidths are required here too. `FCReg` also requires `outGrid`;
`grid=nothing` uses the sorted union of all functional observation times, the
default of its internal `MvCov`. `FCReg`'s optional `diag1D` one-dimensional
smoothing (default `"none"`) and its unused `useGAM` argument are not offered.
Basis-represented data must be evaluated on a grid explicitly.

The fit stores the grid, intercept, coefficients (functional covariates then
scalar covariates, as listed), the pointwise covariance matrices of
`(X..., z..., Y)` at `(t,t)`, `R2(t)`, the means, every tuning choice, and
copies of the data. FCReg returns no intervals; [`coefficient_bands`](@ref)
offers subject-bootstrap percentile intervals with the same bandwidths.
"""
struct ConcurrentRegression{T<:AbstractFloat,M,K,D,S}
    grid::Vector{T}
    intercept::Vector{T}
    coefficients::Matrix{T}
    covariances::Array{T,3}
    r_squared::Vector{T}
    means::M
    scalar_means::Vector{T}
    nfunctional::Int
    nscalar::Int
    mean_bandwidth::T
    covariance_bandwidth::T
    kernel::K
    measurement_error::Bool
    data::D
    settings::S
end

function _concurrent_inputs(x)
    curves = _regression_inputs(x)
    all(c -> c isa Union{GridFunctionalData,IrregularFunctionalData}, curves) || throw(ArgumentError(
        "concurrent regression smooths raw observations; evaluate basis covariates on a grid explicitly"))
    curves
end

function _concurrent_bandwidth(h, kernel, name)
    try
        LocalPolynomial(; bandwidth=h, kernel).bandwidth
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("$name must be a finite positive number"))
    end
end

function _concurrent_times(fd, ::Type{T}) where {T}
    times = T[]
    for i in 1:nobs(fd)
        t, _ = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values for concurrent regression"))
        append!(times, t)
    end
    sort!(unique!(times))
end

# fdapace GetSmoothedMeanCurve at the sorted unique observation times.
function _concurrent_mean(fd, h, kernel, ::Type{T}) where {T}
    times = _concurrent_times(fd, T)
    length(times) >= 2 || throw(ArgumentError(
        "each functional variable needs at least two distinct observation times"))
    fit(SmoothedMean, fd; grid=times, bandwidth=h, kernel, weighting=:obs)
end

# approxfun(rule = 2): linear inside the mean grid and constant outside it.
_concurrent_mean_values(m::SmoothedMean, grid) =
    evaluate(m, clamp.(grid, first(m.grid), last(m.grid)))

function _concurrent_cross(fd1, fd2, m1, m2, grid, h, kernel, exclude)
    pool = _cross_pool(fd1, fd2, m1, m2, :obs, exclude, nothing; aggregate=true, require_pairs=false)
    isempty(pool.z) && throw(ArgumentError("concurrent regression has no within-subject product pairs"))
    values = KernelSmoothing.local_polynomial((pool.x, pool.y), pool.z, [(t, t) for t in grid];
        bandwidth=h, degree=1, kernel, weights=pool.weights)
    for (t, value) in zip(grid, values)
        isfinite(value) || throw(ArgumentError(
            "concurrent covariance has an undefined local linear fit at ($t, $t); use a larger covariance_bandwidth"))
    end
    values
end

function _concurrent_functional_scalar(fd, m, z, grid, h, kernel)
    t, y, w, subjects = _mean_pool(fd, :obs)
    products = z[subjects] .* (y - evaluate(m, t))
    _mean_predict(t, products, w, grid, LocalPolynomial(; bandwidth=h, kernel, degree=1);
        label="concurrent functional-scalar covariance")
end

# One FCReg estimate. `variables` lists the functional covariates and then the response.
function _concurrent_estimate(variables, z::Matrix{T}, grid::Vector{T}, hmean, hcov,
        kernel, measurement_error) where {T}
    n = nobs(first(variables))
    n >= 2 || throw(ArgumentError("concurrent regression requires at least two subjects"))
    pf, ps = length(variables) - 1, size(z, 2)
    q, m = pf + ps, length(grid)
    means = map(v -> _concurrent_mean(v, hmean, kernel, T), variables)
    zmean = vec(sum(z; dims=1) ./ n)
    zc = z .- transpose(zmean)
    # Positions in the covariance array: functional covariates, scalars, response.
    position(i) = i <= pf ? i : q + 1
    covariances = zeros(T, q + 1, q + 1, m)
    for i in 1:pf+1, j in 1:i
        values = _concurrent_cross(variables[i], variables[j], means[i], means[j], grid,
            hcov, kernel, i == j && measurement_error)
        a, b = position(i), position(j)
        covariances[a, b, :] = values
        covariances[b, a, :] = values
    end
    for i in 1:pf+1, k in 1:ps
        values = _concurrent_functional_scalar(variables[i], means[i], view(zc, :, k), grid, hcov, kernel)
        covariances[position(i), pf + k, :] = values
        covariances[pf + k, position(i), :] = values
    end
    if ps > 0
        scalar_covariance = transpose(zc)*zc/(n - 1)
        for t in 1:m
            covariances[pf+1:q, pf+1:q, t] = scalar_covariance
        end
    end
    all(isfinite, covariances) || throw(ArgumentError(
        "concurrent covariances must be finite; rescale the observations"))
    coefficients = zeros(T, q, m)
    r_squared = zeros(T, m)
    for t in 1:m
        C = covariances[:, :, t]
        XX = C[1:q, 1:q]
        factor = lu(XX; check=false)
        issuccess(factor) && rank(XX) == q || throw(ArgumentError(
            "covariate covariance is singular at grid point $(grid[t]); use larger bandwidths or remove redundant covariates"))
        coefficients[:, t] = factor \ C[1:q, q+1]
        r_squared[t] = clamp(dot(C[q+1, 1:q], coefficients[:, t]) / C[q+1, q+1], zero(T), one(T))
    end
    all(isfinite, coefficients) || throw(ArgumentError(
        "concurrent coefficients must be finite; use larger bandwidths or rescale"))
    intercept = _concurrent_mean_values(means[end], grid)
    for j in 1:pf
        intercept .-= coefficients[j, :] .* _concurrent_mean_values(means[j], grid)
    end
    for k in 1:ps
        intercept .-= coefficients[pf + k, :] .* zmean[k]
    end
    (; intercept, coefficients, covariances, r_squared, means, zmean)
end

Base.@constprop :aggressive function fit(::Type{ConcurrentRegression}, x,
        y::Union{GridFunctionalData,IrregularFunctionalData};
        mean_bandwidth, covariance_bandwidth, scalar_covariates=nothing, grid=nothing,
        kernel=KernelSmoothing.GaussianKernel(), measurement_error=true)
    curves = _concurrent_inputs(x)
    nobs(y) == nobs(first(curves)) || throw(DimensionMismatch(
        "response must have the functional covariate subject count"))
    _validate_component_ids((curves..., y))
    measurement_error isa Bool || throw(ArgumentError("measurement_error must be a Bool"))
    hmean = _concurrent_bandwidth(mean_bandwidth, kernel, "mean_bandwidth")
    hcov = _concurrent_bandwidth(covariance_bandwidth, kernel, "covariance_bandwidth")
    scalar_covariates === nothing || scalar_covariates isa _RegressionScalars ||
        throw(ArgumentError("scalar_covariates must be a finite real vector or matrix"))
    scalar_type = scalar_covariates === nothing ? eltype(y) : Base.nonmissingtype(eltype(scalar_covariates))
    T = _floattype(promote_type(eltype(y), scalar_type, map(eltype, curves)...,
        typeof(hmean), typeof(hcov)))
    _concurrent_fit(T, curves, y, scalar_covariates, grid, hmean, hcov, kernel, measurement_error)
end
fit(::Type{ConcurrentRegression}, x, y; kwargs...) = throw(ArgumentError(
    "concurrent regression requires a grid or irregular functional response"))

# Function barrier: `T` is a static parameter here, so closures over it stay inferred.
function _concurrent_fit(::Type{T}, curves, y, scalar_covariates, grid, hmean, hcov, kernel,
        measurement_error) where {T}
    variables = map(v -> _convert_component(T, v), (curves..., y))
    z = _regression_scalars(scalar_covariates, nobs(y), T)
    g = if grid === nothing
        sort!(unique!(reduce(vcat, map(v -> _concurrent_times(v, T), variables))))
    else
        grid isa AbstractVector{<:Real} || throw(ArgumentError("grid must be a real vector"))
        Vector{T}(grid)
    end
    _validate_strict_grid(g)
    estimate = _concurrent_estimate(variables, z, g, T(hmean), T(hcov), kernel, measurement_error)
    settings = (; reference="fdapace 0.6.0 FCReg (successor of ConcurReg)",
        grid_source=grid === nothing ? :union_of_observation_times : :supplied,
        mean_grid=:unique_observation_times, mean_extrapolation=:constant,
        degree=1, weighting=:obs, diag1D=:none, binning=:none,
        scalar_covariance_denominator=:n_minus_1, r_squared_clipping=:unit_interval,
        coefficient_order=:functional_then_scalar)
    data = (; covariates=Base.front(variables), response=last(variables), scalars=copy(z))
    ConcurrentRegression(copy(g), estimate.intercept, estimate.coefficients, estimate.covariances,
        estimate.r_squared, estimate.means, estimate.zmean, length(curves), size(z, 2),
        T(hmean), T(hcov), deepcopy(kernel), measurement_error, data, settings)
end

"""
Return a `(1 + q) × length(grid)` matrix: the intercept function in the first
row, then functional and scalar covariate coefficients, on the fitted grid.
"""
coef(f::ConcurrentRegression) = vcat(transpose(f.intercept), f.coefficients)
"""Return the number of subjects used in a concurrent regression."""
nobs(f::ConcurrentRegression) = nobs(f.data.response)

"""
    coefficient_functions(f::ConcurrentRegression)

Return a tuple of one-curve `GridFunctionalData` on the fitted grid: the
intercept `beta_0`, then each functional and scalar covariate coefficient.
"""
coefficient_functions(f::ConcurrentRegression) =
    Tuple(GridFunctionalData(reshape(row, 1, :), copy(f.grid)) for row in eachrow(coef(f)))

pointwise_r2(f::ConcurrentRegression) = (; grid=copy(f.grid), values=copy(f.r_squared))

function _concurrent_values(fd, grid, evaluator)
    evaluator === nothing || evaluator isa Union{PiecewiseLinear,NoInterpolation} ||
        throw(ArgumentError("evaluator must be nothing, PiecewiseLinear(), or NoInterpolation()"))
    values = if fd isa BasisFunctionalData
        evaluate(fd, grid)
    elseif fd isa GridFunctionalData && fd.observed === nothing && fd.grid == grid
        fd.values
    else
        evaluator === nothing && throw(ArgumentError(
            "covariates not completely observed on the fitted grid need evaluator=PiecewiseLinear() or NoInterpolation()"))
        evaluate(fd, grid, evaluator)
    end
    all(isfinite, values) || throw(ArgumentError(
        "covariate values on the fitted grid must be finite under the explicit evaluator"))
    values
end

"""
    predict(f::ConcurrentRegression, x; scalar_covariates=nothing, evaluator=nothing)

Return `GridFunctionalData` with `beta_0(t) + sum_j beta_j(t) X_j(t) + sum_k gamma_k(t) z_k`
on the fitted grid. Covariates completely observed on exactly that grid, and
basis curves, are used directly; other raw curves need an explicit
`evaluator=PiecewiseLinear()` or `NoInterpolation()` (one per covariate is
allowed), and must cover the grid. No smoother is refitted.
"""
function predict(f::ConcurrentRegression, x; scalar_covariates=nothing, evaluator=nothing)
    curves = _regression_inputs(x)
    length(curves) == f.nfunctional || throw(DimensionMismatch(
        "prediction functional covariate count must match the concurrent fit"))
    T = eltype(f.grid)
    z = _regression_scalars(scalar_covariates, nobs(first(curves)), T)
    size(z, 2) == f.nscalar || throw(DimensionMismatch(
        "prediction scalar_covariates column count must match the concurrent fit"))
    evaluators = _regression_each(evaluator, length(curves), "evaluator")
    values = repeat(transpose(f.intercept), nobs(first(curves)), 1)
    for (j, (c, e)) in enumerate(zip(curves, evaluators))
        values .+= _concurrent_values(c, f.grid, e) .* transpose(f.coefficients[j, :])
    end
    for k in 1:f.nscalar
        values .+= z[:, k] .* transpose(f.coefficients[f.nfunctional + k, :])
    end
    all(isfinite, values) || throw(ArgumentError("concurrent predictions must be finite"))
    GridFunctionalData(values, copy(f.grid))
end

"""
    fitted(f::ConcurrentRegression; evaluator=nothing)

Predict the training responses on the fitted grid from the stored covariates;
see [`predict`](@ref) for the evaluator rule.
"""
fitted(f::ConcurrentRegression; evaluator=nothing) =
    predict(f, f.data.covariates; scalar_covariates=f.nscalar == 0 ? nothing : f.data.scalars, evaluator)

"""
    coefficient_bands(f::ConcurrentRegression; alpha=0.05, B=999, rng=nothing)

Pointwise subject-bootstrap percentile intervals for the intercept and every
coefficient, returned as a tuple in `coef` order. Each of `B` resamples of
whole subjects refits the means and covariances with the stored bandwidths,
kernel, grid and noise setting (no bandwidth reselection), and R type-7
quantiles give the interval. fdapace 0.6.0 `FCReg` provides no intervals; the
resampling scheme, `B=999` and `alpha=0.05` follow its `FLMCI` regression
bootstrap. A failed refit throws with its replicate number; no draw is
dropped. `se` is the bootstrap standard deviation and `critical_value` is
`nothing`. `rng=nothing` uses `Random.default_rng()`, copied so the caller's
state is unchanged; its initial state is stored in `settings.rng_state`.
Intervals are pointwise and ignore smoothing bias.
"""
function coefficient_bands(f::ConcurrentRegression; alpha=0.05, B=999, rng=nothing)
    a = _band_alpha(alpha)
    _integer_in(B, 2) || throw(ArgumentError("B must be an integer of at least two"))
    generator = rng === nothing ? Random.default_rng() : rng
    generator isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG or nothing"))
    generator isa Random.RandomDevice && throw(ArgumentError(
        "rng must support reproducible state copying; RandomDevice is not supported"))
    initial = deepcopy(copy(generator))
    try
        _concurrent_bootstrap(f, a, Int(B), deepcopy(initial), initial)
    finally
        generator isa Random.TaskLocalRNG && copy!(generator, initial)
    end
end

function _concurrent_bootstrap(f, a, B, generator, initial)
    T = eltype(f.grid)
    estimate = coef(f)
    draws = Array{T}(undef, B, size(estimate)...)
    variables = (f.data.covariates..., f.data.response)
    n = nobs(f)
    for b in 1:B
        index = rand(generator, 1:n, n)
        replicate = try
            _concurrent_estimate(map(v -> _regression_resample(v, index), variables),
                f.data.scalars[index, :], f.grid, f.mean_bandwidth, f.covariance_bandwidth,
                f.kernel, f.measurement_error)
        catch err
            err isa Union{ArgumentError,DimensionMismatch} || rethrow()
            throw(ArgumentError("concurrent bootstrap replicate $b failed: $(sprint(showerror, err))"))
        end
        draws[b, 1, :] = replicate.intercept
        draws[b, 2:end, :] = replicate.coefficients
    end
    settings = (; target=:coefficient, method=:bootstrap, B, quantile=:type7,
        resampling=:subjects, bandwidth_selection=:fixed,
        reference="fdapace 0.6.0 FLMCI resampling applied to FCReg", rng_state=initial)
    Tuple(map(axes(estimate, 1)) do j
        column(t) = view(draws, :, j, t)
        lower = T[_band_percentile(column(t), a/2) for t in axes(estimate, 2)]
        upper = T[_band_percentile(column(t), 1 - a/2) for t in axes(estimate, 2)]
        se = T[sqrt(sum(abs2, column(t) .- sum(column(t))/B)/(B - 1)) for t in axes(estimate, 2)]
        all(isfinite, lower) && all(isfinite, upper) || throw(ArgumentError(
            "bootstrap interval endpoints must be finite"))
        (; grid=copy(f.grid), estimate=estimate[j, :], se, lower, upper, alpha=a,
            critical_value=nothing, settings)
    end)
end

function Base.show(io::IO, f::ConcurrentRegression)
    print(io, "ConcurrentRegression(n=", nobs(f), ", functional_covariates=", f.nfunctional,
        ", scalar_covariates=", f.nscalar, ", grid=", length(f.grid), " points",
        ", mean_bandwidth=", f.mean_bandwidth, ", covariance_bandwidth=", f.covariance_bandwidth,
        ", kernel=", f.kernel, ", measurement_error=", f.measurement_error,
        ", settings=", f.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::ConcurrentRegression) = show(io, f)
