# PACE uses the existing pooled smoothers and the weighted FPCA eigen step.
# The helpers below also handle explicit userMu/userCov-style inputs.
function _pace_input_grid(grid, label)
    grid isa AbstractVector{<:Real} || throw(ArgumentError("$label grid must be a real vector"))
    Base.require_one_based_indexing(grid)
    length(grid) >= 2 || throw(ArgumentError("$label grid must contain at least two points"))
    all(isfinite, grid) || throw(ArgumentError("$label grid must be finite"))
    all(i -> grid[i] < grid[i+1], 1:(length(grid)-1)) ||
        throw(ArgumentError("$label grid must be strictly increasing"))
    copy(grid)
end

function _pace_check_coverage(grid, points, label)
    all(t -> first(grid) <= t <= last(grid), points) ||
        throw(ArgumentError("$label grid must cover the work grid and every observation time"))
end

function _pace_observations(fd, grid)
    for i in 1:nobs(fd)
        times, _ = fd[i]
        isempty(times) && throw(ArgumentError("curve $i has no observed values for PACE"))
        all(t -> first(grid) <= t <= last(grid), times) || throw(ArgumentError(
            "curve $i has observation times outside the PACE work grid"))
    end
    nothing
end

function _pace_validate_mean(mean, workgrid, fd)
    grid = _pace_input_grid(mean.grid, "supplied mean")
    mean.values isa AbstractVector{<:Real} ||
        throw(ArgumentError("supplied mean values must be a real vector"))
    Base.require_one_based_indexing(mean.values)
    length(mean.values) == length(grid) ||
        throw(DimensionMismatch("supplied mean values must match its grid length"))
    all(isfinite, mean.values) || throw(ArgumentError("supplied mean values must be finite"))
    _pace_check_coverage(grid, workgrid, "supplied mean")
    for i in 1:nobs(fd)
        _pace_check_coverage(grid, fd[i][1], "supplied mean")
    end
    nothing
end

function _pace_mean(fd, ::Nothing, grid, bandwidth, kernel, weighting)
    bandwidth === nothing && throw(ArgumentError(
        "mean_bandwidth is required when estimating the PACE mean"))
    fit(SmoothedMean, fd; grid, bandwidth, kernel, weighting), :estimated
end

function _pace_mean(fd, mean::SmoothedMean, grid, bandwidth, kernel, weighting)
    _require_value_mean(mean)
    _pace_validate_mean(mean, grid, fd)
    deepcopy(mean), :supplied
end

function _pace_mean(fd, mean::NamedTuple, grid, bandwidth, kernel, weighting)
    haskey(mean, :grid) && haskey(mean, :values) ||
        throw(ArgumentError("supplied mean must be a named tuple with grid and values"))
    _pace_validate_mean(mean, grid, fd)
    (; grid=copy(mean.grid), values=copy(mean.values)), :supplied
end

_pace_mean(fd, mean, grid, bandwidth, kernel, weighting) = throw(ArgumentError(
    "mean must be nothing, SmoothedMean, or a named tuple with grid and values"))

function _pace_mean_values(mean::SmoothedMean, points)
    _require_value_mean(mean)
    evaluate(mean, points)
end
function _pace_mean_values(mean::NamedTuple, points)
    T = _floattype(promote_type(eltype(mean.grid), eltype(mean.values), eltype(points)))
    _evaluate_curve!(Vector{T}(undef, length(points)), mean.grid, mean.values,
        points, PiecewiseLinear())
end

function _pace_validate_covariance(covariance, workgrid)
    grid = _pace_input_grid(covariance.grid, "supplied covariance")
    covariance.surface isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("supplied covariance surface must be a real matrix"))
    Base.require_one_based_indexing(covariance.surface)
    size(covariance.surface) == (length(grid), length(grid)) ||
        throw(DimensionMismatch("supplied covariance surface must be square and match its grid length"))
    all(isfinite, covariance.surface) ||
        throw(ArgumentError("supplied covariance surface must be finite"))
    S = covariance.surface
    # Accept rounding-level asymmetry (R's isSymmetric allows ~1.5e-8); callers symmetrize.
    isapprox(S, transpose(S); rtol=sqrt(eps(float(eltype(S))))) ||
        throw(ArgumentError("supplied covariance surface must be symmetric up to a relative tolerance of sqrt(eps)"))
    _pace_check_coverage(grid, workgrid, "supplied covariance")
    nothing
end

function _pace_covariance(fd, mean::SmoothedMean, ::Nothing, grid,
        bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    bandwidth === nothing && throw(ArgumentError(
        "covariance_bandwidth is required when estimating the PACE covariance"))
    fitted = fit(SmoothedCovariance, fd, mean; grid, bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    _pace_covariance_storage(fitted), :estimated
end

# Automatic binning is data-dependent; retain a fixed optional field type so
# its decision does not make the public FPCA result type depend on the data.
function _pace_covariance_storage(c::SmoothedCovariance{T,H,D,K,S,N}) where {T,H,D,K,S,N}
    SmoothedCovariance{T,H,D,K,S,N,Union{Nothing,Int}}(c.grid, c.surface, c.weights,
        c.bandwidth, c.diagonal_bandwidth, c.kernel, c.weighting, c.binning,
        c.degree, c.design, c.folds, c.selection, c.noise_variance, c.noise_details)
end

function _pace_covariance(fd, mean, ::Nothing, grid, bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    throw(ArgumentError("estimating the PACE covariance requires an estimated mean or a SmoothedMean; a user-supplied mean named tuple cannot be used by SmoothedCovariance"))
end

function _pace_covariance(fd, mean, covariance::SmoothedCovariance, grid,
        bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    _pace_validate_covariance(covariance, grid)
    c = deepcopy(covariance)
    c.surface .= (covariance.surface .+ transpose(covariance.surface)) ./ 2
    c, :supplied
end

function _pace_covariance(fd, mean, covariance::NamedTuple, grid,
        bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    haskey(covariance, :grid) && haskey(covariance, :surface) ||
        throw(ArgumentError("supplied covariance must be a named tuple with grid and surface"))
    _pace_validate_covariance(covariance, grid)
    surface = (covariance.surface .+ transpose(covariance.surface)) ./ 2
    (; grid=copy(covariance.grid), surface), :supplied
end

_pace_covariance(fd, mean, covariance, grid, bandwidth, diagonal_bandwidth, kernel, weighting, binning) =
    throw(ArgumentError("covariance must be nothing, SmoothedCovariance, or a named tuple with grid and surface"))

function _pace_bilinear(grid, surface, s, t)
    i = min(searchsortedlast(grid, s), length(grid)-1)
    j = min(searchsortedlast(grid, t), length(grid)-1)
    u = (s-grid[i])/(grid[i+1]-grid[i])
    v = (t-grid[j])/(grid[j+1]-grid[j])
    (1-u)*(1-v)*surface[i, j] + u*(1-v)*surface[i+1, j] +
        (1-u)*v*surface[i, j+1] + u*v*surface[i+1, j+1]
end

function _pace_surface(covariance, points)
    T = _floattype(promote_type(eltype(covariance.grid), eltype(covariance.surface), eltype(points)))
    surface = T[_pace_bilinear(covariance.grid, covariance.surface, s, t) for s in points, t in points]
    # Independent interpolation at transposed coordinates can differ by one ulp.
    (surface + transpose(surface))/2
end

function _pace_noise(covariance::SmoothedCovariance, ::Nothing)
    value = covariance.noise_variance
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("SmoothedCovariance noise_variance must be finite and nonnegative"))
    value, :estimated
end

_pace_noise(covariance, ::Nothing) = throw(ArgumentError(
    "noise_variance is required with a user-supplied covariance surface"))

function _pace_noise(covariance, value)
    value isa Real && isfinite(value) && value >= 0 ||
        throw(ArgumentError("noise_variance must be a finite nonnegative number"))
    converted = float(value)
    isfinite(converted) || throw(ArgumentError("noise_variance must remain finite after conversion"))
    converted, :supplied
end

function _pace_interpolated_basis(grid, functions, points)
    T = promote_type(eltype(grid), eltype(functions), eltype(points))
    values = Matrix{T}(undef, length(points), size(functions, 2))
    for j in axes(functions, 2)
        _evaluate_curve!(view(values, :, j), grid, view(functions, :, j), points,
            PiecewiseLinear())
    end
    values
end

_pace_failure_advice(::Nothing, advice) = advice

function _pace_failure_advice(interpolation::AbstractMatrix, advice)
    rows = size(interpolation, 1)
    row_rank = rank(interpolation)
    summary = "interpolation matrix has numerical row rank $row_rank of $rows"
    row_rank < rows && return "$summary; refine the work grids to separate observation times, or change the observation design; additional grid noise cannot restore interpolation row rank"
    "$summary; $advice"
end

function _pace_cholesky(covariance, curve; criterion=false, label="PACE",
        advice="supply a positive noise_variance", interpolation=nothing)
    all(isfinite, covariance) || throw(ArgumentError(
        "$label covariance for curve $curve must be finite; rescale the observations"))
    factor = cholesky(Symmetric(covariance); check=false)
    if !isposdef(factor)
        diagnostic = _pace_failure_advice(interpolation, advice)
        throw(ArgumentError(
            "$label covariance for curve $curve is not positive definite$(criterion ? " during component selection" : ""); $diagnostic"))
    end
    factor
end

_pace_interpolated_mean(grid, mu, times) = _evaluate_curve!(
    Vector{promote_type(eltype(grid), eltype(mu), eltype(times))}(undef, length(times)),
    grid, mu, times, PiecewiseLinear())

function _pace_ce(fd, grid, mu, functions, values, projected, sigma2;
        curve_ce=_pace_curve_ce, basis_evaluator=_pace_interpolated_basis,
        mean_evaluator=_pace_interpolated_mean)
    T = promote_type(eltype(grid), eltype(mu), eltype(functions), eltype(values),
        eltype(projected), typeof(sigma2), eltype(fd))
    k, n = length(values), nobs(fd)
    output = Matrix{T}(undef, n, k)
    covariances = Array{T}(undef, k, k, n)
    for i in 1:n
        times, observations = fd[i]
        isempty(times) && throw(ArgumentError("curve $i has no observed values for conditional scores"))
        phi = basis_evaluator(grid, functions, times)
        local_mean = convert(Vector{T}, mean_evaluator(grid, mu, times))
        output[i, :], covariances[:, :, i] = curve_ce(times,
            observations-local_mean, phi, values, grid, projected, sigma2, i, T)
    end
    all(isfinite, output) && all(isfinite, covariances) || throw(ArgumentError(
        "PACE scores and score covariances must be finite; rescale the observations"))
    (; scores=output, covariances)
end

_pace_observation_covariance(grid, projected, times) =
    [_pace_bilinear(grid, projected, s, t) for s in times, t in times]

function _pace_curve_ce(times, residuals, phi, values, grid, projected::AbstractMatrix,
        sigma2, curve, ::Type{T}; covariance_solver=_pace_cholesky) where {T}
    covariance = Matrix{T}(_pace_observation_covariance(grid, projected, times))
    for j in axes(covariance, 1)
        covariance[j, j] += sigma2
    end
    factor = covariance_solver(covariance, curve)
    numerator = phi .* transpose(values)
    projected_scores = transpose(numerator) * (factor \ residuals)
    omega = Diagonal(values) - transpose(numerator) * (factor \ numerator)
    projected_scores, (omega + transpose(omega))/2
end

function _pace_likelihood_selection(rule::Union{AIC,BIC}, fd, grid, mu, system, max_k, sigma2)
    values, functions = system.eigenvalues, system.eigenfunctions
    T = eltype(values)
    candidates = collect(1:max_k)
    criterion = zeros(T, length(candidates))
    for i in 1:nobs(fd)
        times, observations = fd[i]
        phi = _pace_interpolated_basis(grid, functions, times)
        local_mean = _evaluate_curve!(Vector{T}(undef, length(times)), grid, mu,
            times, PiecewiseLinear())
        residuals = observations - local_mean
        for k in candidates
            # The projected covariance has zero eigenvalues beyond its positive
            # rank; retain those nominal K candidates for the complete criterion.
            retained = min(k, length(values))
            covariance = (view(phi, :, 1:retained) .* transpose(view(values, 1:retained))) *
                transpose(view(phi, :, 1:retained))
            for j in axes(covariance, 1)
                covariance[j, j] += sigma2
            end
            factor = _pace_cholesky(covariance, i; criterion=true)
            criterion[k] += length(times)*log(T(2)*T(pi)) +
                2sum(log, diag(factor.L)) + dot(residuals, factor \ residuals)
        end
    end
    for k in candidates
        criterion[k] += rule isa AIC ? 2k : k*log(T(nobs(fd)))
    end
    all(isfinite, criterion) || throw(ArgumentError(
        "PACE component-selection criteria must be finite; rescale the observations"))
    increase = findfirst(>(zero(T)), diff(criterion))
    chosen = increase === nothing ? length(candidates) : increase
    (; rule=rule isa AIC ? :AIC : :BIC, threshold=nothing, max_k, maxK=max_k,
        chosen, scores=criterion, candidates, global_minimizer=argmin(criterion),
        stopping=:first_increase)
end

function _pace_selection(rule, fd, grid, mu, system, max_k, sigma2)
    _fpca_selection(rule, system.fve, max_k)
end

function _pace_selection(rule::Union{AIC,BIC}, fd, grid, mu, system, max_k, sigma2)
    _pace_likelihood_selection(rule, fd, grid, mu, system, max_k, sigma2)
end

_pace_mean_settings(mean::SmoothedMean) = (; bandwidth=mean.bandwidth,
    kernel=deepcopy(mean.kernel), degree=mean.degree, derivative=mean.derivative, weighting=mean.weighting,
    design=mean.design, minobs=mean.minobs, binning=mean.binning, folds=mean.folds,
    selection=deepcopy(mean.selection), grid=copy(mean.grid))
_pace_mean_settings(mean::NamedTuple) = (; interpolation=:piecewise_linear, grid=copy(mean.grid))

_pace_covariance_settings(covariance::SmoothedCovariance) =
    merge((; bandwidth=covariance.bandwidth, diagonal_bandwidth=covariance.diagonal_bandwidth,
        kernel=deepcopy(covariance.kernel), weighting=covariance.weighting,
        degree=covariance.degree, design=covariance.design,
        folds=covariance.folds, selection=deepcopy(covariance.selection),
        noise_details=deepcopy(covariance.noise_details), grid=copy(covariance.grid)),
        NamedTuple{(:binning,),Tuple{Union{Nothing,Int}}}((covariance.binning,)))
_pace_covariance_settings(covariance::NamedTuple) =
    (; interpolation=:bilinear, grid=copy(covariance.grid))

function _fit_pace(fd; method=:auto, grid=nothing, ncomponents=FVE(0.99), max_k=nothing,
        mean=nothing, covariance=nothing, noise_variance=nothing, mean_bandwidth=nothing,
        covariance_bandwidth=nothing, diagonal_bandwidth=nothing, kernel=nothing,
        weighting=:obs, binning=:auto, eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL)
    nobs(fd) >= 2 || throw(ArgumentError("PACE FPCA requires at least two curves"))
    mean === nothing || mean_bandwidth === nothing || throw(ArgumentError(
        "mean_bandwidth cannot be used with a supplied mean; supply a fitted SmoothedMean or let FPCA estimate the mean"))
    covariance === nothing || (covariance_bandwidth === nothing && diagonal_bandwidth === nothing) ||
        throw(ArgumentError("covariance_bandwidth and diagonal_bandwidth cannot be used with a supplied covariance; let FPCA estimate the covariance to tune its bandwidths"))
    covariance === nothing || binning === :auto || throw(ArgumentError(
        "binning cannot be used with a supplied covariance; binning only affects covariance estimation"))
    smoothing = mean === nothing || covariance === nothing
    if !smoothing
        kernel === nothing || throw(ArgumentError(
            "kernel cannot be used with a supplied mean and covariance; no smoother is estimated"))
        weighting == :obs || throw(ArgumentError(
            "weighting cannot be used with a supplied mean and covariance; no smoother is estimated"))
    end
    grid_source = grid === nothing ? :observed_range_51 : :supplied
    grid = grid === nothing ? _pace_default_grid(fd) : grid
    _pace_input_grid(grid, "PACE work")
    workgrid = _evaluation_points(fd, grid)
    weights = quadrature_weights(workgrid)
    _pace_observations(fd, workgrid)
    weighting in (:obs, :subj) || throw(ArgumentError("weighting must be :obs or :subj"))
    requested_kernel = deepcopy(kernel)
    kernel = kernel === nothing ? KernelSmoothing.GaussianKernel() : kernel
    kernel isa KernelSmoothing.AbstractKernel ||
        throw(ArgumentError("kernel must be a KernelSmoothing.AbstractKernel"))
    requested_binning = binning
    binning, binning_rule = covariance === nothing ? _pace_binning(fd, binning) :
        (nothing, :not_applicable)
    a, b = _pace_observed_range(fd)
    mean_bandwidth_source = mean === nothing ? (mean_bandwidth === nothing ? :default_fraction : :supplied) : :mean_input
    covariance_bandwidth_source = covariance === nothing ? (covariance_bandwidth === nothing ? :default_fraction : :supplied) : :covariance_input
    requested_mean_bandwidth = deepcopy(mean_bandwidth)
    requested_covariance_bandwidth = deepcopy(covariance_bandwidth)
    mean === nothing && mean_bandwidth === nothing && (mean_bandwidth = _pace_default_mean_bandwidth(a, b))
    covariance === nothing && covariance_bandwidth === nothing && (covariance_bandwidth = _pace_default_covariance_bandwidth(a, b))
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    requested_max_k = max_k
    max_k === nothing && (max_k = max(1, min(20, length(workgrid)-2, nobs(fd)-2)))
    _integer_in(max_k, 1, length(workgrid)) ||
        throw(ArgumentError("PACE max_k must be between 1 and the work-grid length"))
    resolved_mean, mean_source = _pace_mean(fd, mean, workgrid, mean_bandwidth, kernel, weighting)
    resolved_covariance, covariance_source = _pace_covariance(fd, resolved_mean,
        covariance, workgrid, covariance_bandwidth, diagonal_bandwidth, kernel, weighting, binning)
    sigma2, noise_source = _pace_noise(resolved_covariance, noise_variance)
    mean_values = _pace_mean_values(resolved_mean, workgrid)
    surface = _pace_surface(resolved_covariance, workgrid)
    T = promote_type(eltype(workgrid), eltype(mean_values), eltype(surface), typeof(sigma2))
    workgrid, weights = Vector{T}(workgrid), Vector{T}(weights)
    mu, surface, sigma2 = Vector{T}(mean_values), Matrix{T}(surface), T(sigma2)
    all(isfinite, mu) && all(isfinite, surface) && isfinite(sigma2) ||
        throw(ArgumentError("PACE mean, covariance, and noise variance must remain finite after conversion"))
    system = _fpca_eigen(surface, weights, mu; max_k, eigen_tol, sign_tol)
    selection = _pace_selection(ncomponents, fd, workgrid, mu, system, Int(max_k), sigma2)
    k = selection.chosen
    projected = (system.eigenfunctions .* transpose(system.eigenvalues)) *
        transpose(system.eigenfunctions)
    projected = (projected + transpose(projected))/2
    basis = FPCBasis(workgrid, system.eigenfunctions[:, 1:k]; weights)
    predicted = _pace_ce(fd, workgrid, mu, basis.values, system.eigenvalues[1:k], projected, sigma2)
    settings = merge((; requested_method=method, grid_source, requested_max_k, requested_maxK=requested_max_k, max_k=Int(max_k), maxK=Int(max_k),
        ncomponents=deepcopy(ncomponents), observed_range=(a, b), observations=:copied_original,
        eigen_tol, sign_tol, quadrature=:work_grid_trapezoidal,
        mean_source, covariance_source, noise_variance_source=noise_source,
        mean_input=resolved_mean, covariance_input=resolved_covariance,
        mean_settings=_pace_mean_settings(resolved_mean),
        covariance_settings=_pace_covariance_settings(resolved_covariance),
        mean_bandwidth=deepcopy(mean_bandwidth), covariance_bandwidth=deepcopy(covariance_bandwidth),
        requested_mean_bandwidth, requested_covariance_bandwidth, mean_bandwidth_source, covariance_bandwidth_source,
        diagonal_bandwidth=deepcopy(diagonal_bandwidth),
        requested_kernel, kernel=smoothing ? deepcopy(kernel) : nothing,
        weighting=smoothing ? weighting : nothing, requested_binning, binning_rule,
        mean_interpolation=:piecewise_linear, eigenfunction_interpolation=:piecewise_linear,
        covariance_interpolation=:bilinear, score_method=:conditional_expectation,
        covariance_components=length(system.eigenvalues), full_projected_covariance=projected,
        likelihood=:gaussian, likelihood_constant=:included,
        noise_floor=nothing, regularization=nothing),
        # The automatic decision is data-dependent. Declare the union field
        # explicitly so the result type does not depend on which count wins.
        NamedTuple{(:binning,),Tuple{Union{Nothing,Int}}}((binning,)))
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry, tolerance=sign_tol)
    FPCA(:pace, sampling_design(fd), workgrid, weights,
        GridFunctionalData(reshape(mu, 1, :), workgrid), surface, sigma2,
        system.eigenvalues[1:k], system.eigenvalues, system.fve, basis, selection,
        system.dropped, sign_rule, predicted.scores, predicted.covariances,
        deepcopy(fd.ids), nothing, settings, _convert_component(T, fd))
end

function _pace_predict(f::FPCA, fd::Union{GridFunctionalData,IrregularFunctionalData})
    _pace_observations(fd, f.grid)
    _pace_ce(fd, f.grid, vec(f.mean.values), f.basis.values, f.eigenvalues,
        f.settings.full_projected_covariance, f.noise_variance)
end
