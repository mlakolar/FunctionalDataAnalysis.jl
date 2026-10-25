# A lazy covariance surface: fitting and scoring never allocate a grid-square
# matrix. Plotting or Matrix(surface) materializes it only when requested.
struct _LowRankCovariance{T} <: AbstractMatrix{T}
    factor::Matrix{T}
end
Base.size(c::_LowRankCovariance) = (size(c.factor, 1), size(c.factor, 1))
Base.getindex(c::_LowRankCovariance, i::Int, j::Int) = dot(view(c.factor, i, :), view(c.factor, j, :))
Base.copy(c::_LowRankCovariance) = _LowRankCovariance(copy(c.factor))
Base.Matrix(c::_LowRankCovariance) = c.factor * transpose(c.factor)

Base.@constprop :aggressive function _face_mean(fd, ::Nothing, center, smoother)
    T = eltype(fd)
    if !center
        smoother === nothing || throw(ArgumentError("FACE mean_smoother requires center=true"))
        return zeros(T, length(fd.grid)), (; source=:zero, smoothing=nothing)
    end
    observed = fd.observed
    mu = T[sum(fd.values[i, j] for i in 1:nobs(fd)) /
        (observed === nothing ? nobs(fd) : count(view(observed, :, j))) for j in eachindex(fd.grid)]
    all(isfinite, mu) || throw(ArgumentError("FACE mean requires at least one observed value in every grid column"))
    if smoother === nothing
        # Use the existing smoother with bounded dimension to preserve FACE's
        # linear dependence on grid length. This is not a P-spline penalty.
        length(fd.grid) >= 4 || throw(ArgumentError("FACE default mean smoother requires at least four grid points; supply mean or center=false"))
        basis = BSplineBasis(4, (first(fd.grid), last(fd.grid)), min(38, length(fd.grid)))
        width = last(fd.grid)-first(fd.grid)
        smoother = PenalizedSpline(basis; lambda=GCV(exp.(range(-20.0, 20.0; length=100)) .* width^3))
    end
    smoother isa PenalizedSpline || throw(ArgumentError("FACE mean_smoother must be a PenalizedSpline"))
    smoother.order == 2 || throw(ArgumentError("FACE mean_smoother must use integrated derivative order 2"))
    fitted = smooth_curves(GridFunctionalData(reshape(mu, 1, :), fd.grid), smoother)
    Vector{T}(vec(evaluate(fitted, fd.grid))), (; source=:estimated, smoothing=deepcopy(fitted.metadata))
end

function _face_mean(fd, mean, center, smoother)
    center || throw(ArgumentError("FACE supplied mean requires center=true"))
    smoother === nothing || throw(ArgumentError("FACE mean_smoother cannot be used with a supplied mean"))
    supplied, _ = _pace_mean(fd, mean, fd.grid, nothing, nothing, :obs)
    Vector{eltype(fd)}(_pace_mean_values(supplied, fd.grid)),
        (; source=:supplied, smoothing=nothing, input=supplied)
end

function _face_initialize(fd, mu)
    Y = fd.values .- transpose(mu)
    initialization = NamedTuple{(:curve, :lambda, :spar, :knots),Tuple{Int,Float64,Float64,Vector{Float64}}}[]
    if fd.observed !== nothing
        for i in 1:nobs(fd)
            keep = findall(view(fd.observed, i, :))
            isempty(keep) && throw(ArgumentError("FACE curve $i is fully missing"))
            absent = findall(!, view(fd.observed, i, :))
            isempty(absent) && continue
            length(keep) >= 4 || throw(ArgumentError("FACE curve $i needs at least four observations for missing-value initialization"))
            initial = _face_initial_curve(Float64.(fd.grid[keep]), Float64.(Y[i, keep]), Float64.(fd.grid[absent]))
            Y[i, absent] = initial.values
            push!(initialization, (; curve=i, initial.lambda, initial.spar, initial.knots))
        end
    end
    Y, initialization
end

function _face_gcv(Y, transformed, s, candidates, alpha, missing_fraction)
    J = size(Y, 2)
    diagonal = vec(sum(abs2, transformed; dims=2))
    total, projected = sum(abs2, Y), sum(diagonal)
    map(candidates) do lambda
        h = inv.(1 .+ lambda .* s)
        numerator = sum(diagonal .* (1 .- h).^2) - projected + total
        denominator = 1-alpha*sum(h)/J/(1-missing_fraction)
        denominator == 0 ? oftype(total, Inf) : numerator/denominator^2
    end
end

# Precision-form solve for the selected FACE model, called explicitly by FACE
# fitting and prediction. It requires the observation covariance phi*Lambda*phi'
# and must not be selected by the storage type: DPC uses a different numerator
# with the full projected observation covariance.
function _face_noiseless_factor(phi, k, curve, ::Type{T}) where {T}
    factor = qr(phi, ColumnNorm())
    diagonal = abs.(diag(factor.R))
    length(diagonal) == k && minimum(diagonal) > max(size(phi)...)*eps(T)*maximum(diagonal) ||
        throw(ArgumentError("FACE noiseless score design for curve $curve is rank deficient"))
    factor
end

function _face_precision_geometry(phi, values, sigma2, curve, ::Type{T}) where {T}
    precision = transpose(phi)*phi + Diagonal(sigma2 ./ values)
    factor = _pace_cholesky(precision, curve)
    omega = sigma2 .* (factor \ Matrix{T}(I, length(values), length(values)))
    factor, (omega+transpose(omega))/2
end

function _face_curve_ce(times, residuals, phi, values, grid, projected,
        sigma2, curve, ::Type{T}) where {T}
    if sigma2 == 0
        factor = _face_noiseless_factor(phi, length(values), curve, T)
        return factor \ residuals, zeros(T, length(values), length(values))
    end
    factor, omega = _face_precision_geometry(phi, values, sigma2, curve, T)
    factor \ (transpose(phi)*residuals), omega
end

function _face_common_scores(fd, grid, mu, functions, values, projected, sigma2)
    T = promote_type(eltype(grid), eltype(mu), eltype(functions), eltype(values),
        eltype(projected), typeof(sigma2), eltype(fd))
    # Interpolation at the identical knots returns these same basis values.
    # Keep the type the former interpolation would produce before its Gram.
    P = promote_type(eltype(grid), eltype(functions), eltype(fd))
    phi = convert(Matrix{P}, functions)
    residuals = fd.values .- transpose(convert(Vector{T}, mu))
    k, n = length(values), nobs(fd)
    if sigma2 == 0
        factor = _face_noiseless_factor(phi, k, 1, T)
        output = Matrix{T}(transpose(factor \ transpose(residuals)))
        omega = zeros(T, k, k)
    else
        factor, omega = _face_precision_geometry(phi, values, sigma2, 1, T)
        output = Matrix{T}(transpose(factor \ (transpose(phi)*transpose(residuals))))
    end
    covariances = repeat(reshape(Matrix{T}(omega), k, k, 1), 1, 1, n)
    all(isfinite, output) && all(isfinite, covariances) || throw(ArgumentError(
        "PACE scores and score covariances must be finite; rescale the observations"))
    (; scores=output, covariances)
end

function _face_scores(fd, grid, mu, functions, values, projected, sigma2)
    if fd isa GridFunctionalData && fd.observed === nothing && fd.grid == grid
        return _face_common_scores(fd, grid, mu, functions, values, projected, sigma2)
    end
    _pace_ce(fd, grid, mu, functions, values, projected, sigma2; curve_ce=_face_curve_ce)
end

Base.@constprop :aggressive function _fit_face(fd::GridFunctionalData{T}; ncomponents=FVE(0.99), max_k=nothing,
        mean=nothing, eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL,
        knots=35, p=3, m=2, lambda=nothing, alpha=1, center=true,
        loglambda=collect(range(-20.0, 20.0; length=100)), mean_smoother=nothing,
        kwargs...) where {T}
    isempty(kwargs) || throw(ArgumentError("unknown FACE settings: $(keys(kwargs))"))
    n, J = size(fd.values)
    n >= 2 || throw(ArgumentError("FACE requires at least two curves"))
    center isa Bool || throw(ArgumentError("FACE center must be true or false"))
    alpha isa Real && !(alpha isa Bool) && isfinite(alpha) && alpha > 0 || throw(ArgumentError("FACE alpha must be finite and positive"))
    isfinite(T(alpha)) && T(alpha) > 0 || throw(ArgumentError("FACE alpha must remain finite and positive after conversion"))
    lambda === nothing || (lambda isa Real && !(lambda isa Bool) && isfinite(lambda) && lambda >= 0) ||
        throw(ArgumentError("FACE lambda must be finite and nonnegative, or nothing for GCV"))
    loglambda isa AbstractVector{<:Real} && !isempty(loglambda) && all(isfinite, loglambda) ||
        throw(ArgumentError("FACE loglambda must be a nonempty finite real vector"))
    Base.require_one_based_indexing(loglambda)
    candidates = T.(exp.(loglambda))
    all(x -> isfinite(x) && x > 0, candidates) && length(unique(candidates)) == length(candidates) ||
        throw(ArgumentError("FACE loglambda must produce distinct finite positive lambdas"))
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    ncomponents isa Union{AIC,BIC} && throw(ArgumentError("AIC and BIC component selection require method=:pace"))
    ncomponents isa FVE || (_integer_in(ncomponents, 1)) ||
        throw(ArgumentError("FACE ncomponents must be FVE or a positive integer representable as Int"))
    setting = _pspline_setting(fd.grid, knots, p, m)
    c = size(setting.A, 2)
    cap = min(n, c)
    max_k = max_k === nothing ? cap : max_k
    _integer_in(max_k, 1, cap) || throw(ArgumentError("FACE max_k must be between 1 and min(nobs, knots+p)"))
    mu, mean_settings = _face_mean(fd, mean, center, mean_smoother)
    Y, initialization = _face_initialize(fd, mu)
    all(isfinite, Y) || throw(ArgumentError("FACE centered observations must be finite; rescale the observations"))
    observed = fd.observed
    missing_fraction = observed === nothing ? zero(T) : T(count(!, observed)/length(observed))
    transformed = transpose(setting.A)*transpose(Y)
    gcv_scores = _face_gcv(Y, transformed, setting.s, candidates, alpha, missing_fraction)
    selected = lambda === nothing ? candidates[argmin(gcv_scores)] : T(lambda)
    isfinite(selected) && selected >= 0 || throw(ArgumentError("FACE lambda must remain finite after conversion"))
    lambda === nothing && !isfinite(minimum(gcv_scores)) && throw(ArgumentError("FACE GCV has no finite candidate"))
    shrink = inv.(1 .+ selected .* setting.s)
    changes = T[]
    converged = observed === nothing
    previous = observed === nothing ? Matrix{T}(undef, 0, 0) : similar(transformed)
    difference = similar(previous)
    smoothed = similar(transformed)
    absent = observed === nothing ? CartesianIndex{2}[] : findall(!, observed)
    completed = Matrix{T}(undef, observed === nothing ? 0 : J, observed === nothing ? 0 : n)
    # Observed entries never change during imputation. Preserve the original
    # packed-vector reduction, but compute it once rather than each iteration.
    observed_mean_square = observed === nothing ? sum(abs2, Y)/length(Y) :
        sum(abs2, Y[observed])/count(observed)
    iterations = 0
    sigma2 = zero(T)
    noise_settings = nothing
    # Refund updates the missing values once AFTER checking convergence of YS.
    # The final eigensystem is from YS, while final scores use the updated Y.
    for iteration in 1:(observed === nothing ? 1 : 100)
        iterations = iteration
        iteration > 1 && mul!(transformed, transpose(setting.A), transpose(Y))
        smoothed .= shrink .* transformed
        all(isfinite, smoothed) || throw(ArgumentError("FACE smoothed data must be finite; rescale the observations"))
        raw_sigma2 = observed_mean_square - sum(abs2, smoothed)/(n*J)
        # Evaluate diag(A*Cov(smoothed)*A') without a grid-square matrix.
        signal_covariance = smoothed*transpose(smoothed)/n
        signal_variance = maximum(sum((setting.A*signal_covariance) .* setting.A; dims=2))
        noise_settings = _fpca_estimated_noise_settings(raw_sigma2, signal_variance)
        sigma2 = max(raw_sigma2, noise_settings.noise_floor)
        if observed !== nothing
            system = _fpca_eigen(smoothed*transpose(smoothed)/n, ones(T, c), zeros(T, c);
                max_k, eigen_tol, sign_tol)
            selection = _fpca_selection(ncomponents, system.fve, Int(max_k))
            U = system.eigenfunctions[:, 1:selection.chosen]
            d = system.eigenvalues[1:selection.chosen]
            mul!(completed, setting.A, U * ((d ./ (d .+ sigma2)) .* (transpose(U)*transformed)))
            for index in absent
                Y[index] = completed[index[2], index[1]]
            end
            if iteration > 1
                denominator = norm(smoothed)
                difference .= smoothed .- previous
                change = denominator == 0 ? zero(T) : norm(difference)/denominator
                push!(changes, change)
                converged = change <= T(0.02)
            end
            copyto!(previous, smoothed)
        end
        converged && break
    end
    # Weighted eigenanalysis is only c-by-c. QR of sqrt(W)*A transfers the
    # stored quadrature metric into that small covariance matrix.
    R = Matrix(qr(sqrt.(fd.weights) .* setting.A).R)
    small = R * (smoothed*transpose(smoothed)/n) * transpose(R)
    small = (small+transpose(small))/2
    system = _fpca_eigen(small, ones(T, c), zeros(T, c); max_k, eigen_tol, sign_tol)
    functions = setting.A * (R \ system.eigenfunctions)
    _fpca_orient!(functions, mu, sign_tol)
    selection = _fpca_selection(ncomponents, system.fve, Int(max_k))
    k = selection.chosen
    basis = FPCBasis(fd.grid, functions[:, 1:k]; weights=fd.weights)
    values = system.eigenvalues[1:k]
    # H' = Q*R implies H*H' = R'*R. Keep at most c columns, including all
    # smoothed covariance directions regardless of max_k or eigen_tol.
    covariance_root = Matrix(qr(transpose(smoothed)).R)
    covariance = _LowRankCovariance(setting.A * transpose(covariance_root) / sqrt(T(n)))
    projected = _LowRankCovariance(functions .* transpose(sqrt.(system.eigenvalues)))
    # Scores use completed curves as refund does; pointwise uncertainty uses
    # the same selected-K model and the same completed observation design.
    completed_fd = GridFunctionalData(Y .+ transpose(mu), fd.grid)
    scored = _face_scores(completed_fd, fd.grid, mu, basis.values, values, projected, sigma2)
    settings = merge((; requested_method=:face, grid_source=:data, ncomponents=deepcopy(ncomponents),
        max_k=Int(max_k), maxK=Int(max_k), eigen_tol, sign_tol, knots=Int(knots), p=Int(p), m=Int(m),
        knot_sequence=copy(setting.knots), lambda=selected, lambda_source=lambda === nothing ? :GCV : :supplied,
        loglambda=T.(loglambda), gcv_lambdas=candidates, gcv_scores,
        alpha=T(alpha), center, mean=mean_settings, covariance_denominator=n,
        quadrature=:stored_trapezoidal, score_method=:conditional_expectation,
        noise_scale=:observed_mean_square_minus_euclidean_trace_over_grid_length,
        noise_variance_source=:estimated,
        design_eigenvalue_floor=T(1e-7), score_design=:completed_training_observed_prediction,
        full_projected_covariance=projected, smoothing_spectrum=copy(setting.s),
        imputation=(; iterations, converged, relative_changes=changes, tolerance=T(0.02),
            maxiter=100, initialization, completed_centered=copy(Y),
            missing_fraction, lambda_refitted=false,
            initialization_rule=(; reference=:R_stats_4_5_1, penalty_quadrature=0.3330,
                knot_rule=:smooth_spline_default, spar_bounds=(-1.5, 1.5),
                spar_tolerance=1e-4, spar_epsilon=2e-8, spar_maxiter=500,
                spar_rule=:last_evaluated_R_fit,
                extrapolation=:observed_centered_mean))), noise_settings)
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry, tolerance=sign_tol)
    FPCA(:face, sampling_design(fd), copy(fd.grid), copy(fd.weights),
        GridFunctionalData(reshape(mu, 1, :), fd.grid), covariance, sigma2,
        values, system.eigenvalues, system.fve, basis, selection, system.dropped,
        sign_rule, scored.scores, scored.covariances, deepcopy(fd.ids), nothing,
        settings, _convert_component(T, fd))
end

function _face_predict(f::FPCA, fd::GridFunctionalData)
    fd.grid == f.grid || throw(ArgumentError("FACE prediction grid must exactly match the training grid"))
    if fd.values == f.training_data.values && fd.observed == f.training_data.observed
        return (; scores=copy(f.scores), covariances=copy(f.score_covariances))
    end
    _face_scores(fd, f.grid, vec(f.mean.values), f.basis.values, f.eigenvalues,
        f.settings.full_projected_covariance, f.noise_variance)
end
