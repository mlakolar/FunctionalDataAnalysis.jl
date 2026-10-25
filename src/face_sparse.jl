# Sparse FACE, audited against face 0.1-8. The covariance is a symmetric
# tensor-product P-spline fitted jointly with the diagonal measurement noise.
# Keep spline evaluation exact at observation times, including new times.
struct _SplineCovariance{T,B} <: AbstractMatrix{T}
    factor::Matrix{T}
    basis::B
    coefficient_root::Matrix{T}
end
Base.size(c::_SplineCovariance) = (size(c.factor, 1), size(c.factor, 1))
Base.getindex(c::_SplineCovariance, i::Int, j::Int) = dot(view(c.factor, i, :), view(c.factor, j, :))
Base.copy(c::_SplineCovariance) = _SplineCovariance(copy(c.factor), deepcopy(c.basis), copy(c.coefficient_root))
Base.Matrix(c::_SplineCovariance) = c.factor * transpose(c.factor)
function _pace_observation_covariance(grid, c::_SplineCovariance, times)
    root = basis_matrix(c.basis, times)*c.coefficient_root
    root*transpose(root)
end

function _face_sparse_lambdas(loglambda, ::Type{T}, label="loglambda") where {T}
    loglambda isa AbstractVector{<:Real} && !isempty(loglambda) && all(isfinite, loglambda) ||
        throw(ArgumentError("sparse FACE $label must be a nonempty finite real vector"))
    Base.require_one_based_indexing(loglambda)
    candidates = T.(exp.(loglambda))
    all(x -> isfinite(x) && x > 0, candidates) && length(unique(candidates)) == length(candidates) ||
        throw(ArgumentError("sparse FACE $label must produce distinct finite positive lambdas"))
    candidates
end

function _face_sparse_lambda(lambda, ::Type{T}, label) where {T}
    lambda === nothing && return nothing
    lambda isa Real && !(lambda isa Bool) && isfinite(lambda) && lambda >= 0 &&
        isfinite(T(lambda)) || throw(ArgumentError("sparse FACE $label must be finite and nonnegative, or nothing"))
    T(lambda)
end

# Demmler-Reinsch coordinates, using the two reference ridge rules explicitly.
function _face_sparse_gram(X, weights=nothing)
    # Form ill-conditioned normal matrices in at least double precision, even
    # for Float32 input. The fitted arrays retain the input floating type.
    W = promote_type(eltype(X), Float64)
    A = Matrix{W}(X)
    transpose(A)*(weights === nothing ? A : W.(weights) .* A)
end

function _face_sparse_coordinates(gram, penalty; covariance=false, nullity=0)
    T, W = eltype(penalty), eltype(gram)
    e = eigen(Symmetric(gram))
    ridge = covariance ? W(1e-6)*maximum(e.values) :
        (minimum(e.values) <= W(1e-7) ? W(1e-6) : zero(W))
    adjusted = e.values .+ ridge
    all(x -> isfinite(x) && x > 0, adjusted) || throw(ArgumentError(
        "sparse FACE penalized design is singular; reduce knots or degree"))
    root = e.vectors * Diagonal(inv.(sqrt.(adjusted))) * transpose(e.vectors)
    pe = eigen(Symmetric(root*penalty*root))
    s = pe.values
    nullity > 0 && (s[1:nullity] .= 0)
    (; A0=Matrix{T}(root*pe.vectors), s=Vector{T}(s), ridge=T(ridge))
end

# face's 'gcv': first-order subject-deletion approximation, with y'y omitted.
# S_ii = w_i A_i diag(d) A_i'; no row-by-row CV or trace denominator.
function _face_subject_gcv(A, y, groups, subject_weights, s, candidates)
    T = eltype(A)
    f = zeros(T, size(A, 2))
    moments = map(eachindex(groups)) do i
        Ai, yi = view(A, groups[i], :), view(y, groups[i])
        local_gram, local_rhs = transpose(Ai)*Ai, transpose(Ai)*yi
        f .+= subject_weights[i] .* local_rhs
        (; gram=local_gram, rhs=local_rhs)
    end
    gram = transpose(A)*A
    rhs = transpose(A)*y
    criterion = map(candidates) do lambda
        d = inv.(1 .+ lambda .* s)
        a = d .* f
        value = dot(a, gram*a) - 2dot(a, rhs)
        for i in eachindex(groups)
            residual_rhs = moments[i].rhs - moments[i].gram*a
            value += 2subject_weights[i]*sum(d .* residual_rhs.^2)
        end
        value
    end
    (; scores=criterion, rhs=f)
end

_face_sparse_mean_values(mean, times) = mean.source === :zero ? zeros(eltype(times), length(times)) :
    mean.source === :supplied ? _pace_mean_values(mean.input, times) :
    basis_matrix(mean.basis, times)*mean.coefficients

function _face_sparse_mean(fd, mean, center, knots, lambda_mean, mean_loglambda, grid)
    T = eltype(fd)
    if mean !== nothing || !center
        mean_loglambda == collect(range(-20.0, 20.0; length=100)) || throw(ArgumentError(
            "sparse FACE mean_loglambda requires an estimated mean with center=true"))
    end
    if mean !== nothing
        center || throw(ArgumentError("sparse FACE supplied mean requires center=true"))
        lambda_mean === nothing || throw(ArgumentError("sparse FACE lambda_mean cannot be used with a supplied mean"))
        supplied, _ = _pace_mean(fd, mean, grid, nothing, nothing, :obs)
        return (; source=:supplied, input=supplied)
    end
    if !center
        lambda_mean === nothing || throw(ArgumentError("sparse FACE lambda_mean requires center=true"))
        return (; source=:zero)
    end
    candidates = _face_sparse_lambdas(mean_loglambda, T, "mean_loglambda")
    selected = _face_sparse_lambda(lambda_mean, T, "lambda_mean")
    a, b = extrema(fd.times)
    width = b-a
    # face.sparse passes its knot count to pspline but does not pass p or m.
    design = _pspline_design(fd.times, knots, 3, 2; interval=(a-T(.001)*width, b+T(.001)*width))
    groups = [fd.offsets[i]:(fd.offsets[i+1]-1) for i in 1:nobs(fd)]
    wi = T[inv(length(g)) for g in groups]
    weights = reduce(vcat, [fill(wi[i], length(groups[i])) for i in eachindex(groups)])
    B = design.B
    coordinates = _face_sparse_coordinates(_face_sparse_gram(B, weights), design.P; nullity=2)
    A = B*coordinates.A0
    cv = _face_subject_gcv(A, fd.values, groups, wi, coordinates.s, candidates)
    selected === nothing && (selected = candidates[argmin(cv.scores)])
    coefficients = coordinates.A0*(cv.rhs ./ (1 .+ selected .* coordinates.s))
    all(isfinite, cv.scores) && all(isfinite, coefficients) || throw(ArgumentError("sparse FACE mean fit must be finite; rescale the observations"))
    (; source=:estimated, basis=design.basis, coefficients, knots=Int(knots), p=3, m=2,
        knot_sequence=design.knots, lambda=selected,
        lambda_source=lambda_mean === nothing ? :subject_CV : :supplied,
        loglambda=T.(mean_loglambda), gcv_lambdas=candidates, gcv_scores=cv.scores,
        weighting=:inverse_observations_per_subject, ridge=coordinates.ridge,
        penalty_nullity=2, normal_matrix_type=promote_type(T,Float64))
end

Base.@constprop :aggressive function _fit_face_sparse(fd::IrregularFunctionalData{T}; grid=nothing,
        ncomponents=FVE(.99), max_k=nothing, mean=nothing,
        eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL,
        knots=7, p=3, m=2, lambda=nothing, center=true,
        loglambda=collect(range(-3.0, 10.0; length=14)), lambda_mean=nothing,
        mean_loglambda=collect(range(-20.0, 20.0; length=100)), kwargs...) where {T}
    isempty(kwargs) || throw(ArgumentError("unknown sparse FACE settings: $(keys(kwargs))"))
    nobs(fd) >= 2 || throw(ArgumentError("sparse FACE requires at least two curves"))
    center isa Bool || throw(ArgumentError("sparse FACE center must be true or false"))
    candidates = _face_sparse_lambdas(loglambda, T)
    _face_sparse_lambdas(mean_loglambda, T, "mean_loglambda")
    selected = _face_sparse_lambda(lambda, T, "lambda")
    _face_sparse_lambda(lambda_mean, T, "lambda_mean")
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    ncomponents isa FVE || (_integer_in(ncomponents, 1)) ||
        throw(ArgumentError("sparse FACE ncomponents must be FVE or a positive integer representable as Int"))
    a, b = extrema(fd.times)
    a < b || throw(ArgumentError("sparse FACE needs at least two distinct observation times"))
    grid_source = grid === nothing ? :observed_range_100 : :supplied
    workgrid = grid === nothing ? collect(range(a, b; length=100)) : T.(_pace_input_grid(grid, "sparse FACE work"))
    weights = quadrature_weights(workgrid)
    _pace_observations(fd, workgrid)
    _pace_check_coverage([a,b], workgrid, "sparse FACE observed range")
    width = b-a
    design = _pspline_design(fd.times, knots, p, m; interval=(a-T(.001)*width, b+T(.001)*width))
    c = size(design.B, 2)
    cap = min(c, length(workgrid))
    max_k = max_k === nothing ? cap : max_k
    _integer_in(max_k, 1, cap) || throw(ArgumentError("sparse FACE max_k must be between 1 and min(knots+p, grid length)"))
    mean_settings = _face_sparse_mean(fd, mean, center, knots, lambda_mean, mean_loglambda, workgrid)
    mu = Vector{T}(_face_sparse_mean_values(mean_settings, workgrid))
    residuals = fd.values - _face_sparse_mean_values(mean_settings, fd.times)
    all(isfinite, residuals) || throw(ArgumentError("sparse FACE residuals must be finite; rescale observations"))
    pairs = [(j,k) for k in 1:c for j in k:c]
    q = length(pairs)
    # Duplication matrix maps the lower triangle to a symmetric coefficient matrix.
    G = zeros(T, c*c, q)
    for (l,(j,k)) in enumerate(pairs)
        G[j+(k-1)*c,l] = 1
        G[k+(j-1)*c,l] = 1
    end
    group_sizes = [length(fd[i][1])*(length(fd[i][1])+1)÷2 for i in 1:nobs(fd)]
    X = zeros(T, sum(group_sizes), q+1)
    y = Vector{T}(undef, size(X,1))
    groups = UnitRange{Int}[]
    row = 0
    for i in 1:nobs(fd)
        indices = fd.offsets[i]:(fd.offsets[i+1]-1)
        push!(groups, (row+1):(row+group_sizes[i]))
        for k in indices, j in k:last(indices)
            row += 1
            for (l,(u,v)) in enumerate(pairs)
                X[row,l] = design.B[j,u]*design.B[k,v]
                u != v && (X[row,l] += design.B[j,v]*design.B[k,u])
            end
            X[row,end] = j == k
            y[row] = residuals[j]*residuals[k]
        end
    end
    all(isfinite, y) || throw(ArgumentError("sparse FACE raw covariance must be finite; rescale observations"))
    Q = zeros(T, q+1, q+1)
    Q[1:q,1:q] = transpose(G)*kron(Matrix{T}(I,c,c), design.P)*G
    coordinates = _face_sparse_coordinates(_face_sparse_gram(X), Q; covariance=true,
        nullity=Int(m)*(Int(m)+1)÷2+1)
    cv = _face_subject_gcv(X*coordinates.A0, y, groups, ones(T,nobs(fd)), coordinates.s, candidates)
    selected === nothing && (selected = candidates[argmin(cv.scores)])
    alpha = coordinates.A0*(cv.rhs ./ (1 .+ selected .* coordinates.s))
    all(isfinite, cv.scores) && all(isfinite, alpha) || throw(ArgumentError("sparse FACE covariance fit must be finite; rescale observations"))
    raw_sigma2 = alpha[end]
    sigma2 = max(raw_sigma2, zero(T))
    theta = reshape(G*alpha[1:q], c, c)
    ce = eigen(Symmetric(theta))
    positive = findall(>(zero(T)), ce.values)
    isempty(positive) && throw(ArgumentError("sparse FACE covariance has no positive eigenvalues"))
    coefficient_root = ce.vectors[:,positive] .* transpose(sqrt.(ce.values[positive]))
    Bnew = basis_matrix(design.basis, workgrid)
    # Transfer the stored quadrature into a small covariance eigensystem.
    gram = transpose(Bnew)*(weights .* Bnew)
    ge = eigen(Symmetric(gram))
    minimum(ge.values) > eps(T)*maximum(ge.values)*c || throw(ArgumentError("sparse FACE work-grid spline design is rank deficient; use a richer grid or fewer knots"))
    root = ge.vectors*Diagonal(sqrt.(ge.values))*transpose(ge.vectors)
    inverse_root = ge.vectors*Diagonal(inv.(sqrt.(ge.values)))*transpose(ge.vectors)
    smallroot = root*coefficient_root
    small = smallroot*transpose(smallroot)
    system = _fpca_eigen((small+transpose(small))/2, ones(T,c), zeros(T,c); max_k, eigen_tol, sign_tol)
    coefficients = inverse_root*system.eigenfunctions
    functions = Bnew*coefficients
    _fpca_orient!(functions, mu, sign_tol)
    # Transfer the shared sign convention to the coefficients as well.
    for j in axes(functions,2)
        dot(functions[:,j], Bnew*coefficients[:,j]) < 0 && (coefficients[:,j] .*= -1)
    end
    selection = _fpca_selection(ncomponents, system.fve, Int(max_k))
    k = selection.chosen
    basis = FPCBasis(workgrid, functions[:,1:k]; weights)
    covariance = _SplineCovariance(Bnew*coefficient_root, design.basis, coefficient_root)
    settings = (; requested_method=:face, route=:sparse, grid_source,
        ncomponents=deepcopy(ncomponents), max_k=Int(max_k), maxK=Int(max_k), eigen_tol, sign_tol,
        knots=Int(knots), p=Int(p), m=Int(m), knot_sequence=design.knots,
        knot_range_expansion=T(.001), lambda=selected,
        lambda_source=lambda === nothing ? :subject_CV : :supplied,
        loglambda=T.(loglambda), gcv_lambdas=candidates, gcv_scores=cv.scores,
        criterion=:first_order_leave_one_subject_out_without_constant,
        pair_weighting=:identity_upper_triangle_including_diagonal,
        covariance_ridge=coordinates.ridge, covariance_ridge_multiplier=T(1e-6),
        penalty_nullity=Int(m)*(Int(m)+1)÷2+1,
        normal_matrix_type=promote_type(T,Float64),
        center, mean=mean_settings, two_step=false,
        raw_noise_variance=raw_sigma2, noise_truncation=:zero,
        coefficient_projection=(; negative_count=count(<(zero(T)),ce.values),
            negative_total=sum(abs,ce.values[ce.values .< 0])),
        quadrature=:stored_trapezoidal, score_method=:conditional_expectation,
        score_covariance=:full_positive_spline_covariance,
        eigenfunction_coefficients=coefficients[:,1:k],
        full_projected_covariance=covariance)
    scored = _face_sparse_scores(fd, workgrid, mu, basis.values,
        system.eigenvalues[1:k], covariance, sigma2, settings)
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry, tolerance=sign_tol)
    FPCA(:face, sampling_design(fd), copy(workgrid), weights,
        GridFunctionalData(reshape(mu,1,:), workgrid), covariance, sigma2,
        system.eigenvalues[1:k], system.eigenvalues, system.fve, basis, selection,
        system.dropped, sign_rule, scored.scores, scored.covariances, deepcopy(fd.ids),
        nothing, settings, _convert_component(T,fd))
end

function _face_sparse_scores(fd, grid, mu, functions, values, covariance, sigma2, settings)
    _pace_ce(fd, grid, mu, functions, values, covariance, sigma2;
        curve_ce=_face_sparse_curve_ce,
        basis_evaluator=(g,f,t)->basis_matrix(covariance.basis,t)*settings.eigenfunction_coefficients,
        mean_evaluator=(g,m,t)->_face_sparse_mean_values(settings.mean,t))
end

_face_sparse_cholesky(covariance, curve) = _pace_cholesky(covariance, curve;
    label="sparse FACE", advice="zero estimated noise can make the spline observation covariance singular; adjust covariance smoothing")

_face_sparse_curve_ce(times, residuals, phi, values, grid, projected, sigma2, curve, ::Type{T}) where {T} =
    _pace_curve_ce(times, residuals, phi, values, grid, projected, sigma2, curve, T;
        covariance_solver=_face_sparse_cholesky)

function _face_sparse_predict(f::FPCA, fd)
    _pace_observations(fd, f.grid)
    _face_sparse_scores(fd, f.grid, vec(f.mean.values), f.basis.values,
        f.eigenvalues, f.settings.full_projected_covariance, f.noise_variance, f.settings)
end
