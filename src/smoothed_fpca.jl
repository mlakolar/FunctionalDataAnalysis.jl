"""
    LambdaCV(lambdas)

Opt-in leave-one-curve-out selection over a copied, nonempty vector of distinct
finite nonnegative lambdas. Ties select the first candidate. Basis FPCA minimizes
the sum of L² orthogonal-projection residuals over leading subspaces 1 through K
(Ramsay and Silverman, 2005, §9.3.3). FPCA requires a positive integer
`ncomponents=K`, keeping the number of subspaces fixed across folds and lambdas.
Functional CCA maximizes the squared correlation of held-out leading variates
(§11.3.3), using the same candidate lambda on both sides. Candidate failures
throw with the candidate and fold identified; no curves or candidates are dropped.
"""
struct LambdaCV{T<:AbstractFloat}
    lambdas::Vector{T}
    function LambdaCV(lambdas::AbstractVector{<:Real})
        validated = try
            GCV(lambdas)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError(replace(err.msg, "GCV" => "LambdaCV")))
        end
        new{eltype(validated.lambdas)}(copy(validated.lambdas))
    end
end
Base.show(io::IO, cv::LambdaCV) = print(io, "LambdaCV(", cv.lambdas, ")")

# An exact eigenbasis with sampled coordinates only for the existing plot layer.
struct _BasisEigenfunctions{T<:AbstractFloat,B<:AbstractBasis} <: AbstractBasis
    basis::B
    coefs::Matrix{T}                 # input-basis coefficients × components
    grid::Vector{T}
    weights::Vector{T}
    values::Matrix{T}
end
Base.eltype(::Type{<:_BasisEigenfunctions{T}}) where {T} = T
nbasis(b::_BasisEigenfunctions) = size(b.coefs, 2)
domain(b::_BasisEigenfunctions) = domain(b.basis)
basis_matrix(b::_BasisEigenfunctions, t::AbstractVector{<:Real}; derivative=0) =
    basis_matrix(b.basis, t; derivative) * b.coefs
penalty(b::_BasisEigenfunctions, order) =
    transpose(b.coefs) * penalty(b.basis, order) * b.coefs

function _basis_lambda(lambda, ::Type{T}) where {T}
    lambda isa Real && isfinite(lambda) && lambda >= 0 ||
        throw(ArgumentError("lambda must be finite and nonnegative, or LambdaCV(lambdas)"))
    value = T(lambda)
    isfinite(value) || throw(ArgumentError("lambda must remain finite in the data element type"))
    lambda > 0 && value == 0 && throw(ArgumentError(
        "positive lambda must remain positive in the data element type"))
    value
end

_default_fpca_penalty(b::BSplineBasis) = max(0, b.order - 2)
_default_fpca_penalty(::FourierBasis) = :harmonic_acceleration
_default_fpca_penalty(::MonomialBasis) = 0
_default_fpca_penalty(b::AbstractBasis) = throw(ArgumentError(
    "basis FPCA requires an explicit penalty for $(typeof(b))"))

function _basis_penalty(b, specification, ::Type{T}) where {T}
    k = nbasis(b)
    matrix = if specification isa AbstractMatrix{<:Real}
        Base.require_one_based_indexing(specification)
        size(specification) == (k, k) || throw(DimensionMismatch(
            "penalty matrix dimensions must equal the number of basis functions"))
        Matrix{T}(specification)
    else
        computed = Matrix{T}(penalty(b, specification))
        # Galerkin assembly can differ by roundoff between the two triangles.
        (computed + transpose(computed)) / 2
    end
    all(isfinite, matrix) && issymmetric(matrix) ||
        throw(ArgumentError("penalty matrix must be finite and symmetric"))
    spectrum = eigvals(Symmetric(matrix))
    minimum(spectrum) >= -100eps(T) * max(one(T), maximum(abs, spectrum)) ||
        throw(ArgumentError("penalty matrix must be positive semidefinite"))
    matrix
end

function _basis_cholesky(matrix, label)
    all(isfinite, matrix) || throw(ArgumentError("$label must be finite; rescale data or lambda"))
    factor = cholesky(Symmetric(matrix); check=false)
    isposdef(factor) || throw(ArgumentError("$label must be positive definite; revise the basis or lambda"))
    factor
end

# Reuse the common eigenvalue cutoff, spectrum accounting, and FVE machinery.
function _basis_pca(residuals::Matrix{T}, G, P, lambda, max_k, eigen_tol,
        sign_tol) where {T}
    n, k = size(residuals)
    U = _basis_cholesky(G + lambda * P, "FPCA penalized Gram matrix").U
    J = transpose(U) \ G
    covariance = transpose(residuals) * residuals / n
    transformed = J * covariance * transpose(J)
    transformed = Matrix(Symmetric((transformed + transpose(transformed)) / 2))
    system = _fpca_eigen(transformed, ones(T, k), zeros(T, k); max_k,
        eigen_tol, sign_tol)
    raw = U \ system.eigenfunctions
    norms = sqrt.(vec(sum(raw .* (G * raw); dims=1)))
    all(x -> isfinite(x) && x > 0, norms) ||
        throw(ArgumentError("FPCA harmonic L² norms must be finite and positive"))
    (; system..., coefs=raw ./ transpose(norms), normalization=norms,
        coefficient_covariance=covariance)
end

function _basis_cv_error(err, lambda, i)
    err isa ArgumentError || err isa DimensionMismatch || err isa PosDefException || rethrow(err)
    throw(ArgumentError("LambdaCV candidate $lambda failed in held-out curve $i: $(sprint(showerror, err))"))
end

function _basis_pca_lambda(lambda::Real, residuals::Matrix{T}, G, P,
        rule, max_k, eigen_tol, sign_tol) where {T}
    _basis_lambda(lambda, T), nothing
end
function _basis_pca_lambda(cv::LambdaCV, residuals::Matrix{T}, G, P,
        rule, max_k, eigen_tol, sign_tol) where {T}
    _integer_in(rule, 1) || throw(ArgumentError(
        "FPCA LambdaCV requires a positive integer ncomponents so all folds and candidates use the same subspace count"))
    n = size(residuals, 1)
    n >= 3 || throw(ArgumentError("LambdaCV requires at least three curves"))
    candidates = [_basis_lambda(lam, T) for lam in cv.lambdas]
    all(isfinite, candidates) && length(unique(candidates)) == length(candidates) ||
        throw(ArgumentError("LambdaCV candidates must remain finite and distinct in the data element type"))
    errors = zeros(T, length(candidates))
    # §9.3.3 centers once using the overall mean, before leaving curves out.
    for (c, lam) in enumerate(candidates), i in 1:n
        try
            train = residuals[[j for j in 1:n if j != i], :]
            system = _basis_pca(train, G, P, lam, max_k, eigen_tol, sign_tol)
            chosen = _fpca_selection(rule, system.fve, max_k).chosen
            for m in 1:chosen
                H = system.coefs[:, 1:m]
                rhs = transpose(H) * G * residuals[i, :]
                projection = H * (_basis_cholesky(transpose(H) * G * H,
                    "FPCA CV subspace Gram matrix") \ rhs)
                r = residuals[i, :] - projection
                errors[c] += max(zero(T), dot(r, G * r))
            end
        catch err
            _basis_cv_error(err, lam, i)
        end
    end
    all(isfinite, errors) || throw(ArgumentError("FPCA LambdaCV scores must be finite; rescale data"))
    chosen = argmin(errors)
    candidates[chosen], (; method=:leave_one_curve_out, candidates, scores=errors,
        chosen, criterion=:sum_subspace_L2_residuals, direction=:minimize,
        centering=:overall_mean, ncomponents=deepcopy(rule), folds=n)
end
_basis_pca_lambda(lambda, args...) = throw(ArgumentError("lambda must be a nonnegative real number or LambdaCV(lambdas)"))

"""
    fit(FPCA, fd::BasisFunctionalData; lambda=0, penalty=nothing,
        center=true, ncomponents=FVE(0.99), max_k=nothing, maxK=nothing, grid=nothing,
        eigen_tol=1e-10, sign_tol=1e-12)

Roughness-penalized basis FPCA, following fda 6.3.0 `pca.fd`/`fdPar` and
Silverman (1996). Solve `G*C*G*h = ν*(G + lambda*P)*h`, where C is the
centered coefficient cross-product divided by n (also when `center=false`).
The default lambda is zero. The default penalty is derivative order
`max(0, order-2)` for B-splines, harmonic acceleration for Fourier bases, and
order zero for monomials, as in `fdPar`. Supply a derivative order, the Fourier
symbol `:harmonic_acceleration`, or a symmetric positive-semidefinite matrix.
`LambdaCV(lambdas)` opts into leave-one-curve-out selection and requires a
positive integer `ncomponents` shared by all folds and candidates.

Eigenvalues and variance proportions keep `pca.fd`'s penalized-metric scale.
Each harmonic is divided by its exact L² norm q; scores are R's scores divided
by q. Thus score variances (denominator n) are ν/q², and positive-lambda
harmonics are individually L²-normalized but generally not L²-orthogonal.
The package's `FVE(0.99)` default departs from R's `nharm=2`; a fixed integer
reproduces `nharm`. The default `max_k=min(nbasis,n-1)` (n when uncentered)
retains the available rank, unlike the grid routes' caps.

The same FPCA result supports exact basis prediction and reconstruction,
and the existing plot-data layer and recipes. `grid` is only a display grid,
defaulting to 201 equally spaced points as in `curve_data`; it never defines
inner products. Mean and eigenfunctions retain exact basis coefficients.
No noise model is assumed; noise and score covariances are `nothing`.
All settings and input copies are retained. See the FPCA page for conversions.
"""
function fit(::Type{FPCA}, fd::BasisFunctionalData{T}; lambda=0, penalty=nothing,
        center=true, ncomponents=FVE(0.99), max_k=nothing, maxK=nothing, grid=nothing,
        eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL) where {T}
    max_k = _max_k_keyword(max_k, maxK)
    center isa Bool || throw(ArgumentError("center must be Boolean"))
    n, k = size(fd.coefs)
    n >= 2 || throw(ArgumentError("basis FPCA requires at least two curves"))
    limit = min(k, center ? n-1 : n)
    cap = max_k === nothing ? limit : max_k
    _integer_in(cap, 1, limit) ||
        throw(ArgumentError("basis FPCA max_k must be between 1 and min(nbasis, centered rank bound)"))
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    b = deepcopy(fd.basis)
    G = Matrix{T}(gram(b))
    specification = penalty === nothing ? _default_fpca_penalty(b) : deepcopy(penalty)
    P = _basis_penalty(b, specification, T)
    mu = center ? vec(sum(fd.coefs; dims=1) / n) : zeros(T, k)
    residuals = fd.coefs .- transpose(mu)
    lam, tuning = _basis_pca_lambda(lambda, residuals, G, P, ncomponents,
        Int(cap), eigen_tol, sign_tol)
    tuning = tuning === nothing ? nothing : merge(tuning,
        (; centering=center ? :overall_mean : :none))
    system = _basis_pca(residuals, G, P, lam, Int(cap), eigen_tol, sign_tol)
    selection = _fpca_selection(ncomponents, system.fve, Int(cap))
    chosen = selection.chosen
    points = grid === nothing ? collect(range(T.(domain(b))...; length=201)) :
        Vector{T}(_basispoints(b, grid))
    weights = quadrature_weights(points)
    X = Matrix{T}(basis_matrix(b, points))
    values = X * system.coefs
    oriented = _fpca_orient!(copy(values), X * mu, sign_tol)
    H = copy(system.coefs)
    for j in axes(H, 2)
        dot(values[:, j], oriented[:, j]) < 0 && (H[:, j] .*= -one(T))
    end
    functions = _BasisEigenfunctions(b, H[:, 1:chosen], points, weights,
        X * H[:, 1:chosen])
    projected = residuals * G * functions.coefs
    covariance = X * system.coefficient_covariance * transpose(X)
    fullfunctions = X * H
    fitted = (fullfunctions .* transpose(system.eigenvalues)) * transpose(fullfunctions)
    all(isfinite, projected) && all(isfinite, covariance) && all(isfinite, fitted) ||
        throw(ArgumentError("basis FPCA results must be finite; rescale data"))
    settings = (; lambda=lam, penalty=specification, penalty_matrix=P, center,
        lambda_selection=tuning, ncomponents=deepcopy(ncomponents), max_k=Int(cap), maxK=Int(cap),
        eigen_tol, sign_tol, covariance_denominator=n, normalization=:exact_L2,
        original_harmonic_norms=system.normalization, eigenvalue_scale=:penalized_metric,
        grid_source=grid === nothing ? :display_201 : :user_display,
        quadrature=:exact_basis_gram, full_projected_covariance=fitted,
        reference="fda 6.3.0 pca.fd / fdPar")
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry,
        tolerance=sign_tol, grid=:display)
    FPCA(:basis, sampling_design(fd), points, weights, BasisFunctionalData(reshape(mu, 1, :), b),
        covariance, nothing, system.eigenvalues[1:chosen], system.eigenvalues,
        system.fve, functions, selection, system.dropped, sign_rule, projected,
        nothing, deepcopy(fd.ids), nothing, settings,
        BasisFunctionalData(fd.coefs, deepcopy(b); ids=fd.ids, metadata=fd.metadata))
end

function predict(f::FPCA, fd::BasisFunctionalData)
    _deprecated_score_prediction()
    _predict_result(f, fd)
end

_predict_result(f::FPCA, fd::BasisFunctionalData) = _basis_predict(f, fd, f.mean)
function _basis_predict(f, fd, mu::BasisFunctionalData)
    fd.basis == mu.basis || throw(ArgumentError("prediction basis must equal the FPCA training basis"))
    projected = (fd.coefs .- mu.coefs) * gram(mu.basis) * f.basis.coefs
    all(isfinite, projected) || throw(ArgumentError("FPCA prediction scores must be finite"))
    projected
end
_basis_predict(f, fd, mu) = throw(ArgumentError("basis prediction requires a basis FPCA fit"))

function _fpca_derivative(f::FPCA{T,M}, order; kwargs...) where {T,M<:BasisFunctionalData}
    throw(ArgumentError("basis FPCA derivative fits are not supported; use DerivativeBasis(eigenfunctions(f), order) for exact harmonic derivatives"))
end

"""
    FunctionalCCA
    fit(FunctionalCCA, fd1::BasisFunctionalData, fd2=fd1;
        ncan=nothing, ncomponents=nothing, lambda=(1e-10, 1e-10), penalty=(2, 2), center=true)

Penalized functional canonical correlation analysis following fda 6.3.0
`cca.fd`/`geigen`. Defaults are R's two canonical pairs, second-derivative
penalties, lambda 1e-10 on both sides, and centering. `lambda` may be a common
nonnegative scalar, a pair of scalars, or `LambdaCV(lambdas)` choosing a common
lambda by held-out leading-variate squared correlation (Ramsay and Silverman,
2005, §11.3.3). Penalties may be a common derivative order or a pair of orders,
Fourier harmonic-acceleration symbols, or positive-semidefinite matrices.
`ncomponents` is a descriptive alias for `ncan`, the number of canonical
pairs, not a marginal FPCA truncation. Supply at most one; omitting both
selects two. The stored settings key remains `ncan`.

`weights1` and `weights2` are exact `BasisFunctionalData` with one canonical
weight per row. As in R, each coefficient vector has Euclidean norm one.
`variates1` and `variates2` have subjects in rows. `correlations` contains the
penalized singular values; `squared_correlations` contains their squares,
which generally differ from empirical squared correlations of the variates.
`predict(f, fd1, fd2)` returns the two variate matrices using training means.
Paired inputs must have the same subject count and matching IDs when either
has IDs; their bases and domains may differ. Singular metrics throw, with no
implicit ridge or truncation. Settings and input bases are copied and shown.
Paired signs make the largest absolute coefficient on the first side positive.
"""
struct FunctionalCCA{T<:AbstractFloat,B1,B2,O}
    weights1::B1
    weights2::B2
    mean1::B1
    mean2::B2
    variates1::Matrix{T}
    variates2::Matrix{T}
    correlations::Vector{T}
    squared_correlations::Vector{T}
    ids::Union{Nothing,Vector{Any}}
    settings::O
end

"""
    canonical_weights(f::FunctionalCCA)

Return `(left=weights1, right=weights2)`, the exact `BasisFunctionalData`
canonical weights, with one canonical pair per row. Both objects are owned by
the fit; use `deepcopy` for independent data, coefficient and basis storage.
"""
canonical_weights(f::FunctionalCCA) = (; left=f.weights1, right=f.weights2)

"""
    canonical_variates(f::FunctionalCCA)

Return `(left=variates1, right=variates2)`, with paired training subjects in
rows and canonical pairs in columns. Both matrices are owned by the fit;
use `map(copy, canonical_variates(f))` for independent array storage.
"""
canonical_variates(f::FunctionalCCA) = (; left=f.variates1, right=f.variates2)

"""
    canonical_correlations(f::FunctionalCCA)

Return the selected penalized canonical correlations (singular values of the
whitened cross-covariance). They generally differ from empirical correlations
of the canonical variates. The vector is owned by the fit; use `copy` for
independent storage.
"""
canonical_correlations(f::FunctionalCCA) = f.correlations

"""
    squared_canonical_correlations(f::FunctionalCCA)

Return the squares of [`canonical_correlations`](@ref), the penalized
criterion values, rather than empirical squared correlations of the variates.
The vector is owned by the fit; use `copy` for independent storage.
"""
squared_canonical_correlations(f::FunctionalCCA) = f.squared_correlations

function _cca_pair(value, label)
    value isa Tuple && length(value) == 2 && return value
    value isa Tuple && throw(ArgumentError("$label must be a scalar setting or a pair"))
    (value, value)
end
function _cca_subjects(fd1, fd2; minobs=2)
    nobs(fd1) == nobs(fd2) || throw(DimensionMismatch("functional CCA inputs must have the same number of curves"))
    fd1.ids == fd2.ids || throw(ArgumentError("functional CCA subject IDs must match in order on both sides"))
    nobs(fd1) >= minobs || throw(ArgumentError("functional CCA requires at least $minobs paired curves"))
end
function _cca_system(X::Matrix{T}, Y::Matrix{T}, P, Q, lambdas, ncan) where {T}
    n = size(X, 1)
    U = _basis_cholesky(transpose(X) * X / n + lambdas[1] * P,
        "CCA first penalized covariance").U
    V = _basis_cholesky(transpose(Y) * Y / n + lambdas[2] * Q,
        "CCA second penalized covariance").U
    cross = transpose(X) * Y / n
    whitened = (transpose(U) \ cross) / V
    all(isfinite, whitened) || throw(ArgumentError("CCA whitened cross-covariance must be finite"))
    decomposition = svd(whitened)
    A = U \ decomposition.U[:, 1:ncan]
    B = V \ decomposition.V[:, 1:ncan]
    for j in 1:ncan
        A[:, j] ./= norm(A[:, j])
        B[:, j] ./= norm(B[:, j])
        if A[argmax(abs.(A[:, j])), j] < 0
            A[:, j] .*= -one(T)
            B[:, j] .*= -one(T)
        end
    end
    all(isfinite, A) && all(isfinite, B) && all(isfinite, decomposition.S) ||
        throw(ArgumentError("CCA results must be finite; rescale data"))
    (; weights1=A, weights2=B, correlations=decomposition.S[1:ncan])
end

function _squared_correlation(x, y)
    a, b = x .- sum(x)/length(x), y .- sum(y)/length(y)
    scale1, scale2 = norm(a), norm(b)
    scale1 > 0 && scale2 > 0 && isfinite(scale1) && isfinite(scale2) ||
        throw(ArgumentError("CCA CV requires finite nonzero held-out variate variance"))
    dot(a / scale1, b / scale2)^2
end
function _cca_lambda(lambda, C, D, G, H, P, Q, center)
    pair = _cca_pair(lambda, "lambda")
    (_basis_lambda(pair[1], eltype(C)), _basis_lambda(pair[2], eltype(C))), nothing
end
function _cca_lambda(cv::LambdaCV, C::Matrix{T}, D, G, H, P, Q, center) where {T}
    n = size(C, 1)
    n >= 3 || throw(ArgumentError("LambdaCV requires at least three paired curves"))
    candidates = [_basis_lambda(lam, T) for lam in cv.lambdas]
    all(isfinite, candidates) && length(unique(candidates)) == length(candidates) ||
        throw(ArgumentError("LambdaCV candidates must remain finite and distinct in the data element type"))
    criteria = zeros(T, length(candidates))
    for (c, lam) in enumerate(candidates)
        x, y = zeros(T, n), zeros(T, n)
        for i in 1:n
            try
                index = [j for j in 1:n if j != i]
                A, B = C[index, :], D[index, :]
                mu = center ? sum(A; dims=1)/(n-1) : zeros(T, 1, size(A, 2))
                nu = center ? sum(B; dims=1)/(n-1) : zeros(T, 1, size(B, 2))
                system = _cca_system((A .- mu)*G, (B .- nu)*H, P, Q, (lam, lam), 1)
                x[i] = only((C[i:i, :] .- mu) * G * system.weights1)
                y[i] = only((D[i:i, :] .- nu) * H * system.weights2)
            catch err
                _basis_cv_error(err, lam, i)
            end
        end
        criteria[c] = _squared_correlation(x, y)
    end
    chosen = argmax(criteria)
    (candidates[chosen], candidates[chosen]), (; method=:leave_one_curve_out,
        candidates, scores=criteria, chosen, criterion=:leading_variate_squared_correlation,
        direction=:maximize, centering=center ? :training_fold_mean : :none,
        sign_rule=:first_side_largest_absolute_coefficient, folds=n)
end

function fit(::Type{FunctionalCCA}, fd1::BasisFunctionalData, fd2::BasisFunctionalData=fd1;
        ncan=nothing, ncomponents=nothing, lambda=(1e-10, 1e-10), penalty=(2, 2), center=true)
    ncan === nothing || ncomponents === nothing || throw(ArgumentError(
        "supply only one of ncan and its alias ncomponents"))
    ncan = something(ncomponents, ncan, 2)
    _cca_subjects(fd1, fd2)
    center isa Bool || throw(ArgumentError("center must be Boolean"))
    _integer_in(ncan, 1, min(nbasis(fd1.basis), nbasis(fd2.basis))) ||
        throw(ArgumentError("ncan must be between one and the smaller basis dimension"))
    T = promote_type(eltype(fd1), eltype(fd2))
    b, d = deepcopy(fd1.basis), deepcopy(fd2.basis)
    C, D = Matrix{T}(fd1.coefs), Matrix{T}(fd2.coefs)
    G, H = Matrix{T}(gram(b)), Matrix{T}(gram(d))
    specifications = _cca_pair(penalty, "penalty")
    P, Q = _basis_penalty(b, specifications[1], T), _basis_penalty(d, specifications[2], T)
    lambdas, tuning = _cca_lambda(lambda, C, D, G, H, P, Q, center)
    mu = center ? sum(C; dims=1)/size(C, 1) : zeros(T, 1, size(C, 2))
    nu = center ? sum(D; dims=1)/size(D, 1) : zeros(T, 1, size(D, 2))
    X, Y = (C .- mu)*G, (D .- nu)*H
    system = _cca_system(X, Y, P, Q, lambdas, Int(ncan))
    settings = (; ncan=Int(ncan), lambda=lambdas, penalty=deepcopy(specifications),
        penalty_matrices=(P, Q), center, lambda_selection=tuning,
        covariance_denominator=size(C, 1), normalization=:unit_coefficient_norm,
        sign_rule=:first_side_largest_absolute_coefficient, singular_metrics=:error,
        reference="fda 6.3.0 cca.fd / geigen")
    FunctionalCCA(BasisFunctionalData(transpose(system.weights1), b),
        BasisFunctionalData(transpose(system.weights2), d), BasisFunctionalData(mu, b),
        BasisFunctionalData(nu, d), X*system.weights1, Y*system.weights2,
        system.correlations, system.correlations.^2, deepcopy(fd1.ids), settings)
end

function predict(f::FunctionalCCA, fd1::BasisFunctionalData, fd2::BasisFunctionalData)
    _cca_subjects(fd1, fd2; minobs=1)
    fd1.basis == f.weights1.basis && fd2.basis == f.weights2.basis ||
        throw(ArgumentError("CCA prediction bases must match their training bases"))
    X = (fd1.coefs .- f.mean1.coefs) * gram(f.weights1.basis) * transpose(f.weights1.coefs)
    Y = (fd2.coefs .- f.mean2.coefs) * gram(f.weights2.basis) * transpose(f.weights2.coefs)
    all(isfinite, X) && all(isfinite, Y) || throw(ArgumentError("CCA prediction variates must be finite"))
    (; variates1=X, variates2=Y)
end
"""
    nobs(f::FunctionalCCA)

Return the number of paired training subjects, using `StatsAPI.nobs`.
"""
nobs(f::FunctionalCCA) = size(f.variates1, 1)

"""
    ncomponents(f::FunctionalCCA)

Return the number of selected canonical pairs.
"""
ncomponents(f::FunctionalCCA) = length(f.correlations)
ids(f::FunctionalCCA) = f.ids === nothing ? nothing : deepcopy(f.ids)
function Base.show(io::IO, f::FunctionalCCA)
    print(io, "FunctionalCCA(n=", nobs(f), ", squared_correlations=", f.squared_correlations,
        ", settings=", _show_summary(f.settings), ")")
end
Base.show(io::IO, ::MIME"text/plain", f::FunctionalCCA) = show(io, f)
