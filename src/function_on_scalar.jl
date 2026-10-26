"""
    FunctionOnScalarRegression
    fit(FunctionOnScalarRegression, z, y::BasisFunctionalData; intercept=true,
        lambda=0, penalty_order=nothing, residual_covariance=:residual)

Function-on-scalar regression `Y_i(t) = sum_k x_ik beta_k(t) + e_i(t)` with a fixed
roughness penalty, following fda 6.3.0 `fRegress` (its `fRegress.fd` method)
with scalar covariates and fixed `fdPar` lambda. `z` is a finite vector or
matrix of scalar covariates (`nothing` with an intercept fits the mean only);
`x_i = [1, z_i]` when `intercept=true`, the constant term `fRegress.formula`
adds. `y` must be `BasisFunctionalData`: like `fRegress`, the response is a
basis object, and raw curves must be converted explicitly (for example with
`to_basis` or `smooth_curves`); no smoothing is chosen here.

Each `beta_k` is expanded in the response basis, `fRegress.formula`'s default
for scalar covariates. The coefficients `b = vec([b_1 ... b_q])` solve
`(X'X ⊗ G + blockdiag(lambda_k R_k)) b = vec(G Y' X)`, where `G` is the exact
Gram matrix of the basis, `Y` the response coefficients and `R_k` the penalty
matrix of the linear differential operator. `lambda` is one nonnegative number
or one per coefficient (default 0, fixed; automatic lambda is not offered).
`penalty_order=nothing` follows `fdPar`'s default operator: derivative order
`order-2` for B-splines, harmonic acceleration for Fourier bases and order 0 for
monomials; an integer (or `:harmonic_acceleration` for Fourier) or one entry
per coefficient overrides it. The system is symmetric positive definite; a
singular system throws.

Standard errors follow `fRegress.stderr`:
`Var(b) = C^(-1) (X'X ⊗ G Σ_c G) C^(-1)`, with `Σ_c` the covariance of one
response's basis coefficients (`y2cMap*SigmaE*y2cMap'` in fda).
`fRegress.stderr` requires the user to supply it. `residual_covariance=:residual`
(default) estimates `Σ_c = E'E/(n-q)` from the coefficient residuals `E`, with
the unpenalized residual degrees of freedom `n-q`; a symmetric positive
semidefinite `nbasis × nbasis` matrix supplies it instead. Intervals then use
Student t with `n-q` degrees of freedom for the estimate and the normal
distribution for a supplied matrix. `fRegress.stderr` integrates with a
Riemann sum on 501 points that gives full weight to both endpoints; the package
uses exact inner products.

Fitted values are `X*B'` in the response basis (exact; `fRegress` projects them
on 501 points). All choices are stored in `settings` and shown.
"""
struct FunctionOnScalarRegression{T<:AbstractFloat,B<:AbstractBasis,S}
    basis::B
    coefficients::Vector{T}
    covariance::Matrix{T}
    design::Matrix{T}
    response::Matrix{T}
    fitted_values::Matrix{T}
    residual_covariance::Matrix{T}
    system::Matrix{T}
    lambda::Vector{T}
    penalty_order::Vector{Any}
    intercept::Bool
    settings::S
end

_fos_default_order(b::BSplineBasis) = max(0, b.order - 2)
_fos_default_order(::FourierBasis) = :harmonic_acceleration
_fos_default_order(::MonomialBasis) = 0
_fos_default_order(b) = throw(ArgumentError(
    "function-on-scalar regression needs an explicit penalty_order for $(typeof(b))"))

function _fos_each(value, q, name)
    value isa Union{Tuple,AbstractVector} || return fill(value, q)
    length(value) == q || throw(DimensionMismatch("$name must be one value or one per coefficient ($q)"))
    collect(Any, value)
end

function _fos_penalty(basis, order, ::Type{T}) where {T}
    order === :harmonic_acceleration && !(basis isa FourierBasis) && throw(ArgumentError(
        "penalty_order=:harmonic_acceleration requires a Fourier basis"))
    order === :harmonic_acceleration || _integer_in(order, 0) || throw(ArgumentError(
        "penalty_order must be a nonnegative integer or :harmonic_acceleration"))
    Matrix{T}(penalty(basis, order))
end

Base.@constprop :aggressive function fit(::Type{FunctionOnScalarRegression}, z, y::BasisFunctionalData;
        intercept=true, lambda=0, penalty_order=nothing, residual_covariance=:residual)
    intercept isa Bool || throw(ArgumentError("intercept must be a Bool"))
    n = nobs(y)
    z === nothing && !intercept && throw(ArgumentError(
        "a model without scalar covariates needs intercept=true"))
    z === nothing || z isa _RegressionScalars ||
        throw(ArgumentError("scalar covariates must be a finite real vector or matrix"))
    scalar_type = z === nothing ? eltype(y) : Base.nonmissingtype(eltype(z))
    T = _floattype(promote_type(eltype(y), scalar_type))
    scalars = _regression_scalars(z, n, T)
    X = intercept ? hcat(ones(T, n), scalars) : scalars
    q, basis = size(X, 2), y.basis
    k = nbasis(basis)
    n > q || throw(ArgumentError("function-on-scalar regression requires more curves than coefficients"))
    lambdas = _fos_each(lambda, q, "lambda")
    all(l -> l isa Real && !(l isa Bool) && isfinite(l) && l >= 0, lambdas) ||
        throw(ArgumentError("lambda must be finite and nonnegative"))
    orders = penalty_order === nothing ? fill(_fos_default_order(basis), q) :
        _fos_each(penalty_order, q, "penalty_order")
    Y = Matrix{T}(y.coefs)
    G = Matrix{T}(gram(basis))
    G = (G + transpose(G))/2
    system = kron(transpose(X)*X, G)
    for j in 1:q
        index = (j-1)*k+1:j*k
        R = _fos_penalty(basis, orders[j], T)
        system[index, index] .+= T(lambdas[j]) .* (R + transpose(R))/2
    end
    system = (system + transpose(system))/2
    all(isfinite, system) || throw(ArgumentError("function-on-scalar system must be finite; rescale the inputs"))
    factor = cholesky(Symmetric(system); check=false)
    issuccess(factor) && minimum(diag(factor.U))^2 > length(system)*eps(T)*maximum(abs, system) ||
        throw(ArgumentError("function-on-scalar system is singular; remove redundant covariates or add a penalty"))
    rhs = vec(G*transpose(Y)*X)
    b = factor \ rhs
    B = reshape(b, k, q)
    fitted_values = X*transpose(B)
    residual = Y - fitted_values
    covariance_source = residual_covariance === :residual ? :residual_coefficients : :supplied
    sigma = if residual_covariance === :residual
        c = transpose(residual)*residual/(n - q)
        (c + transpose(c))/2
    else
        residual_covariance isa AbstractMatrix{<:Real} || throw(ArgumentError(
            "residual_covariance must be :residual or a real matrix"))
        size(residual_covariance) == (k, k) || throw(DimensionMismatch(
            "residual_covariance must be nbasis × nbasis"))
        supplied = Matrix{T}(residual_covariance)
        all(isfinite, supplied) && issymmetric(supplied) || throw(ArgumentError(
            "residual_covariance must be finite and symmetric"))
        minimum(eigvals(Symmetric(supplied))) >= -sqrt(eps(T))*max(one(T), maximum(abs, supplied)) ||
            throw(ArgumentError("residual_covariance must be positive semidefinite"))
        supplied
    end
    inverse = factor \ Matrix{T}(I, q*k, q*k)
    covariance = inverse*kron(transpose(X)*X, G*sigma*G)*inverse
    covariance = (covariance + transpose(covariance))/2
    all(isfinite, b) && all(isfinite, covariance) || throw(ArgumentError(
        "function-on-scalar coefficients and covariance must be finite; rescale the inputs"))
    settings = (; reference="fda 6.3.0 fRegress (fRegress.fd) and fRegress.stderr",
        coefficient_basis=:response_basis, intercept, lambda_selection=:fixed,
        penalty_source=penalty_order === nothing ? :fdPar_default : :supplied,
        inner_products=:exact_gram, residual_covariance=covariance_source,
        residual_dof=n - q, interval_distribution=covariance_source === :supplied ? :normal : :student_t,
        fitted=:exact_in_response_basis, observation_weights=:unit)
    FunctionOnScalarRegression(basis, b, covariance, X, Y, fitted_values, sigma, system,
        T.(lambdas), Vector{Any}(orders), intercept, settings)
end
fit(::Type{FunctionOnScalarRegression}, z, y; kwargs...) = throw(ArgumentError(
    "function-on-scalar regression requires a BasisFunctionalData response; convert raw curves explicitly with to_basis or smooth_curves"))

_fos_q(f) = size(f.design, 2)

"""Return the stacked basis coefficients `vec([b_1 ... b_q])`, intercept first."""
coef(f::FunctionOnScalarRegression) = copy(f.coefficients)
"""Return the `fRegress.stderr` covariance of `coef(f)` with exact inner products."""
vcov(f::FunctionOnScalarRegression) = copy(f.covariance)
"""Return standard errors of `coef(f)`."""
stderror(f::FunctionOnScalarRegression) = sqrt.(max.(diag(f.covariance), zero(eltype(f.covariance))))
"""Return the number of response curves."""
nobs(f::FunctionOnScalarRegression) = size(f.response, 1)
"""Return the number of basis coefficients estimated."""
dof(f::FunctionOnScalarRegression) = length(f.coefficients)
"""Return `nobs(f) - q`, the denominator of the estimated residual coefficient covariance."""
dof_residual(f::FunctionOnScalarRegression) = nobs(f) - _fos_q(f)
"""Return fitted response curves as `BasisFunctionalData` in the response basis."""
fitted(f::FunctionOnScalarRegression) = BasisFunctionalData(copy(f.fitted_values), f.basis)
"""Return response-minus-fitted curves as `BasisFunctionalData` in the response basis."""
residuals(f::FunctionOnScalarRegression) = BasisFunctionalData(f.response - f.fitted_values, f.basis)

"""
    coefficient_functions(f::FunctionOnScalarRegression)

Return a tuple of one-curve `BasisFunctionalData`, one per coefficient
(intercept first when fitted), in the response basis.
"""
function coefficient_functions(f::FunctionOnScalarRegression)
    k = nbasis(f.basis)
    Tuple(BasisFunctionalData(reshape(f.coefficients[(j-1)*k+1:j*k], 1, :), f.basis) for j in 1:_fos_q(f))
end

function _fos_grid(f, grid)
    a, b = domain(f.basis)
    g = grid === nothing ? collect(range(a, b; length=max(51, 10*nbasis(f.basis) + 1))) : grid
    g isa AbstractVector{<:Real} || throw(ArgumentError("grid must be a real vector"))
    T = eltype(f.coefficients)
    out = Vector{T}(g)
    _validate_strict_grid(out)
    all(t -> a <= t <= b, out) || throw(ArgumentError("grid must lie inside the basis domain"))
    out
end

function _fos_critical(f, alpha)
    a = _band_alpha(alpha)
    critical = f.settings.interval_distribution === :normal ? sqrt(_chisq_upper_quantile(1, a)) :
        _regression_tcritical(dof_residual(f), a)
    a, critical
end

function _fos_interval(f, estimate, variance, alpha, target)
    a, critical = _fos_critical(f, alpha)
    se = sqrt.(max.(variance, zero(eltype(variance))))
    lower, upper = estimate - critical .* se, estimate + critical .* se
    all(isfinite, lower) && all(isfinite, upper) || throw(ArgumentError("interval endpoints must be finite"))
    (; estimate=copy(estimate), se, lower, upper, alpha=a, critical_value=critical,
        settings=(; target, distribution=f.settings.interval_distribution,
            dof=f.settings.interval_distribution === :normal ? nothing : dof_residual(f),
            covariance=:fRegress_stderr_exact, residual_covariance=f.settings.residual_covariance,
            pointwise=true))
end

"""
    coefficient_bands(f::FunctionOnScalarRegression; alpha=0.05, grid=nothing)

Pointwise intervals `beta_k(t) ± critical*se(t)` for every coefficient (intercept
first), with `se(t)^2 = phi(t)' Var(b_k) phi(t)`. `grid=nothing` uses
`fRegress.stderr`'s plotting grid of `max(51, 10*nbasis+1)` equally spaced points.
"""
function coefficient_bands(f::FunctionOnScalarRegression; alpha=0.05, grid=nothing)
    g = _fos_grid(f, grid)
    Phi = Matrix{eltype(g)}(basis_matrix(f.basis, g))
    k = nbasis(f.basis)
    Tuple(map(1:_fos_q(f)) do j
        index = (j-1)*k+1:j*k
        estimate = Phi*f.coefficients[index]
        variance = vec(sum((Phi*f.covariance[index, index]).*Phi; dims=2))
        merge((; grid=copy(g)), _fos_interval(f, estimate, variance, alpha, :coefficient))
    end)
end

"""
    predict(f::FunctionOnScalarRegression, z; interval=:none, alpha=0.05, grid=nothing)

Predict response curves for new scalar covariates (the intercept is added as
in the fit). `interval=:none` returns `BasisFunctionalData`. `:confidence` gives
`fRegress.stderr`'s fitted-curve standard errors on a grid; `:prediction` adds
`phi(t)' Σ_c phi(t)` for a new response curve. Both return `grid`,
`estimate`, `se`, `lower`, `upper`, `alpha`, `critical_value` and `settings`
with subject × grid arrays.
"""
function predict(f::FunctionOnScalarRegression, z; interval=:none, alpha=0.05, grid=nothing)
    interval in (:none, :confidence, :prediction) || throw(ArgumentError(
        "interval must be :none, :confidence, or :prediction"))
    T = eltype(f.coefficients)
    z isa _RegressionScalars || throw(ArgumentError(
        "scalar covariates must be a real vector or matrix; use an m × 0 matrix for an intercept-only model"))
    scalars = _regression_scalars(z, size(z, 1), T)
    X = f.intercept ? hcat(ones(T, size(scalars, 1)), scalars) : scalars
    size(X, 2) == _fos_q(f) || throw(DimensionMismatch(
        "prediction covariate columns must match the fitted model"))
    k = nbasis(f.basis)
    coefficients = X*transpose(reshape(f.coefficients, k, :))
    interval === :none && grid !== nothing && throw(ArgumentError("grid is only used with intervals"))
    interval === :none && return BasisFunctionalData(coefficients, f.basis)
    g = _fos_grid(f, grid)
    Phi = Matrix{T}(basis_matrix(f.basis, g))
    estimate = coefficients*transpose(Phi)
    variance = similar(estimate)
    for i in axes(X, 1)
        A = kron(transpose(X[i, :]), Phi)
        variance[i, :] = vec(sum((A*f.covariance).*A; dims=2))
    end
    interval === :prediction && (variance .+= transpose(vec(sum((Phi*f.residual_covariance).*Phi; dims=2))))
    merge((; grid=g), _fos_interval(f, estimate, variance, alpha, interval))
end

function Base.show(io::IO, f::FunctionOnScalarRegression)
    print(io, "FunctionOnScalarRegression(n=", nobs(f), ", coefficients=", _fos_q(f),
        ", basis=", f.basis, ", lambda=", f.lambda, ", penalty_order=", f.penalty_order,
        ", settings=", f.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::FunctionOnScalarRegression) = show(io, f)
