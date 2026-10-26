# Penalized scalar-on-function regression after refund 0.1-40 `pfr` with
# `lf()` terms (Goldsmith, Bobb, Crainiceanu, Caffo and Reich 2011, JCGS
# 20:830-851), fitted by the package's REML/GCV solver (`PenalizedLeastSquares`).

"""
    PenalizedFunctionalRegression
    fit(PenalizedFunctionalRegression, x, y; scalar_covariates=nothing,
        k=nothing, basis=nothing, penalty_order=nothing, integration=:quadrature,
        presmooth=nothing, presmooth_options=(;), grid=nothing,
        family=:gaussian, link=nothing, criterion=:REML, lambda=nothing)

Penalized scalar-on-function regression

    g(E yᵢ) = ηᵢ = α + zᵢᵀγ + Σⱼ ∫ Xᵢⱼ(t) βⱼ(t) dt,

the model of refund 0.1-40 `pfr(y ~ z + lf(X1) + lf(X2), method="REML")`
(Gaussian responses with the identity link by default) and of
`pfr(..., family=binomial)` and similar for exponential-family responses.
`family` (`:gaussian`, `:binomial` with 0/1 responses, `:poisson`) and `link`
(`nothing` for the canonical link) are those of
[`PenalizedLeastSquares`](@ref), which fits the model by penalized IRLS with
Laplace-approximate REML, as mgcv 1.9-3 `gam` does for refund.
Each `βⱼ` is expanded in a basis `Bⱼ`, the integral is a weighted sum over the
covariate grid, `Σₜ Lⱼₜ Xᵢⱼ(t) Bⱼ(t)`, and `βⱼ` has its own smoothing parameter
`λⱼ` on a roughness penalty. Scalar covariates (a finite vector or matrix,
without an intercept column) and the intercept are unpenalized. `x` is one
univariate functional dataset, a tuple/vector of them, or
`MultivariateFunctionalData`; `y` is a finite response vector.

Per-covariate options may be given once or as a tuple/vector with one entry
per functional covariate. Defaults and sources:

- `basis=nothing`, `k=nothing`: a [`ThinPlateRegressionBasis`](@ref) of
  dimension `k=10` on the covariate grid with penalty `∫β''²`, refund's `lf()`
  default (`s(bs="tp")`, mgcv 1.9-3 `k=10`). Any other `AbstractBasis`
  covering the grid can be supplied (then `k` must be `nothing`) with
  `penalty(basis, penalty_order)` as its penalty (`penalty_order=2`).
- Penalties are rescaled as mgcv `smoothCon(scale.penalty=TRUE)` does:
  `S/s` with `s = ‖S‖₁/‖B‖∞²` over the grid; `lambda` refers to the rescaled
  penalties, like mgcv's `sp`, and `s` is stored as `penalty_scale`.
- `integration=:quadrature` (the grid's stored trapezoidal weights, equal to
  refund's `integration="trapezoidal"`). This departs from refund's default
  `"simpson"`, whose weights `(t_J-t_1)/J/3 * [1,4,2,4,…,2,1]` divide by the
  number of points `J` instead of the `J-1` intervals and are not Simpson's
  rule for an even number of intervals, so `β` is inflated by about
  `J/(J-1)`. `:refund_simpson` and `:riemann` reproduce refund's formulas
  exactly; a vector supplies weights.
- `presmooth=nothing` (refund `presmooth=NULL`): complete grid data are used
  as observed and basis data are evaluated on `grid`. Refund's options
  `"bspline"`, `"interpolate"` and its FPCA reconstructions are `:bspline`
  (cubic B-spline least squares per curve, `presmooth_options=(; nbasis=10)`,
  knots at quantiles of the grid as R's `splines::bs`), `:interpolate` (linear
  interpolation of observed values, constant beyond the first and last
  observation, as R's `approx(rule=2)`) and `:fpca`, which fits the package's
  `fit(FPCA, x; presmooth_options...)` and uses its reconstruction
  `mean + scores*eigenfunctions` instead of refund's `fpca.sc`/`fpca.face`.
  Sparse (irregular) or incomplete curves need a presmoother. Only the
  presmoothed curves enter the regression: presmoothing uncertainty, as in
  refund, is not propagated to `vcov` or the bands.
- `grid=nothing`: the curves' grid for grid data, the FPCA work grid for
  `:fpca` on irregular data; required for basis data and for `:bspline` or
  `:interpolate` on irregular data.
- `criterion=:REML` (refund `method="REML"`), or mgcv's `"GCV.Cp"`: `:GCV`
  for the Gaussian family (unknown scale) and `:UBRE` for the binomial and
  Poisson families (known scale); fixed `lambda` (one finite nonnegative value
  per functional covariate) skips selection.

A functional term whose integrals `Σₜ LₜXᵢ(t)` are equal for every subject is
confounded with the intercept and throws (mgcv would add a centering
constraint). Designs that do not identify the penalty null spaces throw.

`coef` orders coefficients as intercept, scalar covariates, then each
functional term's basis coefficients. `vcov(f; type)` is the Bayesian
posterior covariance `type=:bayesian` (mgcv `Vp`, the default),
`:frequentist` (mgcv `Ve`) or `:corrected`, mgcv's `Vc`, which adds the
first- and second-order corrections for smoothing parameter uncertainty
(Wood, Pya and Säfken 2016) and is available after REML selection.
`settings.corrected_covariance` records `:mgcv_Vc` when `Vc` was computed,
`:unavailable` without REML selection, `:unavailable_infinite_lambda` when a
smoothing parameter is effectively infinite (then `XᵀX + S_λ` is numerically
singular and the λ-derivatives are meaningless) and
`:unavailable_not_positive_definite` when its Cholesky factorization fails;
without `Vc`, bands and intervals default to `Vp`.
For exponential-family fits `Vp`, `Ve` and `Vc` use the final Fisher
weights and `Vc` the smoothing parameter Hessian of the Laplace REML (over
`log λ`, and `log φ` for the Gaussian family), as mgcv's `gam.fit3.post.proc`.
`dof` is the total effective degrees of freedom (intercept included) and
`dof_residual` is `nobs - dof`; `sigma2` is mgcv's scale estimate (`RSS/dof_residual`
for the identity link, one for binomial and Poisson, Pearson over
`dof_residual` otherwise). `r2` is mgcv's deviance explained `1 - D/D₀` (`D₀`
the deviance of the intercept-only model), which is the ordinary
`1 - RSS/TSS` for the Gaussian identity link. `fitted` returns the means and
`residuals` the response-scale residuals `y - μ`; `pls.linear_predictors`
holds `η`.

Fields include `pls` (the [`PenalizedLeastSquares`](@ref) fit), `lambda`,
`edf_terms`, `penalty_scale`, `sigma2`, `terms` (grids, integration weights,
bases, penalties, presmoothers and the processed training curves) and
`settings`. See the Regression documentation for the refund audit.
"""
struct PenalizedFunctionalRegression{T<:AbstractFloat,P<:PenalizedLeastSquares,TM<:Tuple,B<:Tuple,O<:NamedTuple}
    pls::P
    terms::TM
    beta::B
    design::Matrix{T}
    response::Vector{T}
    nscalar::Int
    lambda::Vector{T}
    penalty_scale::Vector{T}
    edf_terms::Vector{T}
    sigma2::T
    r_squared::T
    corrected_covariance::Union{Nothing,Matrix{T}}
    settings::O
end

const _PFR_PRESMOOTH = (:bspline, :interpolate, :fpca)
const _PFR_INTEGRATION = (:quadrature, :refund_simpson, :riemann)

# R's quantile(type = 7).
function _pfr_quantile(x::Vector{T}, p) where {T}
    h = (length(x) - 1) * p + 1
    lo = floor(Int, h)
    lo >= length(x) && return x[end]
    x[lo] + (h - lo) * (x[lo+1] - x[lo])
end

# R's splines::bs(grid, degree=3, df=nbasis, intercept=TRUE) as a basis.
function _pfr_bspline_basis(grid::Vector{T}, nbasis) where {T}
    _integer_in(nbasis, 4) || throw(ArgumentError("presmooth_options.nbasis must be an integer of at least 4"))
    interior = Int(nbasis) - 4
    probs = range(0, 1; length=interior + 2)[2:end-1]
    sorted = sort(grid)
    breaks = vcat(first(sorted), T[_pfr_quantile(sorted, p) for p in probs], last(sorted))
    all(i -> breaks[i+1] > breaks[i], 1:length(breaks)-1) || throw(ArgumentError(
        "B-spline presmoothing knots at grid quantiles must be distinct; lower presmooth_options.nbasis"))
    BSplineBasis(4, breaks)
end

function _pfr_presmooth_options(presmooth, options)
    options isa NamedTuple || throw(ArgumentError("presmooth_options entries must be named tuples"))
    if presmooth === :bspline
        all(in((:nbasis,)), keys(options)) || throw(ArgumentError(
            "presmooth=:bspline accepts only presmooth_options.nbasis"))
    elseif presmooth === :interpolate || presmooth === nothing
        isempty(options) || throw(ArgumentError("presmooth_options apply only to presmooth=:bspline or :fpca"))
    end
    options
end

function _pfr_target_grid(x, presmooth, grid, state, ::Type{T}) where {T}
    if grid === nothing
        x isa GridFunctionalData && return Vector{T}(x.grid)
        presmooth === :fpca && x isa IrregularFunctionalData && return Vector{T}(state.grid)
        throw(ArgumentError(x isa BasisFunctionalData ?
            "basis functional covariates need an explicit grid" :
            "irregular functional covariates need an explicit grid for presmooth=:$presmooth"))
    end
    grid isa AbstractVector{<:Real} || throw(ArgumentError("grid must be a real vector"))
    _validate_strict_grid(grid)
    x isa GridFunctionalData && Vector{T}(grid) != x.grid && throw(ArgumentError(
        "grid functional covariates use their own grid; omit the grid option"))
    Vector{T}(grid)
end

function _pfr_curve_check(t, name, i)
    isempty(t) && throw(ArgumentError("curve $i has no observed values for presmooth=:$name"))
end

# Per-curve presmoothers do not depend on the training sample.
function _pfr_interpolate(x, grid::Vector{T}) where {T}
    out = Matrix{T}(undef, nobs(x), length(grid))
    for i in 1:nobs(x)
        t, y = x[i]
        _pfr_curve_check(t, :interpolate, i)
        points = clamp.(grid, first(t), last(t))
        _evaluate_curve!(view(out, i, :), Vector{T}(t), Vector{T}(y), points, PiecewiseLinear())
    end
    out
end

function _pfr_bspline(x, grid::Vector{T}, basis) where {T}
    out = Matrix{T}(undef, nobs(x), length(grid))
    G = basis_matrix(basis, grid)
    a, z = domain(basis)
    for i in 1:nobs(x)
        t, y = x[i]
        _pfr_curve_check(t, :bspline, i)
        all(s -> a <= s <= z, t) || throw(ArgumentError(
            "curve $i has observations outside the presmoothing grid range [$a, $z]"))
        B = basis_matrix(basis, t)
        factor = qr(B, ColumnNorm())
        R = factor.R
        tolerance = max(size(B)...) * eps(T) * maximum(abs, diag(R); init=zero(T))
        length(t) >= nbasis(basis) && all(d -> abs(d) > tolerance, diag(R)) || throw(ArgumentError(
            "curve $i does not identify the $(nbasis(basis)) B-spline presmoothing coefficients"))
        out[i, :] .= G * (factor \ Vector{T}(y))
    end
    out
end

function _pfr_presmooth(x, presmooth, options, grid, ::Type{T}) where {T}
    if presmooth === nothing
        x isa IrregularFunctionalData && throw(ArgumentError(
            "irregular functional covariates need presmooth=:fpca, :bspline or :interpolate"))
        target = _pfr_target_grid(x, presmooth, grid, nothing, T)
        if x isa GridFunctionalData
            x.observed === nothing || all(x.observed) || throw(ArgumentError(
                "functional covariates with missing values need presmooth=:fpca, :bspline or :interpolate"))
            return Matrix{T}(x.values), target, nothing
        end
        return Matrix{T}(evaluate(x, target)), target, nothing
    end
    presmooth in _PFR_PRESMOOTH || throw(ArgumentError("presmooth must be nothing, :bspline, :interpolate or :fpca"))
    if presmooth === :fpca
        model = fit(FPCA, x; options...)
        target = _pfr_target_grid(x, presmooth, grid, model, T)
        values = Matrix{T}(reconstruct(model; grid=target).values)
        return values, target, model
    end
    x isa BasisFunctionalData && throw(ArgumentError(
        "presmooth=:$presmooth applies to raw grid or irregular covariates"))
    target = _pfr_target_grid(x, presmooth, grid, nothing, T)
    if presmooth === :interpolate
        return _pfr_interpolate(x, target), target, nothing
    end
    basis = _pfr_bspline_basis(target, get(options, :nbasis, 10))
    _pfr_bspline(x, target, basis), target, basis
end

function _pfr_integration(integration, grid::Vector{T}) where {T}
    J = length(grid)
    if integration === :quadrature
        return quadrature_weights(grid)
    elseif integration === :refund_simpson
        J >= 3 || throw(ArgumentError("integration=:refund_simpson needs at least three grid points"))
        pattern = vcat(one(T), T[isodd(j) ? 4 : 2 for j in 1:(J-2)], one(T))
        return (grid[J] - grid[1]) / J / 3 .* pattern
    elseif integration === :riemann
        d = diff(grid)
        return vcat(sum(d) / length(d), d)
    elseif integration isa AbstractVector{<:Real}
        Base.require_one_based_indexing(integration)
        length(integration) == J || throw(DimensionMismatch("integration weights must have one entry per grid point ($J)"))
        all(isfinite, integration) || throw(ArgumentError("integration weights must be finite"))
        return Vector{T}(integration)
    end
    throw(ArgumentError("integration must be :quadrature, :refund_simpson, :riemann or a weight vector"))
end

function _pfr_basis(basis, k, penalty_order, grid::Vector{T}) where {T}
    if basis === nothing
        penalty_order === nothing || throw(ArgumentError(
            "the thin plate regression spline penalty order is fixed at 2; omit penalty_order"))
        dimension = k === nothing ? 10 : k
        (k === nothing || k isa Integer) && !(k isa Bool) ||
            throw(ArgumentError("k must be a positive integer"))
        b = ThinPlateRegressionBasis(grid, dimension)
        return b, Matrix{T}(b.penalty), Int(dimension), 2
    end
    basis isa AbstractBasis || throw(ArgumentError("basis must be nothing or an AbstractBasis"))
    k === nothing || throw(ArgumentError("k sets the default basis dimension; omit it when supplying a basis"))
    order = penalty_order === nothing ? 2 : penalty_order
    a, z = domain(basis)
    a <= first(grid) && last(grid) <= z || throw(ArgumentError("the supplied basis must cover the covariate grid"))
    S = Matrix{T}(penalty(basis, order))
    S = (S + transpose(S)) / 2
    maximum(abs, S) > 0 || throw(ArgumentError("the basis penalty of order $order is zero; lower penalty_order"))
    deepcopy(basis), S, nbasis(basis), Int(order)
end

function _pfr_term(x, k, basis, penalty_order, integration, presmooth, options, grid, ::Type{T}) where {T}
    options = _pfr_presmooth_options(presmooth, options)
    values, target, presmoother = _pfr_presmooth(x, presmooth, options, grid, T)
    all(isfinite, values) || throw(ArgumentError("functional covariate values must be finite"))
    L = _pfr_integration(integration, target)
    b, S, dimension, order = _pfr_basis(basis, k, penalty_order, target)
    B = Matrix{T}(basis_matrix(b, target))
    # mgcv smoothCon(scale.penalty=TRUE), before by-variable handling.
    size_x = maximum(sum(abs, B; dims=2))^2
    scale = opnorm(S, 1) / size_x
    isfinite(scale) && scale > 0 || throw(ArgumentError("functional term penalty scale must be finite and positive"))
    integrals = values * L
    centre = sum(integrals) / length(integrals)
    spread = sqrt(sum(abs2, integrals .- centre) / max(length(integrals) - 1, 1))
    spread > abs(centre) * eps(T) * 1000 || throw(ArgumentError(
        "the functional covariate's integrals are equal for every subject, so its coefficient is confounded with the intercept"))
    (; grid=target, weights=L, integration=integration isa Symbol ? integration : :supplied,
        basis=b, penalty=S, penalty_scale=scale, penalty_order=order, k=dimension,
        presmooth, presmooth_options=options, presmoother, training=deepcopy(x),
        values, sampling=sampling_design(x))
end

_pfr_block(term, values) = (values .* transpose(term.weights)) * basis_matrix(term.basis, term.grid)

function _pfr_new_values(term, x, ::Type{T}) where {T}
    _regression_same_data(x, term.training) && return term.values
    presmooth = term.presmooth
    if presmooth === nothing
        x isa IrregularFunctionalData && throw(ArgumentError(
            "irregular functional covariates need a presmoother fitted with the model"))
        if x isa GridFunctionalData
            x.grid == term.grid || throw(ArgumentError("prediction curves must use the fitted covariate grid"))
            x.observed === nothing || all(x.observed) || throw(ArgumentError(
                "prediction curves with missing values need a presmoother fitted with the model"))
            return Matrix{T}(x.values)
        end
        return Matrix{T}(evaluate(x, term.grid))
    elseif presmooth === :fpca
        projected = project_scores(term.presmoother, x)
        return Matrix{T}(reconstruct(term.presmoother, projected; grid=term.grid).values)
    end
    x isa BasisFunctionalData && throw(ArgumentError(
        "presmooth=:$presmooth applies to raw grid or irregular covariates"))
    presmooth === :interpolate ? _pfr_interpolate(x, term.grid) :
        _pfr_bspline(x, term.grid, term.presmoother)
end

# mgcv 1.9-3 dchol: derivative of the upper Cholesky factor R (R'R = A).
function _pfr_dchol(dA::Matrix{T}, R::Matrix{T}) where {T}
    p = size(R, 1)
    dR = zeros(T, p, p)
    for i in 1:p, j in i:p
        x = zero(T)
        for k in 1:(i-1)
            x += R[k, i] * dR[k, j] + R[k, j] * dR[k, i]
        end
        dR[i, j] = j > i ? (dA[i, j] - x - R[i, j] * dR[i, i]) / R[i, i] :
            (dA[i, i] - x) / 2 / R[i, i]
    end
    dR
end

# mgcv 1.9-3 gam.fit3.post.proc and Vb.corr for a Gaussian REML fit with
# estimated scale: Vc = Vp + J Vρ Jᵀ + second-order Cholesky correction. The
# REML Hessian is taken over (log λ, log φ) at φ = Dp/(n - Mp).
function _pfr_corrected_covariance(X::Matrix{T}, y::Vector{T}, penalties, columns, lambda,
        beta, sigma2, null_dimension) where {T}
    n, p = size(X)
    M = length(lambda)
    full = _pfr_full_penalties(penalties, columns, p, T)
    H = Matrix(Symmetric(transpose(X) * X + sum(lambda .* full)))
    factor = cholesky(Symmetric(H); check=false)
    issuccess(factor) || return nothing, nothing
    Hinv = inv(factor)
    Sbeta = [S * beta for S in full]
    db = reduce(hcat, [-lambda[j] .* (Hinv * Sbeta[j]) for j in 1:M])
    rss = sum(abs2, y - X * beta)
    g = T[lambda[j] * dot(beta, Sbeta[j]) for j in 1:M]
    Dp = rss + sum(g)
    phi = Dp / (n - null_dimension)
    HA = [Hinv * (lambda[j] .* full[j]) for j in 1:M]
    hess = zeros(T, M + 1, M + 1)
    for j in 1:M, k in 1:M
        hess_dp = (j == k ? g[j] : zero(T)) + 2 * lambda[j] * dot(Sbeta[j], db[:, k])
        hess_logdet = (j == k ? tr(HA[j]) : zero(T)) - _trace_product(HA[j], HA[k])
        hess[j, k] = hess_dp / (2 * phi) + hess_logdet / 2
    end
    for j in 1:M
        hess[j, M+1] = hess[M+1, j] = -g[j] / (2 * phi)
    end
    hess[M+1, M+1] = Dp / (2 * phi)
    _pfr_vc(factor, db, hess, lambda, full, sigma2)
end

function _pfr_full_penalties(penalties, columns, p, ::Type{T}) where {T}
    map(penalties, columns) do S, c
        out = zeros(T, p, p)
        out[c, c] .= S
        out
    end
end

# The same correction for an exponential-family REML fit (gam.fit3.post.proc):
# H = XᵀW_F X + S_λ with the final Fisher weights, dβ/dρ from the solver's
# implicit (Newton) derivatives, and the REML Hessian over log λ (known scale)
# or over (log λ, log φ) at the REML scale φ (Gaussian family).
function _pfr_glm_corrected_covariance(X::Matrix{T}, pls, final, penalties, columns) where {T}
    p = size(X, 2)
    lambda = pls.lambda
    M = length(lambda)
    full = _pfr_full_penalties(penalties, columns, p, T)
    wf = _pirls_values(final.state, Val(:wf))
    H = Matrix(Symmetric(transpose(X) * (wf .* X) + sum(lambda .* full)))
    factor = cholesky(Symmetric(H); check=false)
    issuccess(factor) || return nothing, nothing
    if pls.settings.scale === :known
        hess = copy(final.hessian)
    else
        phi = pls.reml_scale
        hess = zeros(T, M + 1, M + 1)
        hess[1:M, 1:M] .= final.hess_Dp ./ (2 * phi) .+ final.hess_logdet ./ 2 .- final.hessian_penalty ./ 2
        hess[1:M, M+1] .= hess[M+1, 1:M] .= -final.grad_Dp ./ (2 * phi)
        hess[M+1, M+1] = final.Dp / (2 * phi)
    end
    _pfr_vc(factor, final.db, hess, lambda, full, pls.sigma2)
end

# Vc = Vp + db Vρ dbᵀ + scale * Σⱼₖ Vr[j,k] dR⁻¹ⱼ dR⁻¹ₖᵀ (mgcv Vb.corr), with Vρ
# the pseudo-inverse of the smoothing parameter Hessian (scale row dropped)
# and Vr its inverse regularized by a weak prior (mgcv 1/10).
function _pfr_vc(factor, db::Matrix{T}, hess::Matrix{T}, lambda, full, sigma2) where {T}
    M = length(lambda)
    p = size(db, 1)
    Hinv = inv(factor)
    decomposition = eigen(Symmetric(hess))
    values, vectors = decomposition.values, decomposition.vectors
    inverse_root = T[v > 0 ? 1 / sqrt(v) : zero(T) for v in values]
    root = (inverse_root .* transpose(vectors))[:, 1:M]
    Vrho = transpose(root) * root
    first_order = db * Vrho * transpose(db)
    # A weak prior regularizes the smoothing parameter covariance (mgcv 1/10).
    regularized = T[1 / (max(v, zero(T)) + T(1) / 10) for v in values]
    Vr = (vectors * Diagonal(regularized) * transpose(vectors))[1:M, 1:M]
    R = Matrix(factor.U)
    Rinv = inv(UpperTriangular(R))
    dRinv = map(1:M) do j
        -Matrix(Rinv * _pfr_dchol(lambda[j] .* full[j], R) * Rinv)
    end
    second_order = zeros(T, p, p)
    for j in 1:M, k in 1:M
        second_order .+= Vr[j, k] .* (dRinv[j] * transpose(dRinv[k]))
    end
    Vp = Hinv .* sigma2
    Vc = Matrix(Symmetric(Vp + first_order + sigma2 .* second_order))
    all(isfinite, Vc) || return nothing, nothing
    Vc, (; hessian=hess, Vrho, Vr)
end

function _pfr_response(y)
    y isa AbstractVector || throw(ArgumentError("response must be a finite real vector"))
    Base.require_one_based_indexing(y)
    missing_index = findfirst(ismissing, y)
    missing_index === nothing || throw(ArgumentError("response contains missing at index $missing_index"))
    all(v -> v isa Real, y) || throw(ArgumentError("response must be a finite real vector"))
    all(isfinite, y) || throw(ArgumentError("response must be finite"))
    y
end

function fit(::Type{PenalizedFunctionalRegression}, x, y; scalar_covariates=nothing,
        k=nothing, basis=nothing, penalty_order=nothing, integration=:quadrature,
        presmooth=nothing, presmooth_options=(;), grid=nothing, family::Symbol=:gaussian,
        link=nothing, criterion::Symbol=:REML, lambda=nothing)
    curves = _regression_inputs(x)
    n, q = nobs(first(curves)), length(curves)
    y = _pfr_response(y)
    length(y) == n || throw(DimensionMismatch("response length must equal the functional covariate subject count"))
    fam = _glm_family(family, link)
    criterion = _pls_criterion(criterion, fam)
    ks = _regression_each(k, q, "k")
    bases = _regression_each(basis, q, "basis")
    orders = _regression_each(penalty_order, q, "penalty_order")
    rules = integration isa AbstractVector{<:Real} ? ntuple(_ -> integration, q) :
        _regression_each(integration, q, "integration")
    smoothers = _regression_each(presmooth, q, "presmooth")
    options = _regression_each(presmooth_options, q, "presmooth_options")
    grids = grid isa AbstractVector{<:Real} ? ntuple(_ -> grid, q) : _regression_each(grid, q, "grid")
    scalar_covariates === nothing || scalar_covariates isa _RegressionScalars ||
        throw(ArgumentError("scalar_covariates must be a finite real vector or matrix"))
    scalar_type = scalar_covariates === nothing ? Float64 : Base.nonmissingtype(eltype(scalar_covariates))
    T = _floattype(promote_type(Base.nonmissingtype(eltype(y)), scalar_type, map(eltype, curves)...))
    response = Vector{T}(y)
    all(isfinite, response) || throw(ArgumentError("response must remain finite after conversion"))
    total = sum(abs2, response .- sum(response) / n)
    isfinite(total) && total > 0 || throw(ArgumentError("response must have finite positive variation"))
    _glm_check_response(fam, response, ones(T, n))
    z = _regression_scalars(scalar_covariates, n, T)
    terms = ntuple(j -> _pfr_term(curves[j], ks[j], bases[j], orders[j], rules[j], smoothers[j],
        options[j], grids[j], T), Val(length(curves)))
    _pfr_fit(terms, response, z, total, criterion, lambda, fam)
end

# Bind the model types independently of the keyword-dependent term types.
function _pfr_fit(terms::Tuple, response::Vector{T}, z::Matrix{T}, total::T, criterion, lambda,
        fam::_GLMFamily) where {T}
    n, q = length(response), length(terms)
    blocks = map(term -> _pfr_block(term, term.values), terms)
    design = hcat(ones(T, n), z, blocks...)
    all(isfinite, design) || throw(ArgumentError("regression design must be finite; rescale the inputs"))
    offset = 1 + size(z, 2)
    starts = cumsum(vcat(offset, [term.k for term in terms]))
    columns = [collect((starts[j]+1):starts[j+1]) for j in 1:q]
    penalties = [term.penalty ./ term.penalty_scale for term in terms]
    if lambda !== nothing
        lambda isa AbstractVector{<:Real} && length(lambda) == q ||
            throw(DimensionMismatch("lambda must be a real vector with one value per functional covariate ($q)"))
    end
    pls, final = _pls_fit(design, response; penalties=[S => c for (S, c) in zip(penalties, columns)],
        weights=nothing, family=fam.family, link=fam.link, criterion, lambda, lower=nothing,
        upper=nothing, rank_tol=nothing, tol=1e-8, maxiter=200, irls_tol=1e-12, irls_maxiter=200)
    beta = map((term, j) -> BasisFunctionalData(reshape(pls.coef[columns[j]], 1, :), deepcopy(term.basis)),
        terms, ntuple(identity, Val(length(terms))))
    corrected = nothing
    hessian = zeros(T, 0, 0)
    corrected_status = :unavailable
    if criterion === :REML && lambda === nothing && isfinite(pls.sigma2)
        # At an effectively infinite λ, XᵀX + S_λ is numerically singular and
        # the λ-derivatives behind Vc are meaningless, so Vp is used instead.
        if any(==(:effectively_infinite), pls.lambda_status)
            corrected_status = :unavailable_infinite_lambda
        else
            corrected, correction = _glm_identity(fam) ?
                _pfr_corrected_covariance(design, response, penalties, columns,
                    pls.lambda, pls.coef, pls.sigma2, pls.null_space_dimension) :
                _pfr_glm_corrected_covariance(design, pls, final, penalties, columns)
            if corrected === nothing
                corrected_status = :unavailable_not_positive_definite
            else
                hessian = correction.hessian
                corrected_status = :mgcv_Vc
            end
        end
    end
    # Deviance explained (mgcv dev.expl); 1 - RSS/TSS for the Gaussian identity.
    explained = _glm_identity(fam) ? one(T) - sum(abs2, pls.residuals) / total :
        one(T) - pls.deviance / _glm_null_deviance(fam, response, ones(T, length(response)))
    settings = (; family=fam.family, link=fam.link, scale=pls.settings.scale, criterion,
        lambda_source=lambda === nothing ? :selected : :fixed,
        lambda_status=copy(pls.lambda_status), intercept=true,
        coefficient_order=:intercept_scalars_functional_basis,
        integration=map(t -> t.integration, terms),
        k=map(t -> t.k, terms), penalty_order=map(t -> t.penalty_order, terms),
        presmooth=map(t -> t.presmooth, terms), presmooth_options=map(t -> t.presmooth_options, terms),
        corrected_covariance=corrected_status,
        smoothing_parameter_hessian=hessian,
        reference="refund 0.1-40 pfr/lf; mgcv 1.9-3 gam(method=\"REML\")")
    PenalizedFunctionalRegression(pls, terms, beta, design, response, size(z, 2), copy(pls.lambda),
        T[t.penalty_scale for t in terms], copy(pls.edf_penalty), pls.sigma2, explained,
        corrected, settings)
end

"""Return copied coefficients ordered as intercept, scalars, then each term's basis coefficients."""
coef(f::PenalizedFunctionalRegression) = copy(f.pls.coef)
"""
    vcov(f::PenalizedFunctionalRegression; type=:bayesian)

Coefficient covariance in the order used by `coef`: `:bayesian` (mgcv `Vp`),
`:frequentist` (mgcv `Ve`) or `:corrected` (mgcv `Vc`, after REML selection;
throws when `settings.corrected_covariance` is not `:mgcv_Vc`).
"""
function vcov(f::PenalizedFunctionalRegression; type::Symbol=:bayesian)
    type === :corrected || return vcov(f.pls; type)
    if f.corrected_covariance === nothing
        status = f.settings.corrected_covariance
        throw(ArgumentError(status === :unavailable_infinite_lambda ?
            "the corrected covariance is unavailable because a smoothing parameter is effectively infinite" :
            status === :unavailable_not_positive_definite ?
            "the corrected covariance is unavailable because XᵀX + S_λ is not numerically positive definite" :
            "the corrected covariance needs REML-selected smoothing parameters"))
    end
    copy(f.corrected_covariance)
end
"""Return copied fitted means `h(η)` (the fitted responses for the identity link)."""
fitted(f::PenalizedFunctionalRegression) = fitted(f.pls)
"""Return copied response-scale residuals `y - μ` from a penalized functional regression."""
residuals(f::PenalizedFunctionalRegression) = residuals(f.pls)
"""Return the number of subjects in a penalized functional regression."""
nobs(f::PenalizedFunctionalRegression) = length(f.response)
"""Return the total effective degrees of freedom, including the intercept and scalar covariates."""
dof(f::PenalizedFunctionalRegression) = f.pls.edf_total
"""Return `nobs(f) - dof(f)`, the residual effective degrees of freedom."""
dof_residual(f::PenalizedFunctionalRegression) = nobs(f) - dof(f)
"""Return the deviance explained `1 - D/D₀` (mgcv `dev.expl`; `1 - RSS/TSS` for the Gaussian identity link)."""
r2(f::PenalizedFunctionalRegression) = f.r_squared
"""Return coefficient standard errors from `vcov(f; type)`."""
stderror(f::PenalizedFunctionalRegression; type::Symbol=:bayesian) =
    sqrt.(max.(diag(vcov(f; type)), zero(eltype(f.response))))
coefficient_functions(f::PenalizedFunctionalRegression) = deepcopy(f.beta)

function _pfr_covariance_choice(f, covariance)
    covariance === nothing && return f.corrected_covariance === nothing ? :bayesian : :corrected
    covariance in (:bayesian, :frequentist, :corrected) ||
        throw(ArgumentError("covariance must be nothing, :bayesian, :frequentist or :corrected"))
    covariance
end

function _pfr_intervals(estimate, variance, alpha, target, covariance, sigma2)
    a = _band_alpha(alpha)
    critical = sqrt(2) * SpecialFunctions.erfinv(1 - a)
    se = sqrt.(max.(variance, zero(eltype(variance))))
    lower, upper = estimate - critical .* se, estimate + critical .* se
    all(isfinite, lower) && all(isfinite, upper) || throw(ArgumentError("interval endpoints must be finite"))
    (; estimate=copy(estimate), se, lower, upper, alpha=a, critical_value=critical,
        settings=(; target, scale=:link, distribution=:normal, covariance, sigma2,
            presmoothing_uncertainty=false, simultaneous=false))
end

# refund 0.1-40 coef.pfr default coordinates (ndefault).
function _pfr_default_grid(grid::Vector{T}) where {T}
    x = sort(unique(grid))
    diffs = unique(round.(diff(x); digits=10))
    n = length(diffs) < 5 ? floor(Int, (x[end] - x[1]) / minimum(diffs) + 1e-10) + 1 : 101
    while n > 101
        n = round(Int, n / 2)
    end
    collect(range(x[1], x[end]; length=n))
end

"""
    coefficient_bands(f::PenalizedFunctionalRegression; alpha=0.05, grid=nothing,
        covariance=nothing)

Pointwise normal intervals `β̂ⱼ(t) ± z₁₋α/₂ se(t)` for each coefficient
function (on the link scale for exponential-family fits), from the posterior
covariance of its basis coefficients.
`covariance=nothing` follows refund 0.1-40 `coef.pfr(useVc=TRUE)`: mgcv's
smoothing-parameter-corrected `Vc` when available (REML selection), else
`Vp`; pass `:bayesian`, `:frequentist` or `:corrected` to choose. The choice is
recorded in `settings.covariance`. `grid=nothing` uses refund's default
coordinates (the covariate grid when it has fewer than five distinct
spacings and at most 101 points, otherwise 101 equally spaced points); one
grid, or a tuple of grids, may be supplied. Refund offers no simultaneous
bands for `pfr`, and none are returned. Presmoothing is treated as fixed.
"""
function coefficient_bands(f::PenalizedFunctionalRegression; alpha=0.05, grid=nothing, covariance=nothing)
    choice = _pfr_covariance_choice(f, covariance)
    V = vcov(f; type=choice)
    q = length(f.terms)
    grids = grid === nothing ? map(t -> _pfr_default_grid(t.grid), f.terms) :
        q == 1 && grid isa AbstractVector{<:Real} ? (grid,) : _regression_each(grid, q, "grid")
    offset = 1 + f.nscalar
    starts = cumsum(vcat(offset, [t.k for t in f.terms]))
    map(f.terms, ntuple(identity, q), grids) do term, j, g
        g isa AbstractVector{<:Real} || throw(ArgumentError("coefficient interval grids must be real vectors"))
        _validate_strict_grid(g)
        B = basis_matrix(term.basis, g)
        index = (starts[j]+1):starts[j+1]
        estimate = B * f.pls.coef[index]
        variance = vec(sum((B * V[index, index]) .* B; dims=2))
        merge((; grid=Vector{eltype(f.response)}(g)),
            _pfr_intervals(estimate, variance, alpha, :coefficient, choice, f.sigma2))
    end
end

function _pfr_newdesign(f::PenalizedFunctionalRegression{T}, x, z)::Matrix{T} where {T}
    curves = _regression_inputs(x)
    length(curves) == length(f.terms) || throw(DimensionMismatch(
        "prediction functional covariate count must match the regression fit"))
    n = nobs(first(curves))
    scalar = _regression_scalars(z, n, T)
    size(scalar, 2) == f.nscalar || throw(DimensionMismatch(
        "prediction scalar_covariates column count must match the regression fit"))
    blocks = map(f.terms, curves) do term, c
        values = _pfr_new_values(term, c, T)
        all(isfinite, values) || throw(ArgumentError("prediction covariate values must be finite"))
        _pfr_block(term, values)
    end
    hcat(ones(T, n), scalar, blocks...)
end

"""
    predict(f::PenalizedFunctionalRegression, x; scalar_covariates=nothing,
        type=:response, interval=:none, alpha=0.05, covariance=:bayesian)

Predict scalar responses for new subjects, as refund 0.1-40 `predict.pfr`:
new curves are presmoothed with the fitted presmoother (projected on the
stored FPCA for `:fpca`), weighted by the stored integration weights on the
fitted grid and multiplied by the fitted basis. Grid curves without a
presmoother must use the fitted grid; basis curves are evaluated on it.
`type=:response` (refund's default) returns means `μ = h(η)`, `type=:link`
the linear predictor `η`; they coincide for the Gaussian identity link.
`interval=:confidence` returns normal intervals for `η` from
`vcov(f; type=covariance)` (default `Vp`, as mgcv `predict.gam`); with
`type=:response` the estimate and both endpoints are mapped through the
inverse link (`se` stays on the link scale, `settings.scale` records the
scale). `interval=:prediction` adds `sigma2` for a new Gaussian response and
is available only for the Gaussian identity link. The fields are those of
[`coefficient_bands`](@ref) except `grid`.
"""
Base.@constprop :aggressive function predict(f::PenalizedFunctionalRegression, x; scalar_covariates=nothing,
        type::Symbol=:response, interval=:none, alpha=0.05, covariance::Symbol=:bayesian)
    interval in (:none, :confidence, :prediction) ||
        throw(ArgumentError("interval must be :none, :confidence, or :prediction"))
    type in (:link, :response) || throw(ArgumentError("type must be :link or :response"))
    design = _pfr_newdesign(f, x, scalar_covariates)
    eta = design * f.pls.coef
    all(isfinite, eta) || throw(ArgumentError("regression predictions must be finite"))
    fam = _GLMFamily(f.settings.family, f.settings.link)
    interval === :none && return type === :link ? eta : _glm_response_values(fam, eta)
    _pfr_check_interval(fam, interval)
    choice = _pfr_covariance_choice(f, covariance)
    variance = vec(sum((design * vcov(f; type=choice)) .* design; dims=2))
    interval === :prediction && (variance .+= f.sigma2)
    _glm_scale_intervals(fam, _pfr_intervals(eta, variance, alpha, interval, choice, f.sigma2), type)
end

function _pfr_check_interval(fam, interval)
    interval === :prediction && !_glm_identity(fam) && throw(ArgumentError(
        "prediction intervals need a Gaussian response with the identity link; use interval=:confidence"))
end

"""
    fitted_bands(f::PenalizedFunctionalRegression; alpha=0.05, interval=:confidence,
        covariance=:bayesian, type=:response)

Pointwise normal intervals for the training responses, as in `predict`.
"""
function fitted_bands(f::PenalizedFunctionalRegression; alpha=0.05, interval=:confidence,
        covariance::Symbol=:bayesian, type::Symbol=:response)
    interval in (:confidence, :prediction) || throw(ArgumentError("interval must be :confidence or :prediction"))
    type in (:link, :response) || throw(ArgumentError("type must be :link or :response"))
    fam = _GLMFamily(f.settings.family, f.settings.link)
    _pfr_check_interval(fam, interval)
    choice = _pfr_covariance_choice(f, covariance)
    variance = vec(sum((f.design * vcov(f; type=choice)) .* f.design; dims=2))
    interval === :prediction && (variance .+= f.sigma2)
    _glm_scale_intervals(fam, _pfr_intervals(f.pls.linear_predictors, variance, alpha, interval,
        choice, f.sigma2), type)
end

function Base.show(io::IO, f::PenalizedFunctionalRegression)
    print(io, "PenalizedFunctionalRegression(n=", nobs(f), ", terms=", length(f.terms),
        ", family=", f.settings.family, ", link=", f.settings.link,
        ", scalar_covariates=", f.nscalar, ", criterion=", f.settings.criterion,
        ", lambda=", f.lambda, ", status=", f.settings.lambda_status,
        ", penalty_scale=", f.penalty_scale, ", edf=", f.edf_terms, ", total edf=", dof(f),
        ", sigma2=", f.sigma2, ", R2=", f.r_squared, ", settings=", f.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::PenalizedFunctionalRegression) = show(io, f)
