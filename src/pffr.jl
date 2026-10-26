# Penalized function-on-function regression after refund 0.1.40 `pffr` with
# `ff()` terms (Ivanescu, Staicu, Scheipl and Greven 2015, Statistics and
# Computing 25:1197-1213; Scheipl, Staicu and Greven 2015, JCGS 24:477-501),
# fitted by the package's REML/GCV solver (`PenalizedLeastSquares`).

"""
    PenalizedFunctionOnFunctionRegression
    fit(PenalizedFunctionOnFunctionRegression, x, y; scalar_covariates=nothing,
        k=(5, 5), integration=:quadrature, presmooth=nothing, presmooth_options=(;),
        grid=nothing, basis=:pffr, penalty_order=nothing, criterion=:REML,
        lambda=nothing, covariance=:cluster)

Penalized function-on-function regression

    Yᵢ(s) = β₀(s) + Σₖ zᵢₖ βₖ(s) + Σⱼ ∫ Xᵢⱼ(t) βⱼ(t, s) dt + εᵢ(s),

the model of refund 0.1.40 `pffr(Y ~ z + ff(X1) + ff(X2))` with `method="REML"`.
`x` is one univariate functional dataset, a tuple/vector of them, or
`MultivariateFunctionalData`; `y` is a `GridFunctionalData` (missing values
allowed) or `IrregularFunctionalData` response, fitted at its observed points
with unit weights as `pffr` does (an irregular response is `pffr`'s `ydata`).
Basis responses throw; convert them with `to_grid`.

Each surface `βⱼ(t, s)` is a tensor product of cubic P-splines in `t` (the
covariate grid) and `s` (the response times) with one smoothing parameter per
direction, `ff()`'s default `te(bs="ps", m=list(c(2,1), c(2,1)), k=c(5,5))`.
As mgcv 1.9-3 `te` does, each marginal penalty (first differences of the
coefficients) is reparameterized to `k` function values at equally spaced
points of the margin's range and normalized to unit largest eigenvalue, so
the two penalties are `Sₜ ⊗ Gₛ` and `Gₜ ⊗ Sₛ` with `G = B(u)ᵀB(u)` at those
points `u`. The integral is the weighted sum `Σₜ Lₜ Xᵢⱼ(t) βⱼ(t, s)` over the
covariate grid. The intercept and scalar-term functions are as in
[`FunctionOnScalarRegression`](@ref) with `basis=:pffr`. Every smoothing
parameter is chosen by REML (`criterion=:REML`, the default) or GCV
(`criterion=:GCV`) on the stacked model with [`PenalizedLeastSquares`](@ref).

Options and the source of each default:

- `scalar_covariates`: `nothing`, a finite real vector or matrix (without an
  intercept column), or a Tables.jl table whose non-numeric columns are
  treatment-coded factors, as in [`FunctionOnScalarRegression`](@ref). The
  functional intercept is always fitted (`pffr`'s default formula).
- `k=(5, 5)`: marginal basis dimensions `(kₜ, kₛ)` of every surface (`ff()`'s
  `splinepars`); one integer sets both, and a vector gives one entry per
  functional covariate. The P-splines are cubic with first-order difference
  penalties (`m=c(2,1)`), with mgcv's knots on the covariate grid range and on
  the range of the observation times (for a grid response, the whole grid,
  as in `FunctionOnScalarRegression`).
- `integration=:quadrature`: the covariate grid's trapezoidal weights, equal
  to `ff(integration="trapezoidal")`. This departs from `ff()`'s default
  `"simpson"` for the reason given for [`PenalizedFunctionalRegression`](@ref):
  refund's Simpson weights divide by the number of grid points instead of the
  number of intervals. `:refund_simpson` and `:riemann` reproduce refund's
  formulas exactly; a vector supplies weights.
- `presmooth`, `presmooth_options`, `grid`: how each covariate becomes values
  on its grid, exactly as in [`PenalizedFunctionalRegression`](@ref) (complete
  grid data are used as observed; `ff()` itself takes only a complete matrix).
- `basis=:pffr`, `penalty_order=nothing`: the intercept and scalar-term bases
  and penalties, as in [`FunctionOnScalarRegression`](@ref) (`pffr`'s
  `bs.int` and `bs.yindex` by default).
- `lambda=nothing` selects every smoothing parameter; a vector fixes them, one
  per penalty in the order: intercept, each scalar coefficient, then `t` and
  `s` for each functional term.
- `covariance=:cluster` (`pffr`'s default `sandwich="cluster"`, the sandwich of
  [`FunctionOnScalarRegression`](@ref) clustering residuals by curve),
  `:bayesian` (`Vp`, `sandwich="none"`) or `:frequentist` (`Ve`).

`ff()` checks identifiability (`check.ident=TRUE`). The effective rank of each
covariate (components explaining 99.5% of `Σᵢ Xᵢ(t)²`), the log10 condition
number of the term's design and the overlap of the null spaces of `X` and of
the `t` penalty are stored on the term. Where refund would replace the term by
a constrained `ti()` smooth (overlap above 0.95 and log10 condition above 6),
the fit throws instead; the constrained term is not implemented. A functional
term whose integrals `Σₜ LₜXᵢ(t)` are equal for every curve throws (mgcv
would add a centering constraint). `ff()`'s `limits` (historical models) are
not supported.

`coef` orders coefficients as the intercept and scalar-term basis
coefficients, then each surface's `kₜkₛ` coefficients with the `t` index
varying slowest. `edf` holds the effective degrees of freedom of each
coefficient function, then each surface; `lambda` one value per penalty.
`dof` is the total effective degrees of freedom and `dof_residual` the number
of observations minus `dof`. Intervals use the normal distribution. All
choices are stored in `settings` and the functional terms in `terms`.
"""
struct PenalizedFunctionOnFunctionRegression{T<:AbstractFloat,B<:AbstractBasis,TM<:Tuple,
        R<:AbstractFunctionalData,F<:AbstractFunctionalData,S}
    bases::Vector{B}
    terms::TM
    coefficients::Vector{T}
    covariance::Matrix{T}
    design::Matrix{T}
    response::R
    fitted_values::F
    residual_values::R
    lambda::Vector{T}
    edf::Vector{T}
    penalty_order::Vector{Any}
    intercept::Bool
    coding::_FOSCoding
    settings::S
end

const _PFFR_COVARIANCES = (:cluster, :bayesian, :frequentist)
# refund 0.1.40 ff(): splinepars = list(bs="ps", m=list(c(2,1), c(2,1)), k=c(5,5)).
const _PFFR_K = (5, 5)

function _pffr_k(k, q)
    entry(value) = value isa Integer && !(value isa Bool) ? (Int(value), Int(value)) :
        value isa Tuple{Integer,Integer} && !any(v -> v isa Bool, value) ? (Int(value[1]), Int(value[2])) :
        throw(ArgumentError("k must be an integer or a tuple (k_t, k_s) of integers, or a vector of these"))
    values = if k isa AbstractVector
        length(k) == q || throw(DimensionMismatch("k must have one entry per functional covariate ($q)"))
        map(entry, Tuple(k))
    else
        ntuple(_ -> entry(k), q)
    end
    all(v -> all(>=(4), v), values) || throw(ArgumentError(
        "k must be at least 4 in each direction for cubic P-splines"))
    values
end

# mgcv 1.9-3 te(np=TRUE) margin: coefficients reparameterized to function
# values at k equally spaced points u of the margin's range, the penalty then
# normalized to unit largest eigenvalue. In the original B-spline coordinates
# the tensor penalties are (S/e) ⊗ G' and G ⊗ (S'/e') with G = B(u)ᵀB(u).
function _pffr_margin(basis::PSplineBasis{T}, range_) where {T}
    k = nbasis(basis)
    S = difference_penalty(basis, 1)
    S = Matrix{T}((S + transpose(S))/2)
    u = collect(Base.range(T(range_[1]), T(range_[2]); length=k))
    P = Matrix{T}(basis_matrix(basis, u))
    singular = svdvals(P)
    if last(singular)/first(singular) < eps(T)^T(0.66)
        # mgcv warns and skips the reparameterization of this margin.
        return (; penalty=S ./ eigmax(Symmetric(S)), gram=Matrix{T}(I, k, k), points=u, reparameterized=false)
    end
    XP = inv(P)
    normalization = eigmax(Symmetric(transpose(XP)*S*XP))
    G = transpose(P)*P
    (; penalty=S ./ normalization, gram=Matrix((G + transpose(G))/2), points=u, reparameterized=true)
end

# refund 0.1.40 ff(check.ident=TRUE) diagnostics.
function _pffr_identifiability(values::Matrix{T}, L, basis_t, grid) where {T}
    ev = svdvals(values).^2
    share = cumsum(ev) ./ sum(ev)
    effective_rank = max(1, something(findfirst(>=(T(0.995)), share), length(ev)))
    Bt = Matrix{T}(basis_matrix(basis_t, grid))
    ds = svdvals((values .* transpose(L))*Bt).^2
    log_condition = log10(maximum(ds)) - log10(minimum(ds))
    # Null space of X (curves in rows) and the integration-weighted null space
    # of the t penalty (MASS::Null, rank by R qr's relative tolerance 1e-7).
    decomposition = svd(values; full=true)
    s = decomposition.S
    rank = count(>(T(1e-7)*first(s)), s)
    NX = decomposition.V[:, rank+1:end]
    S = difference_penalty(basis_t, 1)
    eig = eigen(Symmetric(Matrix{T}(S)))
    nullspace = eig.vectors[:, eig.values .<= eps(T)^(3/4)*maximum(eig.values)]
    Npen = Diagonal(L)*Bt*nullspace
    overlap = size(NX, 2) == 0 || size(Npen, 2) == 0 ? zero(T) :
        sum(abs2, transpose(Matrix(svd(NX).U))*Matrix(svd(Npen).U))
    (; effective_rank, log_condition, null_overlap=overlap)
end

function _pffr_term(x, k, integration, presmooth, options, grid, s_range, ::Type{T}) where {T}
    options = _pfr_presmooth_options(presmooth, options)
    values, target, presmoother = _pfr_presmooth(x, presmooth, options, grid, T)
    all(isfinite, values) || throw(ArgumentError("functional covariate values must be finite"))
    length(target) >= 2 && first(target) < last(target) || throw(ArgumentError(
        "functional covariate grids must span an interval"))
    L = _pfr_integration(integration, target)
    basis_t = _pspline_basis((first(target), last(target)), k[1], Val(3), 1)
    basis_s = _pspline_basis(s_range, k[2], Val(3), 1)
    margin_t = _pffr_margin(basis_t, (first(target), last(target)))
    margin_s = _pffr_margin(basis_s, s_range)
    integrals = values*L
    centre = sum(integrals)/length(integrals)
    spread = sqrt(sum(abs2, integrals .- centre)/max(length(integrals) - 1, 1))
    spread > abs(centre)*eps(T)*1000 || throw(ArgumentError(
        "the functional covariate's integrals are equal for every curve, so its surface is confounded with the intercept"))
    diagnostics = _pffr_identifiability(values, L, basis_t, target)
    diagnostics.null_overlap > T(0.95) && diagnostics.log_condition > 6 && throw(ArgumentError(
        "the functional covariate's null space overlaps the t penalty's null space and the term is badly conditioned " *
        "(overlap $(diagnostics.null_overlap), log10 condition $(diagnostics.log_condition)); refund would constrain " *
        "the surface (Scheipl and Greven 2016), which is not implemented: lower k or use FPCRegression"))
    penalties = (kron(margin_t.penalty, margin_s.gram), kron(margin_t.gram, margin_s.penalty))
    (; grid=target, weights=L, integration=integration isa Symbol ? integration : :supplied,
        basis=basis_t, response_basis=basis_s, k=k, penalties,
        reparameterization_points=(margin_t.points, margin_s.points),
        reparameterized=(margin_t.reparameterized, margin_s.reparameterized),
        presmooth, presmooth_options=options, presmoother, training=deepcopy(x), values,
        sampling=sampling_design(x), diagnostics,
        integrated=(values .* transpose(L))*Matrix{T}(basis_matrix(basis_t, target)))
end

Base.@constprop :aggressive function fit(::Type{PenalizedFunctionOnFunctionRegression}, x,
        y::Union{GridFunctionalData,IrregularFunctionalData}; scalar_covariates=nothing,
        k=_PFFR_K, integration=:quadrature, presmooth=nothing, presmooth_options=(;),
        grid=nothing, basis=:pffr, penalty_order=nothing, criterion::Symbol=:REML,
        lambda=nothing, covariance::Symbol=:cluster)
    curves = _regression_inputs(x)
    n, p = nobs(y), length(curves)
    nobs(first(curves)) == n || throw(DimensionMismatch(
        "functional covariates and the response must have the same number of curves"))
    criterion in (:REML, :GCV) || throw(ArgumentError("criterion must be :REML or :GCV"))
    covariance in _PFFR_COVARIANCES || throw(ArgumentError(
        "covariance must be one of $(join(_PFFR_COVARIANCES, ", "))"))
    ks = _pffr_k(k, p)
    rules = integration isa AbstractVector{<:Real} ? ntuple(_ -> integration, p) :
        _regression_each(integration, p, "integration")
    smoothers = _regression_each(presmooth, p, "presmooth")
    options = _regression_each(presmooth_options, p, "presmooth_options")
    grids = grid isa AbstractVector{<:Real} ? ntuple(_ -> grid, p) : _regression_each(grid, p, "grid")
    T = _floattype(promote_type(eltype(y), _fos_scalar_type(scalar_covariates), map(eltype, curves)...))
    X, coding = _fos_design(scalar_covariates, n, T, true)
    s_range = _fos_time_range(y)
    terms = ntuple(j -> _pffr_term(curves[j], ks[j], rules[j], smoothers[j], options[j], grids[j],
        s_range, T), Val(p))
    _pffr_fit(terms, X, coding, y, basis, penalty_order, criterion, lambda, covariance)
end
fit(::Type{PenalizedFunctionOnFunctionRegression}, x, y; kwargs...) = throw(ArgumentError(
    "penalized function-on-function regression requires a GridFunctionalData or IrregularFunctionalData response; convert basis responses with to_grid"))

# Coefficient index ranges: scalar-term blocks, then one block per surface.
function _pffr_ranges(bases, terms)
    sizes = vcat(nbasis.(bases), [prod(term.k) for term in terms])
    stops = cumsum(sizes)
    [stop - size + 1:stop for (size, stop) in zip(sizes, stops)]
end

# Rows of the functional-term design at response times t for curve i:
# Σ_t L_t X_i(t) B_t(t) ⊗ B_s(s).
_pffr_block(term, i, t, ::Type{T}) where {T} =
    kron(reshape(term.integrated[i, :], 1, :), Matrix{T}(basis_matrix(term.response_basis, t)))

function _pffr_fit(terms::Tuple, X::Matrix{T}, coding, y, basis_input, penalty_order, criterion,
        lambda, covariance) where {T}
    n, q = size(X)
    n > q || throw(ArgumentError("penalized function-on-function regression requires more curves than scalar coefficients"))
    bases, basis_source = _fos_supplied_bases(y, basis_input, q, true, T)
    orders = _fos_orders(penalty_order, bases)
    ranges = _pffr_ranges(bases, terms)
    K = last(last(ranges))
    entries = Pair{Matrix{T},Vector{Int}}[_fos_penalty(bases[j], orders[j], T) => collect(ranges[j]) for j in 1:q]
    for (j, term) in enumerate(terms), P in term.penalties
        push!(entries, Matrix{T}(P) => collect(ranges[q + j]))
    end
    fixed = if lambda === nothing
        nothing
    else
        lambda isa AbstractVector{<:Real} && length(lambda) == length(entries) || throw(DimensionMismatch(
            "lambda must be a real vector with one value per penalty ($(length(entries)): intercept, scalar terms, then t and s per functional term)"))
        all(l -> !(l isa Bool) && isfinite(l) && l >= 0, lambda) ||
            throw(ArgumentError("lambda must be finite and nonnegative"))
        Vector{T}(lambda)
    end
    observations = _fos_observations(y, bases, T)
    counts = [length(last(o)) for o in observations]
    N = sum(counts)
    D = zeros(T, N, K)
    r = zeros(T, N)
    row = 0
    for (i, (Phis, v)) in enumerate(observations)
        rows = row+1:row+length(v)
        for j in 1:q
            D[rows, ranges[j]] .= X[i, j] .* Phis[j]
        end
        t, _ = y[i]
        for (j, term) in enumerate(terms)
            D[rows, ranges[q + j]] .= _pffr_block(term, i, t, T)
        end
        r[rows] .= v
        row += length(v)
    end
    all(isfinite, D) || throw(ArgumentError("regression design must be finite; rescale the inputs"))
    pls = fit(PenalizedLeastSquares, D, r; penalties=entries, criterion, lambda=fixed)
    isfinite(pls.sigma2) && sum(abs2, pls.residuals) > eps(T)^(3/2)*sum(abs2, r) || throw(ArgumentError(
        "the responses are fitted exactly, so the error variance and the coefficient covariance are not estimable"))
    b = pls.coef
    V = covariance === :bayesian ? pls.vcov_bayesian :
        covariance === :frequentist ? pls.vcov_frequentist : _fos_cluster_covariance(D, pls, counts)
    V = Matrix((V + transpose(V))/2)
    all(isfinite, b) && all(isfinite, V) || throw(ArgumentError(
        "regression coefficients and covariance must be finite; rescale the inputs"))
    residual_curves = Vector{T}[]
    row = 0
    for count in counts
        push!(residual_curves, pls.residuals[row+1:row+count])
        row += count
    end
    fitted_values, residual_values = _pffr_fitted(y, bases, terms, ranges, b, X, D, residual_curves, counts)
    edf = T[sum(pls.edf[range]) for range in ranges]
    solver = pls.settings
    settings = (; reference="refund 0.1.40 pffr with ff() terms; mgcv 1.9-3 gam and te",
        estimator=:penalized_least_squares, design_columns=copy(coding.labels),
        coefficient_basis=basis_source, intercept=true,
        lambda_selection=lambda === nothing ? criterion : :fixed,
        lambda_order=:intercept_scalar_terms_then_t_s_per_functional_term,
        lambda_status=copy(pls.lambda_status), criterion, criterion_value=pls.criterion_value,
        sigma2=pls.sigma2, edf_total=pls.edf_total,
        penalty_source=penalty_order === nothing ? :default_order : :supplied,
        surface_basis=:te_cubic_pspline_first_difference, k=map(t -> t.k, terms),
        integration=map(t -> t.integration, terms), presmooth=map(t -> t.presmooth, terms),
        presmooth_options=map(t -> t.presmooth_options, terms),
        coefficient_order=:scalar_term_bases_then_surfaces_t_slowest,
        loss=:observed_points, observation_weights=:unit, nobservations=N, covariance,
        residual_covariance=:not_estimated, interval_distribution=:normal,
        solver=(; initial_lambda=solver.initial_lambda, iterations=solver.iterations,
            convergence=solver.convergence, tol=solver.tol, maxiter=solver.maxiter,
            rank_tol=solver.rank_tol))
    PenalizedFunctionOnFunctionRegression(bases, terms, b, V, X, deepcopy(y), fitted_values,
        residual_values, copy(pls.lambda), edf, Vector{Any}(orders), true, coding, settings)
end

# Surface coefficients as a kₜ × kₛ matrix.
_pffr_theta(term, b) = permutedims(reshape(b, term.k[2], term.k[1]))

# Functional-term contribution on response points s for curves with integrated
# covariate rows C (curves × kₜ).
_pffr_term_curves(term, C, b, s, ::Type{T}) where {T} =
    C*_pffr_theta(term, b)*transpose(Matrix{T}(basis_matrix(term.response_basis, s)))

function _pffr_fitted(y::GridFunctionalData, bases, terms, ranges, b::Vector{T}, X, D, residual_curves, counts) where {T}
    q = length(bases)
    values = _fos_curves(bases, ranges[1:q], b, X, y.grid)
    for (j, term) in enumerate(terms)
        values .+= _pffr_term_curves(term, term.integrated, b[ranges[q + j]], y.grid, T)
    end
    residual = Matrix{Union{Missing,T}}(missing, nobs(y), length(y.grid))
    for i in 1:nobs(y)
        columns = y.observed === nothing ? axes(residual, 2) : findall(view(y.observed, i, :))
        residual[i, columns] .= residual_curves[i]
    end
    GridFunctionalData(values, y.grid; ids=y.ids), GridFunctionalData(residual, y.grid; ids=y.ids)
end
_pffr_fitted(y::IrregularFunctionalData, bases, terms, ranges, b, X, D, residual_curves, counts) =
    _fos_fitted(y, bases, ranges, b, X, D, residual_curves, counts)

# ---------------------------------------------------------------------------
# Accessors.

"""Return the stacked coefficients: intercept and scalar-term basis coefficients, then each surface's (`t` index slowest)."""
coef(f::PenalizedFunctionOnFunctionRegression) = copy(f.coefficients)
"""Return the covariance of `coef(f)` chosen by `covariance` (`f.settings.covariance`)."""
vcov(f::PenalizedFunctionOnFunctionRegression) = copy(f.covariance)
"""Return standard errors of `coef(f)`."""
stderror(f::PenalizedFunctionOnFunctionRegression) = sqrt.(max.(diag(f.covariance), zero(eltype(f.covariance))))
"""Return the number of response curves."""
nobs(f::PenalizedFunctionOnFunctionRegression) = size(f.design, 1)
"""Return the total effective degrees of freedom of the coefficient functions and surfaces."""
dof(f::PenalizedFunctionOnFunctionRegression) = f.settings.edf_total
"""Return the number of observed response values minus the total effective degrees of freedom."""
dof_residual(f::PenalizedFunctionOnFunctionRegression) = f.settings.nobservations - f.settings.edf_total
"""
Return fitted response curves: on the whole response grid for grid responses,
and at the observation times for irregular responses.
"""
fitted(f::PenalizedFunctionOnFunctionRegression) = deepcopy(f.fitted_values)
"""Return response-minus-fitted values at the observed points (missing grid values stay missing)."""
residuals(f::PenalizedFunctionOnFunctionRegression) = deepcopy(f.residual_values)

function _pffr_response_grid(f, response_grid, surface::Bool)
    T = eltype(f.coefficients)
    # The surfaces' response basis spans the observation times.
    a, b = domain(first(f.terms).response_basis)
    g = if response_grid !== nothing
        response_grid
    elseif f.response isa GridFunctionalData
        f.response.grid
    elseif surface
        # refund 0.1.40 coef.pffr: n2 = 40 points per axis for surfaces.
        collect(range(a, b; length=40))
    else
        # predict.pffr evaluates at the observed response index values.
        sort!(unique(f.response.times))
    end
    g isa AbstractVector{<:Real} || throw(ArgumentError("response_grid must be a real vector"))
    out = Vector{T}(g)
    _validate_strict_grid(out)
    all(t -> a <= t <= b, out) || throw(ArgumentError("response_grid must lie inside the response domain [$a, $b]"))
    out
end

function _pffr_covariate_grids(f, grid)
    p = length(f.terms)
    grids = grid === nothing ? map(t -> t.grid, f.terms) :
        p == 1 && grid isa AbstractVector{<:Real} ? (grid,) : _regression_each(grid, p, "grid")
    T = eltype(f.coefficients)
    map(f.terms, grids) do term, g
        g isa AbstractVector{<:Real} || throw(ArgumentError("covariate grids must be real vectors"))
        out = Vector{T}(g)
        _validate_strict_grid(out)
        a, b = domain(term.basis)
        all(t -> a <= t <= b, out) || throw(ArgumentError("covariate grids must lie inside the covariate range [$a, $b]"))
        out
    end
end

"""
    coefficient_functions(f::PenalizedFunctionOnFunctionRegression; grid=nothing, response_grid=nothing)

Return `(; scalar, functional)`: `scalar` is a tuple of one-curve
`BasisFunctionalData`, the intercept and each scalar coefficient function in
its basis; `functional` is a tuple with one surface per functional covariate,
`(; grid, response_grid, values)` with `values[i, j] = βⱼ(grid[i],
response_grid[j])` (covariate grid in rows). The grids default as in
[`coefficient_bands`](@ref).
"""
function coefficient_functions(f::PenalizedFunctionOnFunctionRegression; grid=nothing, response_grid=nothing)
    T = eltype(f.coefficients)
    q = length(f.bases)
    ranges = _pffr_ranges(f.bases, f.terms)
    scalar = Tuple(BasisFunctionalData(reshape(f.coefficients[ranges[j]], 1, :), f.bases[j]) for j in 1:q)
    s = _pffr_response_grid(f, response_grid, true)
    functional = map(f.terms, _pffr_covariate_grids(f, grid), ntuple(identity, length(f.terms))) do term, g, j
        Bt = Matrix{T}(basis_matrix(term.basis, g))
        Bs = Matrix{T}(basis_matrix(term.response_basis, s))
        (; grid=g, response_grid=copy(s), values=Bt*_pffr_theta(term, f.coefficients[ranges[q + j]])*transpose(Bs))
    end
    (; scalar, functional)
end

"""
    coefficient_bands(f::PenalizedFunctionOnFunctionRegression; alpha=0.05, grid=nothing,
        response_grid=nothing)

Pointwise normal intervals from `vcov(f)` (the covariance chosen at the fit,
`f.settings.covariance`). Return `(; scalar, functional)`: `scalar` has one
entry per intercept or scalar coefficient function (`grid`, `estimate`, `se`,
`lower`, `upper`, `alpha`, `critical_value`, `settings`), `functional` one per
surface with `grid` (covariate), `response_grid` and `t × s` arrays.
`response_grid=nothing` uses the response grid of a grid response and 40
equally spaced points of the response domain for an irregular response
(refund 0.1.40 `coef.pffr`'s `n2=40`); `grid=nothing` uses each covariate grid.
One grid or a tuple of grids may be supplied.
"""
function coefficient_bands(f::PenalizedFunctionOnFunctionRegression; alpha=0.05, grid=nothing,
        response_grid=nothing)
    T = eltype(f.coefficients)
    q = length(f.bases)
    ranges = _pffr_ranges(f.bases, f.terms)
    s = _pffr_response_grid(f, response_grid, true)
    scalar = Tuple(begin
        Phi = Matrix{T}(basis_matrix(f.bases[j], s))
        index = ranges[j]
        variance = vec(sum((Phi*f.covariance[index, index]).*Phi; dims=2))
        merge((; grid=copy(s)), _fos_interval(f, Phi*f.coefficients[index], variance, alpha, :coefficient))
    end for j in 1:q)
    functional = map(f.terms, _pffr_covariate_grids(f, grid), ntuple(identity, length(f.terms))) do term, g, j
        Bt = Matrix{T}(basis_matrix(term.basis, g))
        Bs = Matrix{T}(basis_matrix(term.response_basis, s))
        index = ranges[q + j]
        estimate = Bt*_pffr_theta(term, f.coefficients[index])*transpose(Bs)
        V = f.covariance[index, index]
        variance = similar(estimate)
        for a in eachindex(g)
            A = kron(reshape(Bt[a, :], 1, :), Bs)
            variance[a, :] = vec(sum((A*V).*A; dims=2))
        end
        merge((; grid=g, response_grid=copy(s)),
            _fos_interval(f, estimate, variance, alpha, :coefficient_surface))
    end
    (; scalar, functional)
end

"""
    predict(f::PenalizedFunctionOnFunctionRegression, x; scalar_covariates=nothing,
        interval=:none, alpha=0.05, grid=nothing)

Predict response curves for new functional covariates (presmoothed with the
fitted presmoothers and integrated with the stored weights on the fitted
grids, as refund 0.1.40 `predict.pffr`) and scalar covariates coded as in the
fit. `grid=nothing` uses the response grid of a grid response and the sorted
observed response times of an irregular response (`pffr`'s stored `yind`).
With `interval=:none` return `GridFunctionalData` on `grid`; `:confidence`
gives pointwise normal intervals for the mean curve from `vcov(f)`, and
`:prediction` adds the error variance `f.settings.sigma2`, both with `grid`,
`estimate`, `se`, `lower`, `upper`, `alpha`, `critical_value` and `settings`
(subject × grid arrays).
"""
function predict(f::PenalizedFunctionOnFunctionRegression, x; scalar_covariates=nothing,
        interval=:none, alpha=0.05, grid=nothing)
    interval in (:none, :confidence, :prediction) || throw(ArgumentError(
        "interval must be :none, :confidence, or :prediction"))
    T = eltype(f.coefficients)
    curves = _regression_inputs(x)
    length(curves) == length(f.terms) || throw(DimensionMismatch(
        "prediction functional covariate count must match the regression fit"))
    m = nobs(first(curves))
    Z = if f.coding.kind === :none
        scalar_covariates === nothing || throw(ArgumentError("the model has no scalar covariates"))
        ones(T, m, 1)
    else
        scalar_covariates === nothing && throw(ArgumentError("the model needs scalar_covariates for prediction"))
        _fos_new_design(f, scalar_covariates)
    end
    size(Z, 1) == m || throw(DimensionMismatch("scalar covariates must have one row per prediction curve"))
    integrated = map(f.terms, curves) do term, c
        values = _pfr_new_values(term, c, T)
        all(isfinite, values) || throw(ArgumentError("prediction covariate values must be finite"))
        (values .* transpose(term.weights))*Matrix{T}(basis_matrix(term.basis, term.grid))
    end
    g = _pffr_response_grid(f, grid, false)
    q = length(f.bases)
    ranges = _pffr_ranges(f.bases, f.terms)
    estimate = _fos_curves(f.bases, ranges[1:q], f.coefficients, Z, g)
    for (j, term) in enumerate(f.terms)
        estimate .+= _pffr_term_curves(term, integrated[j], f.coefficients[ranges[q + j]], g, T)
    end
    interval === :none && return GridFunctionalData(estimate, g)
    Phis = [Matrix{T}(basis_matrix(basis, g)) for basis in f.bases]
    Bs = [Matrix{T}(basis_matrix(term.response_basis, g)) for term in f.terms]
    variance = similar(estimate)
    for i in 1:m
        A = hcat((Z[i, j] .* Phis[j] for j in 1:q)...,
            (kron(reshape(integrated[j][i, :], 1, :), Bs[j]) for j in eachindex(f.terms))...)
        variance[i, :] = vec(sum((A*f.covariance).*A; dims=2))
    end
    interval === :prediction && (variance .+= f.settings.sigma2)
    merge((; grid=g), _fos_interval(f, estimate, variance, alpha, interval))
end

function Base.show(io::IO, f::PenalizedFunctionOnFunctionRegression)
    print(io, "PenalizedFunctionOnFunctionRegression(n=", nobs(f), ", functional terms=", length(f.terms),
        ", scalar coefficients=", length(f.bases), ", k=", map(t -> t.k, f.terms),
        ", lambda=", f.lambda, ", edf=", f.edf, ", total edf=", dof(f),
        ", sigma2=", f.settings.sigma2, ", settings=", f.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::PenalizedFunctionOnFunctionRegression) = show(io, f)
