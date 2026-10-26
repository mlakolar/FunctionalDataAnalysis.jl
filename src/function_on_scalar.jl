# How scalar covariates become design columns: a vector or matrix, or a table
# with numeric and treatment-coded factor columns.
struct _FOSCoding
    kind::Symbol
    names::Vector{Symbol}
    factor::Vector{Bool}
    levels::Vector{Vector{Any}}
    full::Vector{Bool}
    labels::Vector{String}
end

"""
    FunctionOnScalarRegression
    fit(FunctionOnScalarRegression, z, y; intercept=true, lambda=nothing,
        criterion=:REML, shared_lambda=false, basis=nothing, penalty_order=nothing,
        covariance=nothing, residual_covariance=nothing, method=nothing)

Function-on-scalar regression `Y_i(t) = sum_k x_ik beta_k(t) + e_i(t)`, with
`x_i = [1, z_i]` when `intercept=true`. `z` is a finite real vector or matrix,
a Tables.jl table, or `nothing` (with an intercept, a mean-only model). Table
columns with real entries are numeric; other columns are factors with levels
`sort(unique(column))`, coded by treatment contrasts against the first level as
R's `model.matrix` does (named `"\$(column)\$(level)"`; without an intercept
the first factor keeps every level). Missing entries throw.

# Penalized REML fit (`lambda=nothing`, the default)

Each `beta_k` is a penalized spline and the smoothing parameters are chosen by
REML (`criterion=:REML`) or GCV (`criterion=:GCV`) with the package's
[`PenalizedLeastSquares`](@ref) solver, one `lambda` per coefficient function
as refund 0.1.40 `pffr` gives each scalar term (`shared_lambda=true` uses one
`lambda` for all, fosr's `multi.sp=FALSE`). Errors are working independent with
a common variance, the model of `pffr`'s `gam` fit.

- Grid (possibly with missing values) and irregular responses are fitted at
  their observed points with unit weights, as `pffr` and `fosr` do. `basis`
  defaults to fosr's 15 cubic B-splines (`nbasis=15`, `norder=4`) on the
  response domain. `basis=:pffr` uses `pffr`'s default bases: a
  [`PSplineBasis`](@ref) with 20 cubic functions for the intercept and 5 for
  every other coefficient (`bs.int` and `bs.yindex`, mgcv `ps` with
  `m=c(2,1)`), with mgcv's knots on the range of the observation times (for
  a grid response, the range of the whole grid, so that fitted curves on the
  grid stay inside the basis domain even when an end point is missing for
  every curve). Any basis whose domain contains the observation times (for a
  grid response, the whole grid) may be supplied, or one per coefficient (a
  tuple or vector sharing one domain).
- Basis responses are fitted in the L2 norm through the exact Gram matrix `G`,
  as fosr does for `fd` responses (pseudo-observations `G^(1/2) c_i`), with
  each `beta_k` in the response basis. Supplying `basis` throws.

`penalty_order=nothing` uses the integrated squared derivative of order
`norder-2` for B-splines (fosr's `pen.order=2` for cubic splines, mgcv's `bs`
penalty `m=c(3,2)`), the coefficient differences of the basis's
`difference_order` for P-splines ([`difference_penalty`](@ref), mgcv's `ps`
penalty), harmonic acceleration for Fourier and order 0 for monomial bases; an
integer (a difference order for P-splines), `:harmonic_acceleration` (Fourier)
or one entry per coefficient overrides it.

`covariance` selects the coefficient covariance used by `vcov`, the bands and
the intervals:

- `:cluster` (default; refund 0.1.40 `pffr`'s default `sandwich="cluster"`):
  `N/(N-1) A⁻¹ (Σ_i X_iᵀr_i r_iᵀX_i) A⁻¹ + Vp - Ve`, clustering the residuals
  `r_i` by curve, with `A = XᵀX + S_λ`. It accounts for correlation of the
  errors within a curve. This departs from the posterior covariance `Vp`
  originally specified for the bands: `Vp` treats the points of one curve as
  independent and understates the uncertainty when residual curves are
  correlated, which is why refund 0.1.40 made the sandwich `pffr`'s default.
- `:bayesian`: the posterior covariance `Vp = A⁻¹σ²` under independent errors
  (`pffr`'s `sandwich="none"`); `:frequentist`: `Ve = A⁻¹XᵀXA⁻¹σ²`.
- `:pooled`: fosr's residual-covariance sandwich
  `A⁻¹ (Σ_i X_iᵀ Σ̂ X_i) A⁻¹` with `Σ̂` the covariance of the residual
  curves divided by `n`, for complete grids and basis responses.

The refund comparisons in the test suite use `:bayesian`, `:frequentist` and
`:cluster` for `pffr` with the package's B-spline basis, `:bayesian` and
`:cluster` for `pffr` with `basis=:pffr`, and `:pooled` for `fosr`.

`residual_covariance` (the field) stores `Σ̂` on the response grid for complete
grids, the covariance of the residual basis coefficients (divided by `n`) for
basis responses, and a `0×0` matrix for irregular or incomplete grids, where
it is not estimated. Intervals use the normal distribution.

# Fixed-lambda fit (`lambda` supplied)

`method=nothing` (the default) fits a `BasisFunctionalData` response with a
supplied `lambda` (one nonnegative number or one per coefficient) by
`method=:fRegress`, and everything else by `method=:penalized`, which with a
supplied `lambda` evaluates the penalized fit at that fixed `lambda`.
`method=:fRegress` follows fda 6.3.0 `fRegress` (its `fRegress.fd` method) with
scalar covariates and a fixed `fdPar` lambda: the coefficients `b = vec([b_1 ... b_q])` solve
`(X'X ⊗ G + blockdiag(lambda_k R_k)) b = vec(G Y' X)`. Standard errors follow
`fRegress.stderr`: `Var(b) = C^(-1) (X'X ⊗ G Σ_c G) C^(-1)`, with `Σ_c` the
covariance of one response's basis coefficients. `residual_covariance=:residual`
(the default) estimates `Σ_c = E'E/(n-q)` from the coefficient residuals with
unpenalized residual degrees of freedom; a symmetric positive semidefinite
`nbasis × nbasis` matrix supplies it instead. Intervals use Student t with
`n-q` degrees of freedom for the estimate and the normal distribution for a
supplied matrix. `fRegress.stderr` integrates with a Riemann sum on 501 points
that gives full weight to both endpoints; the package uses exact inner products.
`criterion`, `shared_lambda`, `basis` and `covariance` do not apply.

Grid and irregular responses with a supplied `lambda` use the penalized fit at
that fixed `lambda` (one value with `shared_lambda=true`).

All choices are stored in `settings` and shown; `bases` holds the coefficient
basis of each coefficient function. `edf` holds the effective
degrees of freedom of each coefficient function (block traces of the influence
matrix), `lambda` one value per coefficient.
"""
struct FunctionOnScalarRegression{T<:AbstractFloat,B<:AbstractBasis,R<:AbstractFunctionalData,
        F<:AbstractFunctionalData,S}
    bases::Vector{B}
    coefficients::Vector{T}
    covariance::Matrix{T}
    design::Matrix{T}
    response::R
    fitted_values::F
    residual_values::R
    residual_covariance::Matrix{T}
    lambda::Vector{T}
    edf::Vector{T}
    penalty_order::Vector{Any}
    intercept::Bool
    coding::_FOSCoding
    settings::S
end

_fos_default_order(b::BSplineBasis) = max(0, b.order - 2)
_fos_default_order(::FourierBasis) = :harmonic_acceleration
_fos_default_order(::MonomialBasis) = 0
_fos_default_order(b::PSplineBasis) = b.difference_order
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
    # P-splines penalize coefficient differences (mgcv's ps penalty).
    R = basis isa PSplineBasis ? difference_penalty(basis, order) : Matrix{T}(penalty(basis, order))
    R = Matrix{T}(R)
    (R + transpose(R))/2
end

function _fos_orders(penalty_order, bases)
    penalty_order === nothing ? Any[_fos_default_order(b) for b in bases] :
        _fos_each(penalty_order, length(bases), "penalty_order")
end

# Coefficient index ranges of the stacked vector vec([b_1 ... b_q]).
function _fos_ranges(bases)
    stops = cumsum(nbasis.(bases))
    [stop - nbasis(b) + 1:stop for (b, stop) in zip(bases, stops)]
end

_fos_common_basis(bases) = all(==(first(bases)), bases)

Base.@constprop :aggressive function fit(::Type{FunctionOnScalarRegression}, z, y::_RegressionCurve;
        intercept=true, lambda=nothing, criterion::Symbol=:REML, shared_lambda::Bool=false,
        basis=nothing, penalty_order=nothing, covariance=nothing, residual_covariance=nothing,
        method=nothing)
    intercept isa Bool || throw(ArgumentError("intercept must be a Bool"))
    n = nobs(y)
    z === nothing && !intercept && throw(ArgumentError(
        "a model without scalar covariates needs intercept=true"))
    T = _floattype(promote_type(eltype(y), _fos_scalar_type(z)))
    X, coding = _fos_design(z, n, T, intercept)
    n > size(X, 2) || throw(ArgumentError("function-on-scalar regression requires more curves than coefficients"))
    method === nothing || method in (:penalized, :fRegress) || throw(ArgumentError(
        "method must be nothing, :penalized or :fRegress"))
    fregress = method === nothing ? lambda !== nothing && y isa BasisFunctionalData : method === :fRegress
    if fregress
        y isa BasisFunctionalData && lambda !== nothing || throw(ArgumentError(
            "method=:fRegress needs a BasisFunctionalData response and a supplied lambda"))
        criterion === :REML && !shared_lambda && basis === nothing && covariance === nothing ||
            throw(ArgumentError("criterion, shared_lambda, basis and covariance apply to the penalized fit; the fixed-lambda basis fit follows fRegress"))
        return _fos_fregress(X, coding, y, intercept, lambda, penalty_order,
            residual_covariance === nothing ? :residual : residual_covariance)
    end
    residual_covariance === nothing || throw(ArgumentError(
        "residual_covariance applies to the fixed-lambda basis fit; the penalized fit estimates it"))
    _fos_penalized(X, coding, y, intercept, lambda, criterion, shared_lambda, basis,
        penalty_order, covariance === nothing ? :cluster : covariance)
end
fit(::Type{FunctionOnScalarRegression}, z, y; kwargs...) = throw(ArgumentError(
    "function-on-scalar regression requires a GridFunctionalData, IrregularFunctionalData or BasisFunctionalData response"))

# ---------------------------------------------------------------------------
# Scalar design: a vector or matrix, or a Tables.jl table with factors.

_fos_scalar_type(::Nothing) = Bool
_fos_scalar_type(z::_RegressionScalars) = Base.nonmissingtype(eltype(z))
_fos_scalar_type(z) = Bool

function _fos_design(z, n, ::Type{T}, intercept) where {T}
    if z === nothing || z isa _RegressionScalars
        scalars = _regression_scalars(z, n, T)
        p = size(scalars, 2)
        names = ["z$j" for j in 1:p]
        coding = _FOSCoding(z === nothing ? :none : :matrix, Symbol[], Bool[], Vector{Any}[],
            Bool[], intercept ? vcat("(Intercept)", names) : names)
    else
        columns = _fos_table(z)
        coding = _fos_table_coding(columns, intercept)
        scalars = _fos_table_matrix(columns, coding, T)
        size(scalars, 1) == n || throw(DimensionMismatch("scalar covariates must have one row per curve"))
    end
    X = intercept ? hcat(ones(T, n), scalars) : scalars
    X, coding
end

function _fos_table(z)
    Tables.istable(z) || throw(ArgumentError(
        "scalar covariates must be a finite real vector or matrix, a Tables.jl table, or nothing"))
    columns = Tables.columntable(z)
    isempty(keys(columns)) && throw(ArgumentError("the scalar covariate table has no columns"))
    columns
end

function _fos_table_coding(columns, intercept)
    names = collect(Symbol, keys(columns))
    factor, levels, full = Bool[], Vector{Any}[], Bool[]
    labels = intercept ? ["(Intercept)"] : String[]
    first_factor = true
    for name in names
        column = columns[name]
        any(ismissing, column) && throw(ArgumentError(
            "scalar covariate column $name contains missing values; no curves are dropped"))
        if all(v -> v isa Real, column)
            push!(factor, false); push!(levels, Any[]); push!(full, false)
            push!(labels, String(name))
        else
            observed = try
                sort!(unique(column))
            catch
                throw(ArgumentError("factor column $name must have sortable levels"))
            end
            length(observed) >= 2 || throw(ArgumentError("factor column $name needs at least two levels"))
            every = !intercept && first_factor
            first_factor = false
            push!(factor, true); push!(levels, Vector{Any}(observed)); push!(full, every)
            append!(labels, ["$(name)$(level)" for level in (every ? observed : observed[2:end])])
        end
    end
    _FOSCoding(:table, names, factor, levels, full, labels)
end

function _fos_table_matrix(columns, coding::_FOSCoding, ::Type{T}) where {T}
    blocks = Matrix{T}[]
    for (j, name) in enumerate(coding.names)
        haskey(columns, name) || throw(ArgumentError("scalar covariate table lacks column $name"))
        column = columns[name]
        any(ismissing, column) && throw(ArgumentError(
            "scalar covariate column $name contains missing values; no curves are dropped"))
        if coding.factor[j]
            all(v -> any(isequal(v), coding.levels[j]), column) || throw(ArgumentError(
                "factor column $name has levels not seen in the fit"))
            used = coding.full[j] ? coding.levels[j] : coding.levels[j][2:end]
            push!(blocks, T[isequal(v, level) for v in column, level in used])
        else
            all(v -> v isa Real, column) || throw(ArgumentError("scalar covariate column $name must be numeric"))
            values = T[v for v in column]
            all(isfinite, values) || throw(ArgumentError("scalar covariate column $name must be finite"))
            push!(blocks, reshape(values, :, 1))
        end
    end
    reduce(hcat, blocks)
end

# Prediction design with the fitted coding.
function _fos_new_design(f::FunctionOnScalarRegression{T}, z) where {T}
    coding = f.coding
    scalars = if coding.kind === :table
        z isa _RegressionScalars && throw(ArgumentError(
            "the model was fitted with a covariate table; predict with a table"))
        _fos_table_matrix(_fos_table(z), coding, T)
    else
        z isa _RegressionScalars || throw(ArgumentError(
            "scalar covariates must be a real vector or matrix; use an m × 0 matrix for an intercept-only model"))
        _regression_scalars(z, size(z, 1), T)
    end
    X = f.intercept ? hcat(ones(T, size(scalars, 1)), scalars) : scalars
    size(X, 2) == _fos_q(f) || throw(DimensionMismatch(
        "prediction covariate columns must match the fitted model"))
    X
end

# ---------------------------------------------------------------------------
# Fixed lambda: fda fRegress.

function _fos_fregress(X::Matrix{T}, coding, y::BasisFunctionalData, intercept, lambda, penalty_order,
        residual_covariance) where {T}
    n, q = size(X)
    basis = y.basis
    k = nbasis(basis)
    lambdas = _fos_each(lambda, q, "lambda")
    all(l -> l isa Real && !(l isa Bool) && isfinite(l) && l >= 0, lambdas) ||
        throw(ArgumentError("lambda must be finite and nonnegative"))
    orders = _fos_orders(penalty_order, fill(basis, q))
    Y = Matrix{T}(y.coefs)
    G = Matrix{T}(gram(basis))
    G = (G + transpose(G))/2
    system = kron(transpose(X)*X, G)
    for j in 1:q
        index = (j-1)*k+1:j*k
        system[index, index] .+= T(lambdas[j]) .* _fos_penalty(basis, orders[j], T)
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
    influence = inverse*kron(transpose(X)*X, G)
    edf = T[tr(influence[(j-1)*k+1:j*k, (j-1)*k+1:j*k]) for j in 1:q]
    settings = (; reference="fda 6.3.0 fRegress (fRegress.fd) and fRegress.stderr",
        estimator=:fRegress, design_columns=copy(coding.labels),
        coefficient_basis=:response_basis, intercept, lambda_selection=:fixed,
        penalty_source=penalty_order === nothing ? :fdPar_default : :supplied,
        inner_products=:exact_gram, covariance=:fRegress_stderr_exact,
        residual_covariance=covariance_source, residual_dof=n - q,
        interval_distribution=covariance_source === :supplied ? :normal : :student_t,
        fitted=:exact_in_response_basis, observation_weights=:unit)
    response = BasisFunctionalData(copy(Y), basis; ids=y.ids)
    FunctionOnScalarRegression(fill(basis, q), b, covariance, X, response,
        BasisFunctionalData(fitted_values, basis; ids=y.ids), BasisFunctionalData(residual, basis; ids=y.ids),
        sigma, T.(lambdas), edf, Vector{Any}(orders), intercept, coding, settings)
end

# ---------------------------------------------------------------------------
# Penalized least squares with REML/GCV (refund pffr/fosr).

const _FOS_DEFAULT_NBASIS = 15
# refund 0.1.40 pffr: bs.int = list(bs="ps", k=20, m=c(2,1)) for the functional
# intercept and bs.yindex = list(bs="ps", k=5, m=c(2,1)) for every other term.
const _FOS_PFFR_NBASIS = (intercept=20, term=5)
const _FOS_COVARIANCES = (:cluster, :bayesian, :frequentist, :pooled)

# Per-curve evaluation matrices (one per coefficient; equal bases share one
# matrix) and observations of the stacked model.
function _fos_observations(y::Union{GridFunctionalData,IrregularFunctionalData}, bases, ::Type{T}) where {T}
    unique_bases = unique(bases)
    # Fitted curves and default bands of a grid response live on the whole grid.
    if y isa GridFunctionalData
        lo, hi = domain(first(bases))
        lo <= first(y.grid) && last(y.grid) <= hi || throw(ArgumentError(
            "the response grid [$(first(y.grid)), $(last(y.grid))] is not inside the coefficient basis domain [$lo, $hi]"))
    end
    index = Int[something(findfirst(==(b), unique_bases)) for b in bases]
    out = Tuple{Vector{Matrix{T}},Vector{T}}[]
    for i in 1:nobs(y)
        t, v = y[i]
        matrices = Matrix{T}[]
        for basis in unique_bases
            lo, hi = domain(basis)
            all(s -> lo <= s <= hi, t) || throw(ArgumentError(
                "observation times of curve $i lie outside the coefficient basis domain"))
            push!(matrices, Matrix{T}(basis_matrix(basis, t)))
        end
        push!(out, (matrices[index], Vector{T}(v)))
    end
    out
end

function _fos_observations(y::BasisFunctionalData, bases, ::Type{T}) where {T}
    G = Matrix{T}(gram(y.basis))
    root = _fos_sqrt(Symmetric((G + transpose(G))/2))
    roots = fill(root, length(bases))
    [(roots, root*Vector{T}(y.coefs[i, :])) for i in 1:nobs(y)]
end

function _fos_sqrt(G::Symmetric{T}) where {T}
    decomposition = eigen(G)
    minimum(decomposition.values) > eps(T)*maximum(decomposition.values)*size(G, 1) ||
        throw(ArgumentError("the response basis Gram matrix is singular"))
    root = decomposition.vectors*Diagonal(sqrt.(decomposition.values))*transpose(decomposition.vectors)
    Matrix((root + transpose(root))/2)
end

# Smallest and largest observation time, the range mgcv builds its knots on.
# A grid response uses its whole grid (identical when the end points are
# observed), since fitted curves are returned on that grid.
function _fos_time_range(y::GridFunctionalData)
    first(y.grid) < last(y.grid) || throw(ArgumentError("basis=:pffr needs observation times spanning an interval"))
    (first(y.grid), last(y.grid))
end
function _fos_time_range(y)
    lo, hi = typemax(eltype(y)), typemin(eltype(y))
    for i in 1:nobs(y)
        t, _ = y[i]
        isempty(t) && continue
        lo, hi = min(lo, first(t)), max(hi, last(t))
    end
    lo < hi || throw(ArgumentError("basis=:pffr needs observation times spanning an interval"))
    (lo, hi)
end

function _fos_coefficient_bases(y::BasisFunctionalData, basis, q, intercept, ::Type)
    basis === nothing || throw(ArgumentError(
        "a basis response uses its own basis for the coefficients; basis cannot be supplied"))
    fill(y.basis, q), :response_basis
end
_fos_coefficient_bases(y, basis, q, intercept, ::Type{T}) where {T} =
    _fos_supplied_bases(y, basis, q, intercept, T)
_fos_supplied_bases(y, ::Nothing, q, intercept, ::Type) =
    fill(BSplineBasis(4, domain(y), _FOS_DEFAULT_NBASIS), q), :fosr_default
_fos_supplied_bases(y, basis::AbstractBasis, q, intercept, ::Type) = fill(basis, q), :supplied
function _fos_supplied_bases(y, basis::Symbol, q, intercept, ::Type)
    basis === :pffr || throw(ArgumentError("basis=:$basis is not available; the only symbol is :pffr"))
    range = _fos_time_range(y)
    bases = fill(_pspline_basis(range, _FOS_PFFR_NBASIS.term, Val(3), 1), q)
    intercept && (bases[1] = _pspline_basis(range, _FOS_PFFR_NBASIS.intercept, Val(3), 1))
    bases, :pffr_default
end
function _fos_supplied_bases(y, basis, q, intercept, ::Type)
    basis isa Union{Tuple,AbstractVector} && all(b -> b isa AbstractBasis, basis) || throw(ArgumentError(
        "basis must be nothing, :pffr, an AbstractBasis, or one AbstractBasis per coefficient"))
    length(basis) == q || throw(DimensionMismatch("basis must be one basis or one per coefficient ($q)"))
    bases = [b for b in basis]
    all(b -> domain(b) == domain(first(bases)), bases) || throw(ArgumentError(
        "the coefficient bases must share one domain"))
    bases, :supplied
end

# Common evaluation matrices of every curve, or nothing (incomplete or irregular).
_fos_common(y::BasisFunctionalData, observations) = first(first(observations))
_fos_common(y::GridFunctionalData, observations) = y.observed === nothing ? first(first(observations)) : nothing
_fos_common(y, observations) = nothing

# Curves sum_j X[:, j] beta_j(t) on the points t.
function _fos_curves(bases, ranges, b::Vector{T}, X, t) where {T}
    out = zeros(T, size(X, 1), length(t))
    for (j, basis) in enumerate(bases)
        out .+= view(X, :, j) .* transpose(Matrix{T}(basis_matrix(basis, t))*b[ranges[j]])
    end
    out
end

function _fos_penalized(X::Matrix{T}, coding, y, intercept, lambda, criterion, shared, basis_input,
        penalty_order, covariance) where {T}
    n, q = size(X)
    covariance in _FOS_COVARIANCES || throw(ArgumentError(
        "covariance must be one of $(join(_FOS_COVARIANCES, ", "))"))
    bases, basis_source = _fos_coefficient_bases(y, basis_input, q, intercept, T)
    ranges = _fos_ranges(bases)
    K = last(last(ranges))
    orders = _fos_orders(penalty_order, bases)
    penalties = [_fos_penalty(bases[j], orders[j], T) for j in 1:q]
    entries = if shared
        full = zeros(T, K, K)
        for j in 1:q
            full[ranges[j], ranges[j]] .= penalties[j]
        end
        [full => collect(1:K)]
    else
        [penalties[j] => collect(ranges[j]) for j in 1:q]
    end
    fixed = if lambda === nothing
        nothing
    else
        values = shared ? (lambda isa Real ? [lambda] : lambda) : _fos_each(lambda, q, "lambda")
        values isa AbstractVector && length(values) == length(entries) || throw(DimensionMismatch(
            "lambda must be one value with shared_lambda=true"))
        all(l -> l isa Real && !(l isa Bool) && isfinite(l) && l >= 0, values) ||
            throw(ArgumentError("lambda must be finite and nonnegative"))
        T[l for l in values]
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
        r[rows] .= v
        row += length(v)
    end
    pls = fit(PenalizedLeastSquares, D, r; penalties=entries, criterion, lambda=fixed)
    # Residuals at rounding level leave no error variance for the covariances.
    isfinite(pls.sigma2) && sum(abs2, pls.residuals) > eps(T)^(3/2)*sum(abs2, r) || throw(ArgumentError(
        "the responses are fitted exactly, so the error variance and the coefficient covariance are not estimable"))
    b = pls.coef
    Vp, Ve = pls.vcov_bayesian, pls.vcov_frequentist
    inverse = Vp ./ pls.sigma2
    common = _fos_common(y, observations)
    # Residual curves at the observation points (pseudo-observations for basis responses).
    residual_curves = Vector{T}[]
    row = 0
    for v in last.(observations)
        push!(residual_curves, pls.residuals[row+1:row+length(v)])
        row += length(v)
    end
    sigma_hat = if common === nothing
        zeros(T, 0, 0)
    else
        E = permutedims(reduce(hcat, residual_curves))
        E = E .- sum(E; dims=1) ./ n
        S = transpose(E)*E ./ n
        S = (S + transpose(S))/2
        if y isa BasisFunctionalData
            # Coefficient residuals: c_i - B x_i = G^(-1/2) times the pseudo-residuals.
            rootinv = inv(first(common))
            S = rootinv*S*rootinv
            S = (S + transpose(S))/2
        end
        S
    end
    V = if covariance === :bayesian
        Vp
    elseif covariance === :frequentist
        Ve
    elseif covariance === :cluster
        meat = zeros(T, K, K)
        row = 0
        for v in last.(observations)
            rows = row+1:row+length(v)
            u = transpose(view(D, rows, :))*pls.residuals[rows]
            meat .+= u .* transpose(u)
            row += length(v)
        end
        n/(n - 1) .* (inverse*meat*inverse) .+ (Vp .- Ve)
    else
        common === nothing && throw(ArgumentError(
            "covariance=:pooled needs a complete common grid or a basis response"))
        Sigma = y isa BasisFunctionalData ? first(common)*sigma_hat*first(common) : sigma_hat
        # sum_i D_i' Sigma D_i, block (j, l) = (X'X)[j, l] Phi_j' Sigma Phi_l.
        XtX = transpose(X)*X
        meat = zeros(T, K, K)
        for j in 1:q, l in 1:q
            meat[ranges[j], ranges[l]] .= XtX[j, l] .* (transpose(common[j])*Sigma*common[l])
        end
        inverse*meat*inverse
    end
    V = Matrix((V + transpose(V))/2)
    all(isfinite, b) && all(isfinite, V) || throw(ArgumentError(
        "function-on-scalar coefficients and covariance must be finite; rescale the inputs"))
    fitted_values, residual_values = _fos_fitted(y, bases, ranges, b, X, D, residual_curves, counts)
    edf = T[sum(pls.edf[ranges[j]]) for j in 1:q]
    lambdas = shared ? fill(only(pls.lambda), q) : copy(pls.lambda)
    solver = pls.settings
    settings = (; reference="refund 0.1.40 pffr (scalar terms) and fosr",
        estimator=:penalized_least_squares, design_columns=copy(coding.labels),
        coefficient_basis=basis_source, intercept,
        lambda_selection=lambda === nothing ? criterion : :fixed,
        lambda_structure=shared ? :shared : :separate, lambda_status=copy(pls.lambda_status),
        criterion, criterion_value=pls.criterion_value, sigma2=pls.sigma2, edf_total=pls.edf_total,
        penalty_source=penalty_order === nothing ? :default_order : :supplied,
        loss=y isa BasisFunctionalData ? :L2_exact_gram : :observed_points,
        observation_weights=:unit, nobservations=N, covariance,
        residual_covariance=common === nothing ? :not_estimated : :pooled_residuals,
        interval_distribution=:normal,
        solver=(; initial_lambda=solver.initial_lambda, iterations=solver.iterations,
            convergence=solver.convergence, tol=solver.tol, maxiter=solver.maxiter,
            rank_tol=solver.rank_tol))
    FunctionOnScalarRegression(bases, b, V, X, deepcopy(y), fitted_values, residual_values,
        sigma_hat, lambdas, edf, Vector{Any}(orders), intercept, coding, settings)
end

function _fos_fitted(y::GridFunctionalData{S}, bases, ranges, b::Vector{T}, X, D, residual_curves, counts) where {S,T}
    fitted = GridFunctionalData(_fos_curves(bases, ranges, b, X, y.grid), y.grid; ids=y.ids)
    residual = Matrix{Union{Missing,T}}(missing, nobs(y), length(y.grid))
    for i in 1:nobs(y)
        columns = y.observed === nothing ? axes(residual, 2) : findall(view(y.observed, i, :))
        residual[i, columns] .= residual_curves[i]
    end
    fitted, GridFunctionalData(residual, y.grid; ids=y.ids)
end

function _fos_fitted(y::IrregularFunctionalData, bases, ranges, b::Vector{T}, X, D, residual_curves, counts) where {T}
    IrregularFunctionalData(copy(y.times), D*b, copy(y.offsets); domain=y.domain, ids=y.ids),
        IrregularFunctionalData(copy(y.times), reduce(vcat, residual_curves), copy(y.offsets); domain=y.domain, ids=y.ids)
end

function _fos_fitted(y::BasisFunctionalData, bases, ranges, b::Vector{T}, X, D, residual_curves, counts) where {T}
    fitted_coefs = X*transpose(reduce(hcat, [b[range] for range in ranges]))
    BasisFunctionalData(fitted_coefs, y.basis; ids=y.ids),
        BasisFunctionalData(Matrix{T}(y.coefs) - fitted_coefs, y.basis; ids=y.ids)
end

# ---------------------------------------------------------------------------
# Accessors.

_fos_q(f) = size(f.design, 2)
_fos_penalized(f::FunctionOnScalarRegression) = f.settings.estimator === :penalized_least_squares

"""Return the stacked basis coefficients `[b_1; ...; b_q]`, intercept first (block `k` has `nbasis(f.bases[k])` entries)."""
coef(f::FunctionOnScalarRegression) = copy(f.coefficients)
"""
Return the covariance of `coef(f)`: the `fRegress.stderr` covariance with exact
inner products for the fixed-lambda basis fit, otherwise the covariance chosen
by `covariance` (`f.settings.covariance`).
"""
vcov(f::FunctionOnScalarRegression) = copy(f.covariance)
"""Return standard errors of `coef(f)`."""
stderror(f::FunctionOnScalarRegression) = sqrt.(max.(diag(f.covariance), zero(eltype(f.covariance))))
"""Return the number of response curves."""
nobs(f::FunctionOnScalarRegression) = size(f.design, 1)
"""
Return the number of basis coefficients estimated (fixed-lambda basis fit) or
the total effective degrees of freedom of the coefficient functions (penalized fit).
"""
dof(f::FunctionOnScalarRegression) = _fos_penalized(f) ? f.settings.edf_total : length(f.coefficients)
"""
Return `nobs(f) - q`, the denominator of the estimated residual coefficient
covariance (fixed-lambda basis fit), or the number of observations minus the
total effective degrees of freedom (penalized fit).
"""
dof_residual(f::FunctionOnScalarRegression) = _fos_penalized(f) ?
    f.settings.nobservations - f.settings.edf_total : nobs(f) - _fos_q(f)
"""
Return fitted response curves: `BasisFunctionalData` for basis responses, the
full response grid for grid responses, and the observation times for irregular
responses.
"""
fitted(f::FunctionOnScalarRegression) = deepcopy(f.fitted_values)
"""
Return response-minus-fitted curves at the observed points, with the response's
data type (missing grid values stay missing).
"""
residuals(f::FunctionOnScalarRegression) = deepcopy(f.residual_values)

"""
    coefficient_functions(f::FunctionOnScalarRegression)

Return a tuple of one-curve `BasisFunctionalData`, one per coefficient
(intercept first when fitted), each in its coefficient basis.
"""
function coefficient_functions(f::FunctionOnScalarRegression)
    Tuple(BasisFunctionalData(reshape(f.coefficients[range], 1, :), basis)
        for (basis, range) in zip(f.bases, _fos_ranges(f.bases)))
end

function _fos_grid(f, grid)
    a, b = domain(first(f.bases))
    g = if grid !== nothing
        grid
    elseif _fos_penalized(f) && f.response isa GridFunctionalData
        f.response.grid
    else
        collect(range(a, b; length=max(51, 10*maximum(nbasis, f.bases) + 1)))
    end
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
    normal = f.settings.interval_distribution === :normal
    (; estimate=copy(estimate), se, lower, upper, alpha=a, critical_value=critical,
        settings=(; target, distribution=f.settings.interval_distribution,
            dof=normal ? nothing : dof_residual(f), covariance=f.settings.covariance,
            residual_covariance=f.settings.residual_covariance, pointwise=true))
end

"""
    coefficient_bands(f::FunctionOnScalarRegression; alpha=0.05, grid=nothing)

Pointwise intervals `beta_k(t) ± critical*se(t)` for every coefficient (intercept
first), with `se(t)^2 = phi_k(t)' Var(b_k) phi_k(t)` from `vcov(f)`.
`grid=nothing` uses the response grid for a penalized fit to a grid response
(fosr's `argvals`), and otherwise `fRegress.stderr`'s plotting grid of
`max(51, 10*nbasis+1)` equally spaced points (the largest `nbasis` when the
coefficients have different bases).
"""
function coefficient_bands(f::FunctionOnScalarRegression; alpha=0.05, grid=nothing)
    g = _fos_grid(f, grid)
    T = eltype(g)
    Tuple(map(zip(f.bases, _fos_ranges(f.bases))) do (basis, index)
        Phi = Matrix{T}(basis_matrix(basis, g))
        estimate = Phi*f.coefficients[index]
        variance = vec(sum((Phi*f.covariance[index, index]).*Phi; dims=2))
        merge((; grid=copy(g)), _fos_interval(f, estimate, variance, alpha, :coefficient))
    end)
end

# Variance of a new response curve around its mean, on the basis evaluation Phi.
_fos_noise_variance(f, Phi) = _fos_penalized(f) ? fill(f.settings.sigma2, size(Phi, 1)) :
    vec(sum((Phi*f.residual_covariance).*Phi; dims=2))

"""
    predict(f::FunctionOnScalarRegression, z; interval=:none, alpha=0.05, grid=nothing)

Predict response curves for new scalar covariates (a vector or matrix, or a
table with the fitted columns and factor levels; the intercept is added as in
the fit). With `interval=:none`, coefficients in one common basis give
`BasisFunctionalData` in that basis (`grid` must be `nothing`); coefficients in
different bases (for example `basis=:pffr`) give `GridFunctionalData` on
`grid`, chosen as in [`coefficient_bands`](@ref) when `nothing`.
`:confidence` gives pointwise intervals for the mean curve from `vcov(f)` on a
grid; `:prediction` adds the variance of a new response curve:
`phi(t)' Σ_c phi(t)` for the fixed-lambda basis fit and the error variance
`f.settings.sigma2` for the penalized fit. Both return `grid`, `estimate`,
`se`, `lower`, `upper`, `alpha`, `critical_value` and `settings` with
subject × grid arrays.
"""
function predict(f::FunctionOnScalarRegression, z; interval=:none, alpha=0.05, grid=nothing)
    interval in (:none, :confidence, :prediction) || throw(ArgumentError(
        "interval must be :none, :confidence, or :prediction"))
    T = eltype(f.coefficients)
    X = _fos_new_design(f, z)
    ranges = _fos_ranges(f.bases)
    common = _fos_common_basis(f.bases)
    if interval === :none && common
        grid === nothing || throw(ArgumentError(
            "grid is only used with intervals or with coefficients in different bases"))
        coefficients = X*transpose(reduce(hcat, [f.coefficients[range] for range in ranges]))
        return BasisFunctionalData(coefficients, first(f.bases))
    end
    g = _fos_grid(f, grid)
    Phis = [Matrix{T}(basis_matrix(basis, g)) for basis in f.bases]
    estimate = zeros(T, size(X, 1), length(g))
    for (j, range) in enumerate(ranges)
        estimate .+= view(X, :, j) .* transpose(Phis[j]*f.coefficients[range])
    end
    interval === :none && return GridFunctionalData(estimate, g)
    variance = similar(estimate)
    for i in axes(X, 1)
        A = reduce(hcat, [X[i, j] .* Phis[j] for j in eachindex(Phis)])
        variance[i, :] = vec(sum((A*f.covariance).*A; dims=2))
    end
    interval === :prediction && (variance .+= transpose(_fos_noise_variance(f, first(Phis))))
    merge((; grid=g), _fos_interval(f, estimate, variance, alpha, interval))
end

function Base.show(io::IO, f::FunctionOnScalarRegression)
    print(io, "FunctionOnScalarRegression(n=", nobs(f), ", coefficients=", _fos_q(f), ", ")
    _fos_common_basis(f.bases) ? print(io, "basis=", first(f.bases)) : print(io, "bases=", f.bases)
    print(io, ", lambda=", f.lambda, ", edf=", f.edf, ", penalty_order=",
        f.penalty_order, ", settings=", f.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::FunctionOnScalarRegression) = show(io, f)
