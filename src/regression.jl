const _RegressionCurve = Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData}
const _RegressionScalars = Union{AbstractVector{<:Union{Missing,Real}},AbstractMatrix{<:Union{Missing,Real}}}

"""
    FPCRegression
    fit(FPCRegression, x, y; scalar_covariates=nothing, fpca=nothing,
        fpca_options=(;), ncomponents=nothing, selection=:fpca,
        response_fpca=nothing, response_fpca_options=(;), response_ncomponents=nothing)

Gaussian regression on leading FPCA scores. `x` is one univariate functional
dataset, a tuple/vector of them, or `MultivariateFunctionalData`. `y` is a
finite response vector, or one univariate functional dataset (grid, irregular
or basis) for function-on-function regression. Scalar covariates are an
optional finite vector or matrix, without an intercept column.

For a scalar response the model is `y = intercept + Z*gamma + sum(scores_j*b_j) + error`;
`coefficient_functions` returns each `beta_j = eigenfunctions_j*b_j`.
`coef` and `vcov` order parameters as intercept, scalar coefficients, then
score coefficients in covariate order. `dof` is their count and
`dof_residual = nobs-dof`. `sigma2` is residual sum of squares / residual dof.
Rank-deficient designs, saturated models and constant responses throw.

For a functional response, `y` is reduced to its own leading FPCA scores
`eta_ik` and each response score is regressed on the same design, as fdapace
0.6.0 `FLM` (`GetBR2New`) does. Then `beta_j(t,s) = phi_j(t)' B_j psi(s)'`, with
covariate eigenfunctions `phi_j` and response eigenfunctions `psi`. `coef`
returns the `p × K_Y` matrix `B` (rows in the scalar-response order, one
column per response component), `vcov` the covariance of `vec(B)`,
`SigmaE ⊗ inv(D'D)`, where `SigmaE` is the residual score cross-product
divided by `dof_residual = nobs - p`. `stderror` has the shape of `coef`;
`dof` counts all `p*K_Y` coefficients. `sigma2` is `tr(SigmaE)`, the integrated
residual variance within the retained response components, and `r2` is
fdapace's score-scale R², `1 - sum(RSS_k)/sum(TSS_k)`. The response FPCA
follows the covariate rules: `response_fpca` supplies a fit (copied; its
stored scores are reused only for its own training data),
`response_fpca_options` tunes a fresh one (fdapace's `optnsListY=list()`
default is the FPCA defaults, including FVE(0.99)), and `response_ncomponents`
fixes or truncates its count. Regression selection (`AIC`, `BIC`, `:loocv`)
is not defined for functional responses and throws; fdapace FLM uses only
FPCA selection.

By default each covariate is fitted with its own `fit(FPCA, x_j)` defaults,
including FVE(0.99), following fdapace 0.6.0 `FLM`. `fpca` can supply a fitted
FPCA or a tuple/vector with one fit (or `nothing`) per covariate.
`fpca_options` is a named tuple shared by newly fitted covariates, or a
corresponding tuple/vector of named tuples. Options for supplied fits throw.
`ncomponents` is `nothing` (keep each FPCA's selection), a common positive
integer or FPCA selector, or a tuple/vector of these. Supplied fits can be
truncated to a fixed count or FVE within their stored harmonics. They are copied;
stored training scores are used only when the covariate equals their copied
training data; otherwise existing FPCA prediction scores are used.

`selection=AIC()` or `BIC()` minimizes the Gaussian regression criterion over
all positive leading-component counts, conditional on the fitted FPC bases.
`selection=:loocv` minimizes held-out squared response error, refitting FPCA
in every fold, as fda.usc 2.2.0 `fregre.pc.cv` does. It omits that source's
extra residual scaling by `n-ck` to target ordinary prediction MSE.
Its cost is `n` FPCA fits per covariate for every candidate count tuple, so it
grows with `n` times the product of the counts searched; cap them with `max_k`.
Fold fits are not reused across counts because sparse (PACE) scores depend on
the number of retained components. Supplied fits are unsupported
for this selector: an external fitting procedure cannot be refitted faithfully.
Regression selection cannot be combined with `ncomponents` or an FPCA option
named `ncomponents`; use `max_k` in `fpca_options` to cap fresh fits. All candidates,
criteria and failures are stored. Ties choose the first lexicographic count.
Nonestimable candidates have infinite criterion and a recorded reason.

The joint leading-count search extends fda.usc's one-predictor selection,
using leading counts instead of its stepwise component subsets;
fdapace FLM itself uses only FPCA selection. No smoothing penalty is fitted.
OLS uses Julia's pivoted QR, with numerical rank tolerance
`max(size(design)...)*eps(T)*maximum(abs, diag(R))`, stored in `settings`.
All inputs and tuning choices are copied. See the Regression documentation
for reference defaults, scale differences and conditional uncertainty.
"""
struct FPCRegression{T<:AbstractFloat,B,S,O,A<:AbstractVecOrMat{T}}
    # FPCA route/settings do not change the regression result's concrete type.
    # This mirrors FSVD's abstract FPCA fields and keeps automatic fits inferred.
    fpca::Tuple{Vararg{FPCA}}
    response_fpca::Union{Nothing,FPCA}
    beta::B
    coefficients::A
    covariance::Matrix{T}
    design::Matrix{T}
    response::A
    fitted_values::A
    residual_values::A
    residual_covariance::Matrix{T}
    sigma2::T
    r_squared::T
    counts::Vector{Int}
    nscalar::Int
    selection::S
    settings::O
end

# Scalar responses store vectors; functional responses store score matrices.
const _ScalarFPCRegression = FPCRegression{<:AbstractFloat,<:Any,<:Any,<:Any,<:AbstractVector}
const _FunctionalFPCRegression = FPCRegression{<:AbstractFloat,<:Any,<:Any,<:Any,<:AbstractMatrix}

_regression_curves(x::_RegressionCurve) = (x,)
_regression_curves(x::MultivariateFunctionalData) = Tuple(x.components)
_regression_curves(x::Union{Tuple,AbstractVector}) = Tuple(x)
_regression_curves(x) = throw(ArgumentError("functional covariates must be functional data or a tuple/vector of univariate functional data"))

Base.@constprop :aggressive function _regression_inputs(x)
    curves = _regression_curves(x)
    !isempty(curves) && all(c -> c isa _RegressionCurve, curves) ||
        throw(ArgumentError("functional covariates must contain at least one univariate functional dataset"))
    n = nobs(first(curves))
    all(c -> nobs(c) == n, curves) || throw(DimensionMismatch("functional covariates must have the same subject count"))
    _validate_component_ids(curves)
    curves
end

_regression_each(x::Union{Tuple,AbstractVector}, p, name) =
    length(x) == p ? Tuple(x) : throw(DimensionMismatch("$name must have one entry per functional covariate"))
Base.@constprop :aggressive _regression_each(x, p, name) = ntuple(_ -> x, Val(p))

function _regression_scalars(z, n, ::Type{T}) where {T}
    z === nothing && return zeros(T, n, 0)
    z isa _RegressionScalars ||
        throw(ArgumentError("scalar_covariates must be a finite real vector or matrix"))
    Base.require_one_based_indexing(z)
    size(z, 1) == n || throw(DimensionMismatch("scalar_covariates must have one row per subject"))
    all(v -> v isa Real && isfinite(v), z) || throw(ArgumentError("scalar_covariates must be finite and contain no missing values"))
    out = z isa AbstractVector ? reshape(Vector{T}(z), :, 1) : Matrix{T}(z)
    all(isfinite, out) || throw(ArgumentError("scalar_covariates must remain finite after conversion"))
    out
end

function _regression_same_data(x::GridFunctionalData, y::GridFunctionalData)
    x.grid == y.grid && x.values == y.values && x.observed == y.observed
end
_regression_same_data(x::IrregularFunctionalData, y::IrregularFunctionalData) =
    x.times == y.times && x.values == y.values && x.offsets == y.offsets
_regression_same_data(x::BasisFunctionalData, y::BasisFunctionalData) =
    x.basis == y.basis && x.coefs == y.coefs
_regression_same_data(x, y) = false

# FPCA route dispatch can obscure prediction's return type. Normalize scores
# at this boundary using the covariate and FPCA element types, preserving both.
function _regression_fit_scores(c::AbstractFunctionalData{S}, f::FPCA{T})::Matrix{promote_type(S,T)} where {S,T}
    projected = _regression_same_data(c,f.training_data) ? scores(f) :
        project_scores(f,c)
    Matrix{promote_type(S,T)}(projected)
end

Base.@constprop :aggressive function _regression_fpca(x, supplied, options, rule, search;
        name="fpca")
    options isa NamedTuple || throw(ArgumentError("$(name)_options entries must be named tuples"))
    for key in (:ncomponents,:max_k,:maxK)
        haskey(options,key) && options[key] isa Bool && throw(ArgumentError(
            "$(name)_options.$key cannot be Boolean"))
    end
    if supplied === nothing
        search && haskey(options, :ncomponents) && throw(ArgumentError(
            "regression selection uses all candidate components; cap them with fpca_options.max_k"))
        rule !== nothing && haskey(options, :ncomponents) && throw(ArgumentError(
            "ncomponents is specified both directly and in $(name)_options"))
        effective = search ? merge(options, (;ncomponents=FVE(1))) :
            rule === nothing ? options : merge(options, (;ncomponents=rule))
        return fit(FPCA, x; effective...)
    end
    supplied isa FPCA || throw(ArgumentError("$name entries must be FPCA fits or nothing"))
    isempty(options) || throw(ArgumentError("$(name)_options cannot tune an already supplied FPCA"))
    deepcopy(supplied)
end

function _regression_count(rule, f)
    rule === nothing && return ncomponents(f)
    rule isa Integer && !(rule isa Bool) && 1 <= rule <= ncomponents(f) && return Int(rule)
    if rule isa FVE
        k = findfirst(>=(rule.threshold), f.fve)
        k !== nothing && k <= ncomponents(f) && return k
        throw(ArgumentError("ncomponents FVE threshold requires harmonics absent from the supplied FPCA"))
    end
    throw(ArgumentError("ncomponents for a supplied FPCA must be a positive available count, FVE, or nothing"))
end

function _regression_design(z, projected, counts, ::Type{T}) where {T}
    n = size(z, 1)
    design = hcat(ones(T, n), z, (Matrix{T}(s[:, 1:k]) for (s,k) in zip(projected,counts))...)
    all(isfinite, design) || throw(ArgumentError("regression design must be finite"))
    design
end

_regression_residual_covariance(r::AbstractVector, df) = fill(sum(abs2, r)/df, 1, 1)
function _regression_residual_covariance(r::AbstractMatrix, df)
    c = transpose(r)*r/df
    (c + transpose(c))/2
end

# Ordinary least squares for one response vector or several response columns
# sharing a design. The coefficient covariance is SigmaE ⊗ inv(D'D).
function _regression_ols(design::Matrix{T}, y::AbstractVecOrMat{T}) where {T}
    n, p = size(design)
    n > p || throw(ArgumentError("regression requires more subjects than coefficients for positive residual degrees of freedom"))
    qrfit = qr(design, ColumnNorm())
    R = Matrix(qrfit.R)
    tolerance = max(n,p) * eps(T) * maximum(abs, diag(R))
    all(d -> abs(d) > tolerance, diag(R)) || throw(ArgumentError(
        "regression design is rank deficient; remove redundant scalar covariates or FPC scores"))
    b = qrfit \ y
    fitted_values = design*b
    residual_values = y-fitted_values
    rss = sum(abs2, residual_values)
    residual_covariance = _regression_residual_covariance(residual_values, n-p)
    sigma2 = tr(residual_covariance)
    inverseR = UpperTriangular(R) \ Matrix{T}(I,p,p)
    unscaledpivot = inverseR * transpose(inverseR)
    unscaled = similar(unscaledpivot)
    unscaled[qrfit.p,qrfit.p] = unscaledpivot
    covariance = kron(residual_covariance, unscaled)
    all(isfinite, b) && all(isfinite, covariance) && isfinite(sigma2) ||
        throw(ArgumentError("regression coefficients and covariance must be finite; rescale the inputs"))
    (; b, fitted_values, residual_values, rss, sigma2, covariance, unscaled,
        residual_covariance, tolerance)
end

# Exact leave-one-subject-out prediction: every fold estimates its own FPCA.
function _regression_cv(curves, y, z, options, counts)
    n = length(y)
    error = zero(eltype(y))
    for i in 1:n
        train = [j for j in 1:n if j != i]
        models = map(curves, options, counts) do x, opt, k
            fit(FPCA, x[train]; merge(opt,(;ncomponents=k))...)
        end
        projected = map(scores, models)
        design = _regression_design(z[train,:], projected, counts, eltype(y))
        ols = _regression_ols(design, y[train])
        held = map((f,x) -> project_scores(f,x[[i]]), models, curves)
        row = _regression_design(z[[i],:], held, counts, eltype(y))
        error += (y[i] - only(row*ols.b))^2
    end
    error/n
end

function _regression_arguments(x, fpca, fpca_options, ncomponents, selection)
    curves = _regression_inputs(x)
    p = length(curves)
    supplied = _regression_each(fpca,p,"fpca")
    options = _regression_each(fpca_options,p,"fpca_options")
    rules = _regression_each(ncomponents,p,"ncomponents")
    any(r -> r isa Bool, rules) && throw(ArgumentError("ncomponents cannot be Boolean"))
    (selection === :fpca || selection isa Union{AIC,BIC} || selection === :loocv) ||
        throw(ArgumentError("selection must be :fpca, AIC(), BIC(), or :loocv"))
    curves, supplied, options, rules
end

Base.@constprop :aggressive function fit(::Type{FPCRegression}, x, y::AbstractVector{<:Real};
        scalar_covariates=nothing, fpca=nothing, fpca_options=(;),
        ncomponents=nothing, selection=:fpca)
    curves, supplied, options, rules = _regression_arguments(x, fpca, fpca_options, ncomponents, selection)
    n = nobs(first(curves))
    Base.require_one_based_indexing(y)
    length(y) == n || throw(DimensionMismatch("response length must equal the functional covariate subject count"))
    all(isfinite, y) || throw(ArgumentError("response must be finite"))
    search = selection !== :fpca
    search && any(!isnothing, rules) && throw(ArgumentError(
        "ncomponents cannot be combined with regression selection; cap fresh FPCA fits using max_k"))
    selection === :loocv && any(!isnothing, supplied) && throw(ArgumentError(
        "selection=:loocv requires fresh FPCA fits so each fitting procedure can be refitted in every fold"))
    models = map((c,f,o,r) -> _regression_fpca(c,f,o,r,search), curves,supplied,options,rules)
    projected = map(_regression_fit_scores,curves,models)
    scalar_covariates === nothing || scalar_covariates isa _RegressionScalars ||
        throw(ArgumentError("scalar_covariates must be a finite real vector or matrix"))
    scalar_type = scalar_covariates === nothing ? eltype(y) : Base.nonmissingtype(eltype(scalar_covariates))
    T = _floattype(promote_type(eltype(y),scalar_type,map(eltype,projected)...))
    response = Vector{T}(y)
    all(isfinite,response) || throw(ArgumentError("response must remain finite after conversion"))
    _fit_regression_scores(curves,response,scalar_covariates,models,projected,
        supplied,options,rules,selection)
end

# Bind the regression precision independently of keyword-dependent FPCA routes.
function _fit_regression_scores(curves,response::Vector{T},scalar_covariates,models,
        projected,supplied,options,rules,selection) where {T}
    n, p = length(response), length(models)
    search = selection !== :fpca
    total = sum(abs2,response .- sum(response)/n)
    isfinite(total) && total > 0 || throw(ArgumentError("response must have finite positive variation for R squared"))
    z = _regression_scalars(scalar_covariates,n,T)
    initial = [supplied[j] === nothing ? FunctionalDataAnalysis.ncomponents(models[j]) :
        _regression_count(rules[j],models[j]) for j in 1:p]
    candidates = Vector{Int}[]
    if search
        for c in Iterators.product((1:k for k in initial)...)
            push!(candidates,Int[c...])
        end
    else
        push!(candidates,initial)
    end
    # Make iteration order explicitly lexicographic, independent of product axes.
    sort!(candidates; by=c -> Tuple(c))
    values = fill(T(Inf),length(candidates))
    reasons = fill("",length(candidates))
    for (i,counts) in enumerate(candidates)
        try
            design = _regression_design(z,projected,counts,T)
            candidate_ols = _regression_ols(design,response)
            values[i] = selection === :loocv ? _regression_cv(curves,response,z,options,Tuple(counts)) :
                selection isa AIC ? n*log(candidate_ols.rss/n)+2size(design,2) :
                selection isa BIC ? n*log(candidate_ols.rss/n)+log(T(n))*size(design,2) : zero(T)
            (isnan(values[i]) || values[i] == Inf) && throw(ArgumentError(
                "regression selection criterion overflowed; rescale the inputs"))
        catch err
            err isa Union{ArgumentError,DimensionMismatch} || rethrow()
            search || rethrow()
            values[i] = T(Inf)
            reasons[i] = sprint(showerror,err)
        end
    end
    valid = findall(i -> isempty(reasons[i]),eachindex(values))
    isempty(valid) && throw(ArgumentError("no estimable regression component candidate: $(join(unique(reasons), "; "))"))
    chosen = valid[argmin(values[valid])]
    counts = copy(candidates[chosen])
    design = _regression_design(z,projected,counts,T)
    ols = _regression_ols(design,response)
    offset = 1+size(z,2)
    starts = cumsum(vcat(offset,counts))
    beta = map(models,ntuple(identity,Val(length(models)))) do f,j
        b = zeros(T,1,FunctionalDataAnalysis.ncomponents(f))
        b[1,1:counts[j]] = ols.b[starts[j]+1:starts[j+1]]
        BasisFunctionalData(b,deepcopy(eigenfunctions(f)))
    end
    criterion = selection === :fpca ? :fpca : selection isa AIC ? :AIC : selection isa BIC ? :BIC : :loocv
    tuning = (; rule=criterion, candidates=deepcopy(candidates), values=copy(values),
        failures=copy(reasons), chosen, refit_fpca=selection === :loocv,
        tie_rule=:lexicographic_first)
    settings = (; fpca_source=map(f -> f === nothing ? :fitted : :supplied,supplied),
        fpca_options=deepcopy(options), ncomponents=deepcopy(rules),
        rank_tolerance=T(ols.tolerance), intercept=true,
        covariance=:conditional_OLS, residual_variance=:RSS_over_residual_dof,
        coefficient_order=:intercept_scalars_functional_scores,
        reference="fdapace 0.6.0 FLM; fda.usc 2.2.0 fregre.pc / fregre.pc.cv")
    FPCRegression(models,nothing,beta,ols.b,ols.covariance,design,response,ols.fitted_values,
        ols.residual_values,ols.residual_covariance,ols.sigma2,one(T)-ols.rss/total,counts,
        size(z,2),tuning,settings)
end
function fit(::Type{FPCRegression},x,y::AbstractVector;kwargs...)
    Base.require_one_based_indexing(y)
    missing_index = findfirst(ismissing,y)
    missing_index === nothing || throw(ArgumentError("response contains missing at index $missing_index"))
    T = Base.nonmissingtype(eltype(y))
    T <: Real || throw(ArgumentError("response must be a finite real vector"))
    fit(FPCRegression,x,Vector{T}(y);kwargs...)
end

Base.@constprop :aggressive function fit(::Type{FPCRegression}, x, y::_RegressionCurve;
        scalar_covariates=nothing, fpca=nothing, fpca_options=(;),
        ncomponents=nothing, selection=:fpca, response_fpca=nothing,
        response_fpca_options=(;), response_ncomponents=nothing)
    curves, supplied, options, rules = _regression_arguments(x, fpca, fpca_options, ncomponents, selection)
    selection === :fpca || throw(ArgumentError(
        "functional responses support only selection=:fpca; fdapace FLM selects components by FPCA"))
    nobs(y) == nobs(first(curves)) || throw(DimensionMismatch(
        "functional response must have the functional covariate subject count"))
    _validate_component_ids((curves..., y))
    response_ncomponents isa Bool && throw(ArgumentError("response_ncomponents cannot be Boolean"))
    models = map((c,f,o,r) -> _regression_fpca(c,f,o,r,false), curves,supplied,options,rules)
    projected = map(_regression_fit_scores,curves,models)
    response_model = _regression_fpca(y, response_fpca, response_fpca_options,
        response_ncomponents, false; name="response_fpca")
    response_count = response_fpca === nothing ? FunctionalDataAnalysis.ncomponents(response_model) :
        _regression_count(response_ncomponents, response_model)
    response_scores = _regression_fit_scores(y, response_model)
    scalar_covariates === nothing || scalar_covariates isa _RegressionScalars ||
        throw(ArgumentError("scalar_covariates must be a finite real vector or matrix"))
    scalar_type = scalar_covariates === nothing ? eltype(response_scores) :
        Base.nonmissingtype(eltype(scalar_covariates))
    T = _floattype(promote_type(eltype(response_scores),scalar_type,map(eltype,projected)...))
    _fit_regression_functional(Matrix{T}(response_scores[:,1:response_count]),
        scalar_covariates,models,projected,supplied,options,rules,response_model,
        response_fpca === nothing ? :fitted : :supplied,response_fpca_options,
        response_ncomponents)
end
fit(::Type{FPCRegression},x,y;kwargs...) = throw(ArgumentError(
    "response must be a finite real vector or univariate functional data"))

# The response FPCA mean and its leading eigenfunctions on a grid.
function _regression_response_basis(f::FPCA, k, grid, ::Type{T}) where {T}
    g = Vector{T}(grid)
    _validate_strict_grid(g)
    a, b = domain(eigenfunctions(f))
    all(t -> a <= t <= b, g) || throw(ArgumentError(
        "response grid must lie inside the response FPCA domain [$a, $b]"))
    psi = Matrix{T}(basis_matrix(eigenfunctions(f), g)[:,1:k])
    mu = Vector{T}(vec(_fpca_mean_values(f.mean, g)))
    g, psi, mu
end

function _regression_covariate_basis(f::FPCA, k, grid, ::Type{T}) where {T}
    g = Vector{T}(grid)
    quadrature_weights(g)
    Matrix{T}(basis_matrix(eigenfunctions(f),g)[:,1:k])
end

function _fit_regression_functional(response::Matrix{T},scalar_covariates,models,projected,
        supplied,options,rules,response_model,response_source,response_options,
        response_rule) where {T}
    n, p = size(response,1), length(models)
    z = _regression_scalars(scalar_covariates,n,T)
    counts = Int[supplied[j] === nothing ? FunctionalDataAnalysis.ncomponents(models[j]) :
        _regression_count(rules[j],models[j]) for j in 1:p]
    design = _regression_design(z,projected,counts,T)
    ols = _regression_ols(design,response)
    centered = response .- sum(response;dims=1) ./ n
    total = sum(abs2,centered)
    isfinite(total) && total > 0 || throw(ArgumentError(
        "response scores must have finite positive variation for R squared"))
    k = size(response,2)
    grid, psi, _ = _regression_response_basis(response_model,k,response_model.grid,T)
    starts = cumsum(vcat(1+size(z,2),counts))
    beta = map(models,ntuple(identity,Val(length(models)))) do f,j
        phi = _regression_covariate_basis(f,counts[j],f.grid,T)
        (; grid=Vector{T}(f.grid), response_grid=copy(grid),
            values=phi*ols.b[starts[j]+1:starts[j+1],:]*transpose(psi))
    end
    tuning = (; rule=:fpca, candidates=[copy(counts)], values=[zero(T)],
        failures=[""], chosen=1, refit_fpca=false, tie_rule=:lexicographic_first)
    settings = (; fpca_source=map(f -> f === nothing ? :fitted : :supplied,supplied),
        fpca_options=deepcopy(options), ncomponents=deepcopy(rules),
        response_fpca_source=response_source, response_fpca_options=deepcopy(response_options),
        response_ncomponents=deepcopy(response_rule), response_count=k,
        rank_tolerance=T(ols.tolerance), intercept=true,
        covariance=:conditional_multivariate_OLS,
        residual_variance=:score_crossproducts_over_residual_dof,
        r_squared=:fdapace_GetBR2New_score_scale,
        coefficient_order=:intercept_scalars_functional_scores_by_response_component,
        reference="fdapace 0.6.0 FLM with a functional response")
    FPCRegression(models,response_model,beta,ols.b,ols.covariance,design,response,
        ols.fitted_values,ols.residual_values,ols.residual_covariance,ols.sigma2,
        one(T)-sum(abs2,ols.residual_values)/total,counts,size(z,2),tuning,settings)
end

"""
    coefficient_functions(f::FPCRegression)
    coefficient_functions(f::PenalizedFunctionalRegression)

Return a tuple with one coefficient per functional covariate. For a scalar
response (FPC or penalized fits) these are copied `BasisFunctionalData`
coefficient functions in FPC or spline bases; evaluate
them using `evaluate(beta, t)`. For a functional response each entry is a
named tuple `(; grid, response_grid, values)` holding `beta_j(t,s)` on the
covariate FPCA work grid (rows, `t`) and response FPCA work grid (columns,
`s`), the orientation of fdapace's `betaList`.
"""
coefficient_functions(f::FPCRegression) = deepcopy(f.beta)
"""
Return copied coefficients ordered as intercept, scalars, then each covariate's
FPC scores: a vector for a scalar response, or a matrix with one column per
response FPC for a functional response.
"""
coef(f::FPCRegression) = copy(f.coefficients)
"""Return the conditional OLS covariance of `vec(coef(f))`."""
vcov(f::FPCRegression) = copy(f.covariance)
"""
    fitted(f::FPCRegression; grid=nothing)

Return copied fitted scalar responses, or for a functional response a
`GridFunctionalData` of fitted curves `mean_Y + fitted_scores*psi'` on the
response FPCA work grid (or a supplied grid inside its domain).
"""
fitted(f::FPCRegression; grid=nothing) = _regression_fitted(f, grid)
function _regression_fitted(f::_ScalarFPCRegression, grid)
    grid === nothing || throw(ArgumentError("grid requires a functional response"))
    copy(f.fitted_values)
end
_regression_fitted(f::_FunctionalFPCRegression, grid) =
    _regression_response_curves(f, f.fitted_values, grid; centered=false)
"""
    residuals(f::FPCRegression; grid=nothing)

Return response-minus-fitted residuals. For a functional response these are the
score residuals `(eta - fitted_scores)*psi'` on the response grid: residuals
of the response reconstructed from its retained components (truncation and
measurement error are not included).
"""
residuals(f::FPCRegression; grid=nothing) = _regression_residuals(f, grid)
function _regression_residuals(f::_ScalarFPCRegression, grid)
    grid === nothing || throw(ArgumentError("grid requires a functional response"))
    copy(f.residual_values)
end
_regression_residuals(f::_FunctionalFPCRegression, grid) =
    _regression_response_curves(f, f.residual_values, grid; centered=true)
"""Return the number of subjects used in an FPC regression."""
nobs(f::FPCRegression) = size(f.response, 1)
"""Return the number of fitted regression coefficients, including intercepts."""
dof(f::FPCRegression) = length(f.coefficients)
"""Return `nobs(f)` minus the number of design columns, the residual degrees of freedom."""
dof_residual(f::FPCRegression) = nobs(f)-size(f.design, 2)
"""Return the centered coefficient of determination (score scale for functional responses)."""
r2(f::FPCRegression) = f.r_squared
"""Return conditional OLS coefficient standard errors with the shape of `coef`."""
stderror(f::FPCRegression) = reshape(sqrt.(max.(diag(f.covariance),
    zero(eltype(f.covariance)))), size(f.coefficients))
"""Return a copied vector of selected FPC counts, one per functional covariate."""
ncomponents(f::FPCRegression) = copy(f.counts)

function _regression_response_curves(f, scores_matrix, grid; centered)
    T = eltype(f.covariance)
    model = f.response_fpca
    g, psi, mu = _regression_response_basis(model, size(f.coefficients,2),
        grid === nothing ? model.grid : grid, T)
    values = scores_matrix*transpose(psi)
    centered || (values .+= transpose(mu))
    GridFunctionalData(values, g)
end

"""
    pointwise_r2(f; grid=nothing)

Return `(; grid, values)` with a pointwise coefficient of determination.
For a functional-response [`FPCRegression`](@ref) this is
`1 - sum_i r_i(s)^2 / sum_i (Y_i(s) - mean(s))^2` for responses reconstructed
from their retained FPC scores, on the response FPCA work grid or a supplied
grid. fdapace 0.6.0 reports only the scalar score-scale R². Where every
retained response eigenfunction vanishes the ratio is `0/0` and the value is
`NaN`; this is reported, not replaced. For [`ConcurrentRegression`](@ref) it
returns fdapace `FCReg`'s stored `R2(t)` on the fitted grid.
"""
function pointwise_r2(f::_FunctionalFPCRegression; grid=nothing)
    T = eltype(f.covariance)
    model = f.response_fpca
    g, psi, _ = _regression_response_basis(model, size(f.coefficients,2),
        grid === nothing ? model.grid : grid, T)
    centered = (f.response .- sum(f.response;dims=1) ./ nobs(f))*transpose(psi)
    residual = f.residual_values*transpose(psi)
    (; grid=g, values=vec(one(T) .- sum(abs2,residual;dims=1) ./ sum(abs2,centered;dims=1)))
end

function _regression_tcritical(df::Int, alpha)
    # P(|T| > t) = I_x(df/2, 1/2) with x = df/(df+t^2); invert the incomplete beta.
    x, y = SpecialFunctions.beta_inc_inv(df/2, 0.5, alpha)
    sqrt(df*y/x)
end

function _regression_intervals(f,estimate,variance,alpha,target)
    a = _band_alpha(alpha)
    critical = _regression_tcritical(dof_residual(f),a)
    se = sqrt.(max.(variance,zero(eltype(variance))))
    width = critical .* se
    lower,upper = estimate-width,estimate+width
    all(isfinite,lower) && all(isfinite,upper) || throw(ArgumentError("regression interval endpoints must be finite"))
    (;estimate=copy(estimate),se,lower,upper,alpha=a,critical_value=critical,
        settings=(;target,distribution=:student_t,dof=dof_residual(f),
            covariance=:conditional_OLS,fpca_uncertainty=false,
            score_uncertainty=false,selection_uncertainty=false))
end

function _regression_band_grids(f,grid)
    grid === nothing ? map(m -> m.grid,f.fpca) :
        length(f.fpca)==1 && grid isa AbstractVector{<:Real} ? (grid,) :
        _regression_each(grid,length(f.fpca),"grid")
end

"""
    coefficient_bands(f::FPCRegression; alpha=0.05, grid=nothing,
        response_grid=nothing, method=:conditional, B=999, rng=nothing)

Pointwise intervals for each coefficient function. `method=:conditional`
(default) gives Student-t intervals conditional on the fitted FPCA and selected
score design, as in fda.usc 2.2.0 `fregre.pc`'s ordinary linear model. Return a
tuple of named tuples with `grid`, `estimate`, `se`, `lower`, `upper`, `alpha`,
`critical_value` and `settings`; functional responses add `response_grid` and
return surfaces (covariate grid in rows). `grid=nothing` uses each FPCA's work
grid; otherwise supply one grid for one predictor, or a tuple of grids.
`response_grid` likewise defaults to the response FPCA work grid. FPCA means,
harmonics, sparse scores, component selection and omitted components are
treated as fixed. For a functional response the variance of `beta_j(t,s)` is
`phi_j(t)'C_jj phi_j(t) * psi(s)'SigmaE psi(s)`.

`method=:bootstrap` gives fdapace 0.6.0 `FLMCI` percentile intervals: `B`
subject resamples (default 999, `FLMCI`'s `R`) each refit every FPCA (with the
stored options and component rules) and the regression, and R type-7
quantiles of the refitted coefficients give the interval. As in `FLMCI`'s
dense branch, a resample is redrawn until every functional variable has the
original observed time range, so refitted work grids agree; 1000 consecutive
rejections throw. Refitted coefficients are evaluated on the requested grids.
`se` is the bootstrap standard deviation and `critical_value` is `nothing`.
Bootstrap intervals require fresh FPCA fits and `selection=:fpca`; supplied
fits throw because their fitting procedure cannot be repeated. `rng=nothing`
uses `Random.default_rng()`; the RNG is copied (the caller's state is unchanged)
and its initial state is stored in `settings.rng_state`. `FLMCI` samples sparse
data from a subset of curves plus forced boundary curves; this package uses
plain subject resampling with the range check for every design. All intervals
are pointwise, not simultaneous bands.
"""
function coefficient_bands(f::FPCRegression;alpha=0.05,grid=nothing,response_grid=nothing,
        method=:conditional,B=nothing,rng=nothing)
    method in (:conditional,:bootstrap) || throw(ArgumentError("method must be :conditional or :bootstrap"))
    if method === :conditional
        B === nothing && rng === nothing || throw(ArgumentError(
            "B and rng are only available for method=:bootstrap"))
        return _regression_conditional_bands(f,alpha,grid,response_grid)
    end
    _regression_bootstrap_bands(f,alpha,grid,response_grid,B === nothing ? 999 : B,rng)
end

function _regression_conditional_bands(f::_ScalarFPCRegression,alpha,grid,response_grid)
    response_grid === nothing || throw(ArgumentError("response_grid requires a functional response"))
    grids = _regression_band_grids(f,grid)
    starts = cumsum(vcat(1+f.nscalar,f.counts))
    map(f.fpca,ntuple(identity,Val(length(f.fpca))),grids) do m,j,g
        g isa AbstractVector{<:Real} || throw(ArgumentError("coefficient interval grids must be real vectors"))
        quadrature_weights(g)
        functions = basis_matrix(eigenfunctions(m),g)[:,1:f.counts[j]]
        index = starts[j]+1:starts[j+1]
        estimate = vec(functions*f.coefficients[index])
        variance = vec(sum((functions*f.covariance[index,index]).*functions;dims=2))
        merge((;grid=copy(g)),_regression_intervals(f,estimate,variance,alpha,:coefficient))
    end
end

function _regression_conditional_bands(f::_FunctionalFPCRegression,alpha,grid,response_grid)
    T = eltype(f.covariance)
    grids = _regression_band_grids(f,grid)
    sg, psi, _ = _regression_response_basis(f.response_fpca,size(f.coefficients,2),
        response_grid === nothing ? f.response_fpca.grid : response_grid,T)
    response_variance = vec(sum((psi*f.residual_covariance).*psi;dims=2))
    unscaled = _regression_unscaled(f)
    starts = cumsum(vcat(1+f.nscalar,f.counts))
    map(f.fpca,ntuple(identity,Val(length(f.fpca))),grids) do m,j,g
        g isa AbstractVector{<:Real} || throw(ArgumentError("coefficient interval grids must be real vectors"))
        phi = _regression_covariate_basis(m,f.counts[j],g,T)
        index = starts[j]+1:starts[j+1]
        estimate = phi*f.coefficients[index,:]*transpose(psi)
        variance = vec(sum((phi*unscaled[index,index]).*phi;dims=2))*transpose(response_variance)
        merge((;grid=Vector{T}(g),response_grid=copy(sg)),
            _regression_intervals(f,estimate,variance,alpha,:coefficient_surface))
    end
end

# inv(D'D) is the leading block of SigmaE ⊗ inv(D'D) divided by SigmaE[1,1];
# recompute it from the design to avoid dividing by a possibly tiny variance.
function _regression_unscaled(f)
    p = size(f.design,2)
    qrfit = qr(f.design,ColumnNorm())
    R = Matrix(qrfit.R)
    inverseR = UpperTriangular(R) \ Matrix{eltype(R)}(I,p,p)
    unscaledpivot = inverseR*transpose(inverseR)
    unscaled = similar(unscaledpivot)
    unscaled[qrfit.p,qrfit.p] = unscaledpivot
    unscaled
end

# Observed time range of each subject, for FLMCI's resampling range check.
function _regression_subject_ranges(fd::Union{GridFunctionalData,IrregularFunctionalData})
    map(1:nobs(fd)) do i
        t, _ = fd[i]
        (first(t), last(t))
    end
end
_regression_subject_ranges(fd::BasisFunctionalData) = fill(domain(fd), nobs(fd))
function _regression_resampled_range(ranges, index)
    (minimum(first(ranges[i]) for i in index), maximum(last(ranges[i]) for i in index))
end

# Bootstrap resamples repeat subjects, so subject ids (which must be unique)
# are not carried into the replicate.
_regression_resample(fd::GridFunctionalData, index) =
    GridFunctionalData(_grid_matrix(fd)[index,:], fd.grid)
_regression_resample(fd::IrregularFunctionalData{T}, index) where {T} =
    IrregularFunctionalData(Vector{T}[fd[i][1] for i in index],
        Vector{T}[fd[i][2] for i in index]; domain=domain(fd))
_regression_resample(fd::BasisFunctionalData, index) =
    BasisFunctionalData(fd.coefs[index,:], fd.basis)
_regression_resample(y::AbstractVector, index) = y[index]

_regression_bootstrap_response(f::_ScalarFPCRegression) = f.response
_regression_bootstrap_response(f::_FunctionalFPCRegression) = f.response_fpca.training_data

function _regression_replicate_values(f::_ScalarFPCRegression, replicate, grids, response_grid)
    map(replicate.fpca, ntuple(identity,Val(length(f.fpca))), grids) do m,j,g
        vec(_regression_covariate_basis(m,replicate.counts[j],g,eltype(f.covariance)) *
            replicate.coefficients[_regression_block(replicate,j)])
    end
end
function _regression_replicate_values(f::_FunctionalFPCRegression, replicate, grids, response_grid)
    T = eltype(f.covariance)
    _, psi, _ = _regression_response_basis(replicate.response_fpca,
        size(replicate.coefficients,2), response_grid, T)
    map(replicate.fpca, ntuple(identity,Val(length(f.fpca))), grids) do m,j,g
        _regression_covariate_basis(m,replicate.counts[j],g,T) *
            replicate.coefficients[_regression_block(replicate,j),:]*transpose(psi)
    end
end
function _regression_block(f, j)
    starts = cumsum(vcat(1+f.nscalar,f.counts))
    starts[j]+1:starts[j+1]
end

function _regression_refit(f::_ScalarFPCRegression, curves, response, z, index)
    fit(FPCRegression, map(c -> _regression_resample(c,index), curves),
        _regression_resample(response,index);
        scalar_covariates=z === nothing ? nothing : z[index,:],
        fpca_options=f.settings.fpca_options, ncomponents=f.settings.ncomponents)
end
function _regression_refit(f::_FunctionalFPCRegression, curves, response, z, index)
    fit(FPCRegression, map(c -> _regression_resample(c,index), curves),
        _regression_resample(response,index);
        scalar_covariates=z === nothing ? nothing : z[index,:],
        fpca_options=f.settings.fpca_options, ncomponents=f.settings.ncomponents,
        response_fpca_options=f.settings.response_fpca_options,
        response_ncomponents=f.settings.response_ncomponents)
end

function _regression_bootstrap_bands(f::FPCRegression,alpha,grid,response_grid,B,rng)
    a = _band_alpha(alpha)
    _integer_in(B, 2) || throw(ArgumentError("B must be an integer of at least two"))
    f.selection.rule === :fpca || throw(ArgumentError(
        "bootstrap intervals require selection=:fpca, as fdapace FLMCI refits FLM"))
    all(==(:fitted), f.settings.fpca_source) &&
        get(f.settings, :response_fpca_source, :fitted) === :fitted || throw(ArgumentError(
        "bootstrap intervals require fresh FPCA fits; a supplied FPCA's procedure cannot be refitted"))
    f isa _ScalarFPCRegression && response_grid !== nothing &&
        throw(ArgumentError("response_grid requires a functional response"))
    generator = rng === nothing ? Random.default_rng() : rng
    generator isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG or nothing"))
    generator isa Random.RandomDevice && throw(ArgumentError(
        "rng must support reproducible state copying; RandomDevice is not supported"))
    initial = deepcopy(copy(generator))
    try
        _regression_bootstrap(f,a,grid,response_grid,Int(B),deepcopy(initial),initial)
    finally
        generator isa Random.TaskLocalRNG && copy!(generator, initial)
    end
end

function _regression_bootstrap(f,a,grid,response_grid,B,generator,initial)
    T = eltype(f.covariance)
    grids = map(g -> Vector{T}(g), _regression_band_grids(f,grid))
    rgrid = f isa _FunctionalFPCRegression ? Vector{T}(response_grid === nothing ?
        f.response_fpca.grid : response_grid) : nothing
    estimates = map(b -> b.estimate, _regression_conditional_bands(f,a,grids,rgrid))
    curves = map(m -> m.training_data, f.fpca)
    response = _regression_bootstrap_response(f)
    z = f.nscalar == 0 ? nothing : f.design[:,2:1+f.nscalar]
    variables = f isa _FunctionalFPCRegression ? (curves..., response) : curves
    ranges = map(_regression_subject_ranges, variables)
    originals = map(r -> _regression_resampled_range(r, eachindex(r)), ranges)
    n = nobs(f)
    draws = map(e -> Array{T}(undef, B, size(e)...), estimates)
    for b in 1:B
        index = Int[]
        rejected = 0
        while true
            index = rand(generator, 1:n, n)
            all(map((r,o) -> _regression_resampled_range(r,index) == o, ranges, originals)) && break
            rejected += 1
            rejected < 1000 || throw(ArgumentError(
                "bootstrap replicate $b: 1000 consecutive resamples lost part of the observed time range"))
        end
        replicate = try
            _regression_refit(f, curves, response, z, index)
        catch err
            err isa Union{ArgumentError,DimensionMismatch} || rethrow()
            throw(ArgumentError("regression bootstrap replicate $b failed: $(sprint(showerror,err))"))
        end
        values = _regression_replicate_values(f, replicate, grids, rgrid)
        for (draw, value) in zip(draws, values)
            selectdim(draw, 1, b) .= value
        end
    end
    settings = (; target=f isa _FunctionalFPCRegression ? :coefficient_surface : :coefficient,
        method=:bootstrap, reference="fdapace 0.6.0 FLMCI", B, quantile=:type7,
        resampling=:subjects, range_check=:all_functional_variables, refit_fpca=true,
        rng_state=initial)
    map(estimates, draws, ntuple(identity,Val(length(estimates)))) do estimate, draw, j
        lower = similar(estimate); upper = similar(estimate); se = similar(estimate)
        for c in CartesianIndices(estimate)
            column = view(draw, :, Tuple(c)...)
            lower[c] = _band_percentile(column, a/2)
            upper[c] = _band_percentile(column, 1-a/2)
            average = sum(column)/B
            se[c] = sqrt(sum(abs2, column .- average)/(B-1))
        end
        all(isfinite,lower) && all(isfinite,upper) || throw(ArgumentError(
            "bootstrap interval endpoints must be finite"))
        base = (;grid=copy(grids[j]))
        f isa _FunctionalFPCRegression && (base = merge(base,(;response_grid=copy(rgrid))))
        merge(base,(;estimate=copy(estimate),se,lower,upper,alpha=a,critical_value=nothing,settings))
    end
end

function _regression_project(m,x,evaluator)
    evaluator === nothing || evaluator isa Union{PiecewiseLinear,NoInterpolation} ||
        throw(ArgumentError("evaluator must be nothing, PiecewiseLinear(), or NoInterpolation()"))
    _regression_same_data(x,m.training_data) && return copy(scores(m))
    _regression_project_new(m,x,evaluator)
end
_regression_project_new(m,x,::Nothing) = project_scores(m,x)
function _regression_project_new(m,x::BasisFunctionalData,::Nothing)
    m.method === :basis && return project_scores(m,x)
    project_scores(m,GridFunctionalData(evaluate(x,m.grid),m.grid))
end
function _regression_project_new(m,x,evaluator::Union{PiecewiseLinear,NoInterpolation})
    (m.method === :pace || (m.method === :face && m.training_data isa IrregularFunctionalData)) &&
        x isa Union{GridFunctionalData,IrregularFunctionalData} &&
        return project_scores(m,x)
    m.method === :face && x isa GridFunctionalData && x.grid == m.grid &&
        return project_scores(m,x)
    values = x isa BasisFunctionalData ? evaluate(x,m.grid) : evaluate(x,m.grid,evaluator)
    all(isfinite,values) || throw(ArgumentError("prediction curves must cover the fitted FPCA grid under the explicit evaluator"))
    if m.method === :basis
        # Explicit numerical projection for a representation without exact basis integrals.
        mu = _fpca_mean_values(m.mean,m.grid)
        return (values .- mu)*(m.weights.*basis_matrix(eigenfunctions(m),m.grid))
    end
    project_scores(m,GridFunctionalData(values,m.grid))
end

function _regression_newdesign(f::FPCRegression{T},x,z,evaluator)::Matrix{T} where {T}
    curves = _regression_inputs(x)
    length(curves)==length(f.fpca) || throw(DimensionMismatch("prediction functional covariate count must match the regression fit"))
    scalar = _regression_scalars(z,nobs(first(curves)),eltype(f.covariance))
    size(scalar,2)==f.nscalar || throw(DimensionMismatch("prediction scalar_covariates column count must match the regression fit"))
    evaluators = _regression_each(evaluator,length(curves),"evaluator")
    projected = map(_regression_project,f.fpca,curves,evaluators)
    _regression_design(scalar,projected,f.counts,eltype(f.covariance))
end

"""
    predict(f::FPCRegression, x; scalar_covariates=nothing, evaluator=nothing,
        interval=:none, alpha=0.05, grid=nothing)

Predict new responses using stored FPCA fits and component counts.
`interval=:none` returns a vector (scalar response) or `GridFunctionalData`
of predicted curves `mean_Y + D*B*psi'` (functional response) on the response
FPCA work grid or a supplied `grid`. `:confidence` and `:prediction` return
pointwise conditional Student-t intervals with `estimate`, `se`, `lower`,
`upper`, `alpha`, `critical_value`, and `settings` (functional responses add
`grid`; arrays are subject × grid). Prediction intervals add `f.sigma2` for a
new independent Gaussian response, or `psi(s)'SigmaE psi(s)` for a new response
curve within the retained response components. All FPCA/score estimation and
selection uncertainty is excluded, including latent sparse score variance.

Native FPCA representations use its existing prediction contract: PACE accepts
both raw types, dense FPCA uses its complete training grid, and basis FPCA uses
its training basis. Basis curves can be evaluated exactly on a raw fit's grid.
For other representations, explicitly pass `evaluator=PiecewiseLinear()` or
`NoInterpolation()` (one per covariate is allowed). Curves must cover the
fitted grid. For a basis fit this explicitly requests numerical projection on
its stored work-grid quadrature, an approximation to exact basis integrals.
No mean, covariance or eigenfunction is refitted. Exact training observations
reuse the stored training scores, including FACE completion and preprocessing.
"""
Base.@constprop :aggressive function predict(f::FPCRegression,x;scalar_covariates=nothing,evaluator=nothing,
        interval=:none,alpha=0.05,grid=nothing)
    interval in (:none,:confidence,:prediction) || throw(ArgumentError("interval must be :none, :confidence, or :prediction"))
    design = _regression_newdesign(f,x,scalar_covariates,evaluator)
    _regression_predict(f,design,interval,alpha,grid)
end

Base.@constprop :aggressive function _regression_predict(f::_ScalarFPCRegression,design,interval,alpha,grid)
    grid === nothing || throw(ArgumentError("grid requires a functional response"))
    estimate = design*f.coefficients
    all(isfinite,estimate) || throw(ArgumentError("regression predictions must be finite"))
    interval === :none && return estimate
    variance = vec(sum((design*f.covariance).*design;dims=2))
    interval === :prediction && (variance .+= f.sigma2)
    _regression_intervals(f,estimate,variance,alpha,interval)
end
Base.@constprop :aggressive _regression_predict(f::_FunctionalFPCRegression,design,interval,alpha,grid) =
    _regression_functional_intervals(f,design,interval,alpha,grid)

Base.@constprop :aggressive function _regression_functional_intervals(f,design,interval,alpha,grid)
    T = eltype(f.covariance)
    g, psi, mu = _regression_response_basis(f.response_fpca,size(f.coefficients,2),
        grid === nothing ? f.response_fpca.grid : grid,T)
    estimate = design*f.coefficients*transpose(psi) .+ transpose(mu)
    all(isfinite,estimate) || throw(ArgumentError("regression predictions must be finite"))
    interval === :none && return GridFunctionalData(estimate,g)
    response_variance = vec(sum((psi*f.residual_covariance).*psi;dims=2))
    leverage = vec(sum((design*_regression_unscaled(f)).*design;dims=2))
    variance = leverage*transpose(response_variance)
    interval === :prediction && (variance .+= transpose(response_variance))
    merge((;grid=g),_regression_intervals(f,estimate,variance,alpha,interval))
end

"""
    fitted_bands(f::FPCRegression; alpha=0.05, interval=:confidence, grid=nothing)

Pointwise conditional Student-t intervals for the training responses.
`interval=:confidence` describes the mean response; `:prediction` also includes
independent response noise. Returns the same interval fields as `predict`;
functional responses use the response grid (or `grid`). The FPCA, scores and
component selection are fixed; see `coefficient_bands`.
"""
function fitted_bands(f::FPCRegression;alpha=0.05,interval=:confidence,grid=nothing)
    interval in (:confidence,:prediction) || throw(ArgumentError("interval must be :confidence or :prediction"))
    _regression_predict(f,f.design,interval,alpha,grid)
end

function Base.show(io::IO,f::FPCRegression)
    print(io,"FPCRegression(n=",nobs(f),", ncomponents=",f.counts)
    f isa _FunctionalFPCRegression && print(io,", response_ncomponents=",size(f.coefficients,2))
    print(io,", scalar_covariates=",f.nscalar,", sigma2=",f.sigma2,", R2=",f.r_squared,
        ", selection=",f.selection,", settings=",f.settings,
        ", fpca=",f.fpca)
    f isa _FunctionalFPCRegression && print(io,", response_fpca=",f.response_fpca)
    print(io,")")
end
Base.show(io::IO,::MIME"text/plain",f::FPCRegression) = show(io,f)
