const _RegressionCurve = Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData}
const _RegressionScalars = Union{AbstractVector{<:Union{Missing,Real}},AbstractMatrix{<:Union{Missing,Real}}}

"""
    FPCRegression
    fit(FPCRegression, x, y; scalar_covariates=nothing, fpca=nothing,
        fpca_options=(;), ncomponents=nothing, selection=:fpca)

Gaussian scalar-on-function regression on leading FPCA scores. `x` is one
univariate functional dataset, a tuple/vector of them, or
`MultivariateFunctionalData`. `y` is a finite response vector. Scalar covariates
are an optional finite vector or matrix, without an intercept column.

The model is `y = intercept + Z*gamma + sum(scores_j*b_j) + error`;
`coefficient_functions` returns each `beta_j = eigenfunctions_j*b_j`.
`coef` and `vcov` order parameters as intercept, scalar coefficients, then
score coefficients in covariate order. `dof` is their count and
`dof_residual = nobs-dof`. `sigma2` is residual sum of squares / residual dof.
Rank-deficient designs, saturated models and constant responses throw.

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
struct FPCRegression{T<:AbstractFloat,B,S,O}
    # FPCA route/settings do not change the regression result's concrete type.
    # This mirrors FSVD's abstract FPCA fields and keeps automatic fits inferred.
    fpca::Tuple{Vararg{FPCA}}
    beta::B
    coefficients::Vector{T}
    covariance::Matrix{T}
    design::Matrix{T}
    response::Vector{T}
    fitted_values::Vector{T}
    residual_values::Vector{T}
    sigma2::T
    r_squared::T
    counts::Vector{Int}
    nscalar::Int
    selection::S
    settings::O
end

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

Base.@constprop :aggressive function _regression_fpca(x, supplied, options, rule, search)
    options isa NamedTuple || throw(ArgumentError("fpca_options entries must be named tuples"))
    for key in (:ncomponents,:max_k,:maxK)
        haskey(options,key) && options[key] isa Bool && throw(ArgumentError(
            "fpca_options.$key cannot be Boolean"))
    end
    if supplied === nothing
        search && haskey(options, :ncomponents) && throw(ArgumentError(
            "regression selection uses all candidate components; cap them with fpca_options.max_k"))
        rule !== nothing && haskey(options, :ncomponents) && throw(ArgumentError(
            "ncomponents is specified both directly and in fpca_options"))
        effective = search ? merge(options, (;ncomponents=FVE(1))) :
            rule === nothing ? options : merge(options, (;ncomponents=rule))
        return fit(FPCA, x; effective...)
    end
    supplied isa FPCA || throw(ArgumentError("fpca entries must be FPCA fits or nothing"))
    isempty(options) || throw(ArgumentError("fpca_options cannot tune an already supplied FPCA"))
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

function _regression_ols(design::Matrix{T}, y::Vector{T}) where {T}
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
    sigma2 = rss/(n-p)
    inverseR = UpperTriangular(R) \ Matrix{T}(I,p,p)
    covpivot = sigma2 * (inverseR * transpose(inverseR))
    covariance = similar(covpivot)
    covariance[qrfit.p,qrfit.p] = covpivot
    all(isfinite, b) && all(isfinite, covariance) && isfinite(sigma2) ||
        throw(ArgumentError("regression coefficients and covariance must be finite; rescale the inputs"))
    (; b, fitted_values, residual_values, rss, sigma2, covariance, tolerance)
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

Base.@constprop :aggressive function fit(::Type{FPCRegression}, x, y::AbstractVector{<:Real};
        scalar_covariates=nothing, fpca=nothing, fpca_options=(;),
        ncomponents=nothing, selection=:fpca)
    curves = _regression_inputs(x)
    n, p = nobs(first(curves)), length(curves)
    Base.require_one_based_indexing(y)
    length(y) == n || throw(DimensionMismatch("response length must equal the functional covariate subject count"))
    all(isfinite, y) || throw(ArgumentError("response must be finite"))
    supplied = _regression_each(fpca,p,"fpca")
    options = _regression_each(fpca_options,p,"fpca_options")
    rules = _regression_each(ncomponents,p,"ncomponents")
    any(r -> r isa Bool, rules) && throw(ArgumentError("ncomponents cannot be Boolean"))
    search = selection !== :fpca
    (selection === :fpca || selection isa Union{AIC,BIC} || selection === :loocv) ||
        throw(ArgumentError("selection must be :fpca, AIC(), BIC(), or :loocv"))
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
    FPCRegression(models,beta,ols.b,ols.covariance,design,response,ols.fitted_values,
        ols.residual_values,ols.sigma2,one(T)-ols.rss/total,counts,size(z,2),tuning,settings)
end
function fit(::Type{FPCRegression},x,y::AbstractVector;kwargs...)
    Base.require_one_based_indexing(y)
    missing_index = findfirst(ismissing,y)
    missing_index === nothing || throw(ArgumentError("response contains missing at index $missing_index"))
    T = Base.nonmissingtype(eltype(y))
    T <: Real || throw(ArgumentError("response must be a finite real vector"))
    fit(FPCRegression,x,Vector{T}(y);kwargs...)
end
fit(::Type{FPCRegression},x,y;kwargs...) = throw(ArgumentError("response must be a finite real vector"))

"""
    coefficient_functions(f::FPCRegression)

Return a tuple of copied `BasisFunctionalData` coefficient functions, one per
functional covariate. Evaluate them using `evaluate(beta, t)`.
"""
coefficient_functions(f::FPCRegression) = deepcopy(f.beta)
"""Return copied coefficients ordered as intercept, scalars, then each covariate's FPC scores."""
coef(f::FPCRegression) = copy(f.coefficients)
"""Return the conditional OLS coefficient covariance in the order used by `coef`."""
vcov(f::FPCRegression) = copy(f.covariance)
"""Return copied fitted scalar responses from an FPC regression."""
fitted(f::FPCRegression) = copy(f.fitted_values)
"""Return copied response-minus-fitted residuals from an FPC regression."""
residuals(f::FPCRegression) = copy(f.residual_values)
"""Return the number of subjects used in an FPC regression."""
nobs(f::FPCRegression) = length(f.response)
"""Return the number of fitted regression coefficients, including the intercept."""
dof(f::FPCRegression) = length(f.coefficients)
"""Return `nobs(f)-dof(f)`, the residual degrees of freedom."""
dof_residual(f::FPCRegression) = nobs(f)-dof(f)
"""Return the ordinary centered coefficient of determination of an FPC regression."""
r2(f::FPCRegression) = f.r_squared
"""Return conditional OLS coefficient standard errors in the order used by `coef`."""
stderror(f::FPCRegression) = sqrt.(max.(diag(f.covariance),zero(eltype(f.coefficients))))
"""Return a copied vector of selected FPC counts, one per functional covariate."""
ncomponents(f::FPCRegression) = copy(f.counts)

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

"""
    coefficient_bands(f::FPCRegression; alpha=0.05, grid=nothing)

Pointwise Student-t intervals for each coefficient function, conditional on the
fitted FPCA and selected score design, as in fda.usc 2.2.0 `fregre.pc`'s ordinary
linear model. Return a tuple of named tuples with `grid`, `estimate`, `se`,
`lower`, `upper`, `alpha`, `critical_value` and `settings`. `grid=nothing` uses
each FPCA's work grid; otherwise supply one grid for one predictor, or a tuple
of grids. FPCA means, harmonics, sparse scores, component selection and omitted
components are treated as fixed. These are pointwise intervals, not simultaneous
bands, and are not fdapace's FPCA-refitting bootstrap `FLMCI` intervals.
"""
function coefficient_bands(f::FPCRegression;alpha=0.05,grid=nothing)
    grids = grid === nothing ? map(m -> m.grid,f.fpca) :
        length(f.fpca)==1 && grid isa AbstractVector{<:Real} ? (grid,) :
        _regression_each(grid,length(f.fpca),"grid")
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
_regression_project_new(m,x,evaluator) = throw(ArgumentError("evaluator must be nothing, PiecewiseLinear(), or NoInterpolation()"))

function _regression_newdesign(f::FPCRegression{T},x,z,evaluator)::Matrix{T} where {T}
    curves = _regression_inputs(x)
    length(curves)==length(f.fpca) || throw(DimensionMismatch("prediction functional covariate count must match the regression fit"))
    scalar = _regression_scalars(z,nobs(first(curves)),eltype(f.coefficients))
    size(scalar,2)==f.nscalar || throw(DimensionMismatch("prediction scalar_covariates column count must match the regression fit"))
    evaluators = _regression_each(evaluator,length(curves),"evaluator")
    projected = map(_regression_project,f.fpca,curves,evaluators)
    _regression_design(scalar,projected,f.counts,eltype(f.coefficients))
end

"""
    predict(f::FPCRegression, x; scalar_covariates=nothing, evaluator=nothing,
        interval=:none, alpha=0.05)

Predict new scalar responses using stored FPCA fits and component counts.
`interval=:none` returns a vector. `:confidence` and `:prediction` return
pointwise conditional Student-t intervals with `estimate`, `se`, `lower`,
`upper`, `alpha`, `critical_value`, and `settings`. Prediction intervals add
`f.sigma2` for a new independent Gaussian response. All FPCA/score estimation
and selection uncertainty is excluded, including latent sparse score variance.

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
        interval=:none,alpha=0.05)
    interval in (:none,:confidence,:prediction) || throw(ArgumentError("interval must be :none, :confidence, or :prediction"))
    design = _regression_newdesign(f,x,scalar_covariates,evaluator)
    estimate = design*f.coefficients
    all(isfinite,estimate) || throw(ArgumentError("regression predictions must be finite"))
    interval === :none && return estimate
    variance = vec(sum((design*f.covariance).*design;dims=2))
    interval === :prediction && (variance .+= f.sigma2)
    _regression_intervals(f,estimate,variance,alpha,interval)
end

"""
    fitted_bands(f::FPCRegression; alpha=0.05, interval=:confidence)

Pointwise conditional Student-t intervals for the training scalar responses.
`interval=:confidence` describes the mean response; `:prediction` also includes
independent response noise. Returns the same interval fields as `predict`.
The FPCA, scores and component selection are fixed; see `coefficient_bands`.
"""
function fitted_bands(f::FPCRegression;alpha=0.05,interval=:confidence)
    interval in (:confidence,:prediction) || throw(ArgumentError("interval must be :confidence or :prediction"))
    variance = vec(sum((f.design*f.covariance).*f.design;dims=2))
    interval === :prediction && (variance .+= f.sigma2)
    _regression_intervals(f,f.fitted_values,variance,alpha,interval)
end

function Base.show(io::IO,f::FPCRegression)
    print(io,"FPCRegression(n=",nobs(f),", ncomponents=",f.counts,
        ", scalar_covariates=",f.nscalar,", sigma2=",f.sigma2,", R2=",f.r_squared,
        ", selection=",f.selection,", settings=",f.settings,
        ", fpca=",f.fpca,")")
end
Base.show(io::IO,::MIME"text/plain",f::FPCRegression) = show(io,f)
