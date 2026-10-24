abstract type _ComponentSelector end

"""
    FVE(threshold=0.99)

Select the smallest number of components whose cumulative fraction of variance
explained is at least `threshold`, a finite number in `(0, 1]`. The denominator
is the sum of positive eigenvalues retained up to `maxK`. The default 0.99 is
fdapace 0.6.0's `FVEthreshold` (`SetOptions.R`).
"""
struct FVE{T<:AbstractFloat} <: _ComponentSelector
    threshold::T
    function FVE(threshold::Real=0.99)
        isfinite(threshold) && 0 < threshold <= 1 ||
            throw(ArgumentError("FVE threshold must be finite and in (0, 1]"))
        value = float(threshold)
        0 < value <= 1 || throw(ArgumentError("FVE threshold must remain in (0, 1] after conversion"))
        new{typeof(value)}(value)
    end
end

"""
    AIC()

Akaike information criterion component selector. Reserved for the PACE
likelihood in task 2; dense FPCA currently throws for this selector.
"""
struct AIC <: _ComponentSelector end

"""
    BIC()

Bayesian information criterion component selector. Reserved for the PACE
likelihood in task 2; dense FPCA currently throws for this selector.
"""
struct BIC <: _ComponentSelector end

Base.show(io::IO, rule::FVE) = print(io, "FVE(", rule.threshold, ")")

"""
    fit(FPCA, fd; method=:auto, ncomponents=FVE(0.99),
        maxK=max(1, min(length(fd.grid)-2, nobs(fd)-2)), presmooth=nothing,
        eigen_tol=1e-10, sign_tol=1e-12)

Univariate functional principal component analysis. Complete grid data use the
dense route: cross-sectional mean and sample covariance with denominator `n-1`
(R's `cov`, also used by fdapace 0.6.0 `GetCovDense`). `:auto` chooses this
route for `sampling_design(fd) == :regular`; irregular or masked data and
`method=:pace` report that PACE is available in task 2.

The eigensystem of `sqrt(W)*G*sqrt(W)` uses the data's stored trapezoidal weights,
including on nonuniform grids. Eigenvalues must exceed `eigen_tol` times the
largest absolute eigenvalue. The relative cutoff `1e-10` is a package numerical
choice, adjustable here. All negative eigenvalues, including roundoff, are
recorded in `dropped` as their `number`, absolute `total`, and `share` of the
absolute spectrum. Positive eigenvalues up to `maxK` define the FVE denominator.
`ncomponents` is a positive integer or [`FVE`](@ref); the default threshold 0.99
is fdapace's. The default `maxK=max(1, min(length(fd.grid)-2, nobs(fd)-2))`
follows fdapace 0.6.0's dense cross-sectional default (`SetOptions.R`), without
the cap of 20 used on its smoothing route. An explicit dense `maxK` cannot
exceed `min(length(fd.grid), nobs(fd)-1)`. A fixed K exceeding `maxK`
or the retained positive rank throws. A zero-variance sample also throws.

Signs make `sum(phi .* mean)` nonnegative, following fdapace. When its magnitude
is at most `sign_tol * norm(phi) * norm(mean)`, the largest absolute entry is
made positive, with the first entry breaking ties. The relative tolerance
`1e-12` is an adjustable package numerical choice; the fallback follows
FunctionalGraphicalModels.jl. Orthonormality is under work-grid weights.

`presmooth=nothing` assumes negligible measurement noise, as in Ramsay and
Silverman (2005), chapter 8. Noisy dense data should use `presmooth` or
`method=:pace` (task 2). A [`LocalPolynomial`](@ref) or [`PenalizedSpline`](@ref)
is applied through `smooth_curves`, evaluated on the original data grid, and
recorded with its selected tuning metadata. Prediction applies the same fitted
preprocessing; a spline GCV choice is fixed at its training-selected lambda.
All choices are stored and shown. Inputs are copied. Dense noise variance and
score covariances are `nothing`.
"""
struct FPCA{T<:AbstractFloat,M,C,N,SC,P,S,D,R,O}
    method::Symbol
    design::Symbol
    grid::Vector{T}
    weights::Vector{T}
    mean::M
    covariance::C
    noise_variance::N
    eigenvalues::Vector{T}
    all_eigenvalues::Vector{T}
    fve::Vector{T}
    basis::FPCBasis{T}
    selection::S
    dropped::D
    sign_rule::R
    scores::Matrix{T}
    score_covariances::SC
    ids::Union{Nothing,Vector{Any}}
    presmooth::P
    settings::O
end

function _fpca_tolerance(value, name)
    value isa Real && isfinite(value) && 0 <= value < 1 ||
        throw(ArgumentError("$name must be finite and in [0, 1)"))
    value
end

# normalization_weights permits an explicit fdapace compatibility calculation:
# rectangular eigenvalue scaling, followed by trapezoidal column normalization.
# Both weights coincide in every public fit.
function _fpca_eigen(surface::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real},
        mean::AbstractVector{<:Real}; maxK=length(weights), eigen_tol=1e-10,
        sign_tol=1e-12, normalization_weights=weights)
    Base.require_one_based_indexing(surface, weights, mean, normalization_weights)
    m = length(weights)
    size(surface) == (m, m) || throw(DimensionMismatch("covariance surface must be square with one row per weight"))
    length(mean) == m || throw(DimensionMismatch("mean length must match covariance grid"))
    length(normalization_weights) == m || throw(DimensionMismatch("normalization weights must match covariance grid"))
    maxK isa Integer && 1 <= maxK <= m || throw(ArgumentError("maxK must be between 1 and the covariance grid length"))
    all(isfinite, surface) && all(isfinite, mean) || throw(ArgumentError("covariance surface and mean must be finite"))
    issymmetric(surface) || throw(ArgumentError("covariance surface must be symmetric"))
    all(w -> isfinite(w) && w > 0, weights) &&
        all(w -> isfinite(w) && w > 0, normalization_weights) ||
        throw(ArgumentError("eigenanalysis and normalization weights must be finite and positive"))
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    T = _floattype(promote_type(eltype(surface), eltype(weights), eltype(mean)))
    w = Vector{T}(weights)
    root = sqrt.(w)
    weighted = Matrix{T}(surface) .* (root * transpose(root))
    all(isfinite, weighted) || throw(ArgumentError("weighted covariance must be finite; rescale the data"))
    decomposition = eigen(Symmetric(weighted))
    spectrum = decomposition.values
    all(isfinite, spectrum) || throw(ArgumentError("FPCA eigenvalues must be finite; rescale the data"))
    scale = maximum(abs, spectrum)
    scale > 0 || throw(ArgumentError("FPCA covariance has no positive eigenvalues above eigen_tol"))
    scaled_spectrum = spectrum ./ scale
    negative = spectrum[spectrum .< 0]
    total = sum(abs, negative)
    isfinite(total) || throw(ArgumentError("FPCA negative eigenvalue total must be finite; rescale the data"))
    absolute = sum(abs, scaled_spectrum)
    dropped = (; number=length(negative), total,
        share=sum(abs, scaled_spectrum[spectrum .< 0]) / absolute)
    cutoff = eigen_tol * scale
    indices = reverse(findall(>(cutoff), spectrum))
    isempty(indices) && throw(ArgumentError("FPCA covariance has no positive eigenvalues above eigen_tol"))
    indices = indices[1:min(Int(maxK), length(indices))]
    values = spectrum[indices]
    functions = decomposition.vectors[:, indices] ./ root
    normalization_root = sqrt.(normalization_weights)
    mean_scale = maximum(abs, mean)
    scaled_mean = mean_scale == 0 ? zeros(T, m) : mean ./ mean_scale
    for j in axes(functions, 2)
        phi = view(functions, :, j)
        phi ./= norm(normalization_root .* phi)
        scaled_phi = phi ./ maximum(abs, phi)
        alignment = dot(scaled_phi, scaled_mean)
        nearzero = abs(alignment) <= sign_tol * norm(scaled_phi) * norm(scaled_mean)
        orient = nearzero ? phi[argmax(abs.(phi))] : alignment
        orient < 0 && (phi .*= -one(T))
    end
    scaled_values = values ./ scale
    cumulative = cumsum(scaled_values) ./ sum(scaled_values)
    cumulative[end] = one(T)
    (; eigenvalues=values, eigenfunctions=functions, fve=cumulative, dropped)
end

function _fpca_selection(rule::Integer, curve, maxK)
    1 <= rule <= maxK || throw(ArgumentError("ncomponents must be between 1 and maxK ($maxK)"))
    rule <= length(curve) || throw(ArgumentError("ncomponents exceeds the retained positive covariance rank ($(length(curve)))"))
    (; rule=:fixed, threshold=nothing, maxK, chosen=Int(rule), scores=eltype(curve)[])
end

function _fpca_selection(rule::FVE, curve, maxK)
    chosen = findfirst(>=(rule.threshold), curve)::Int
    (; rule=:FVE, threshold=rule.threshold, maxK, chosen, scores=copy(curve))
end

_fpca_selection(::_ComponentSelector, curve, maxK) =
    throw(ArgumentError("AIC and BIC component selection are available with PACE in task 2"))
_fpca_selection(rule, curve, maxK) =
    throw(ArgumentError("ncomponents must be a positive integer or FVE(threshold)"))

_fpca_preprocess(fd, ::Nothing) = (copy(fd.values), nothing)
function _fpca_preprocess(fd, method::LocalPolynomial)
    smoothed = smooth_curves(fd, method; grid=fd.grid)
    smoothed.observed === nothing || throw(ArgumentError("presmooth must produce complete values on the training grid"))
    smoothed.values, deepcopy(smoothed.metadata)
end
function _fpca_preprocess(fd, method::PenalizedSpline)
    smoothed = smooth_curves(fd, method)
    evaluate(smoothed, fd.grid), deepcopy(smoothed.metadata)
end
_fpca_preprocess(fd, method) =
    throw(ArgumentError("presmooth must be nothing, LocalPolynomial, or PenalizedSpline"))

function fit(::Type{FPCA}, fd::GridFunctionalData; method=:auto,
        ncomponents=FVE(0.99), maxK=max(1, min(length(fd.grid)-2, nobs(fd)-2)),
        presmooth=nothing, eigen_tol=1e-10, sign_tol=1e-12)
    method in (:auto, :dense, :pace) || throw(ArgumentError("FPCA method must be :auto, :dense, or :pace"))
    design = sampling_design(fd)
    (method == :pace || (method == :auto && design != :regular)) &&
        throw(ArgumentError("PACE FPCA is available in task 2"))
    fd.observed === nothing || throw(ArgumentError("dense FPCA requires complete grid data; PACE is available in task 2"))
    n, m = size(fd.values)
    n >= 2 || throw(ArgumentError("dense FPCA requires at least two curves"))
    maxK isa Integer && 1 <= maxK <= min(m, n-1) || throw(ArgumentError(
        "dense FPCA maxK must be between 1 and min(grid length, nobs-1)"))
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    values, preprocessing = _fpca_preprocess(fd, presmooth)
    T = eltype(values)
    mu = vec(sum(values; dims=1) ./ n)
    residuals = values .- transpose(mu)
    covariance = transpose(residuals) * residuals / (n-1)
    all(isfinite, covariance) && all(isfinite, mu) ||
        throw(ArgumentError("dense FPCA mean and covariance must be finite; rescale the observations"))
    grid, weights = Vector{T}(fd.grid), Vector{T}(fd.weights)
    system = _fpca_eigen(covariance, weights, mu; maxK, eigen_tol, sign_tol)
    selection = _fpca_selection(ncomponents, system.fve, Int(maxK))
    k = selection.chosen
    basis = FPCBasis(grid, system.eigenfunctions[:, 1:k]; weights)
    projected = residuals * (weights .* basis.values)
    all(isfinite, projected) || throw(ArgumentError("dense FPCA scores must be finite; rescale the observations"))
    settings = (; requested_method=method, eigen_tol, sign_tol,
        covariance_denominator=n-1, quadrature=:stored_trapezoidal,
        preprocessing)
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry, tolerance=sign_tol)
    FPCA(:dense, design, grid, weights, GridFunctionalData(reshape(mu, 1, :), grid),
        covariance, nothing, system.eigenvalues[1:k], system.eigenvalues,
        system.fve, basis, selection, system.dropped, sign_rule, projected,
        nothing, deepcopy(fd.ids), deepcopy(presmooth), settings)
end

function fit(::Type{FPCA}, fd::IrregularFunctionalData; method=:auto, kwargs...)
    method in (:auto, :dense, :pace) || throw(ArgumentError("FPCA method must be :auto, :dense, or :pace"))
    method == :dense && throw(ArgumentError("dense FPCA requires complete GridFunctionalData"))
    throw(ArgumentError("PACE FPCA is available in task 2"))
end

"""
    eigenfunctions(f::FPCA)

Return the selected eigenfunctions as an [`FPCBasis`](@ref).
"""
eigenfunctions(f::FPCA) = f.basis

"""
    eigvals(f::FPCA)

Return the selected FPCA eigenvalues (the `LinearAlgebra.eigvals` generic).
"""
LinearAlgebra.eigvals(f::FPCA) = f.eigenvalues

"""
    scores(f::FPCA)

Return the training score matrix, with subjects in rows and components in columns.
The returned matrix is the fit's own storage: mutating it changes the fit and
subsequent reconstruction. Use `copy(scores(f))` for an independent matrix.
"""
scores(f::FPCA) = f.scores

"""
    score_covariances(f::FPCA)

Return `nothing` for dense FPCA. PACE will return conditional score covariance
matrices of size `K × K × n` in task 2.
"""
score_covariances(f::FPCA) = f.score_covariances

"""
    fve(f::FPCA)

Return cumulative fractions of variance explained for all retained positive
eigenvalues up to `maxK`, including those beyond the selected components.
"""
fve(f::FPCA) = f.fve

"""
    noise_variance(f::Union{FPCA,SmoothedCovariance})

Return estimated measurement-error variance. Dense FPCA returns `nothing`;
`SmoothedCovariance` returns its stored nonnegative variance estimate.
"""
noise_variance(f::Union{FPCA,SmoothedCovariance}) = f.noise_variance

"""
    ncomponents(f::FPCA)

Return the number of selected functional principal components.
"""
ncomponents(f::FPCA) = length(f.eigenvalues)
nobs(f::FPCA) = size(f.scores, 1)
ids(f::FPCA) = f.ids === nothing ? nothing : deepcopy(f.ids)

function _fpca_prediction_method(f)
    method = f.presmooth
    if method isa PenalizedSpline && method.lambda isa GCV
        return PenalizedSpline(method.basis; lambda=f.settings.preprocessing.lambda, order=method.order)
    end
    method
end

"""
    predict(f::FPCA, newdata::GridFunctionalData)

Project new dense curves using the training mean, basis, weights, and fitted
preprocessing. Return an `n × K` score matrix. Data must be complete and use
exactly the training grid. This extends `StatsAPI.predict`, as in
MultivariateStats.jl's PCA projection interface.
"""
function predict(f::FPCA, fd::GridFunctionalData)
    length(fd.grid) == length(f.grid) || throw(DimensionMismatch("prediction grid length must match the FPCA training grid"))
    fd.grid == f.grid || throw(ArgumentError("prediction grid must exactly match the FPCA training grid"))
    fd.observed === nothing || throw(ArgumentError("dense FPCA prediction requires complete grid data"))
    values, _ = _fpca_preprocess(fd, _fpca_prediction_method(f))
    projected = (values .- f.mean.values) * (f.weights .* f.basis.values)
    all(isfinite, projected) || throw(ArgumentError("FPCA prediction scores must be finite; rescale the observations"))
    projected
end

predict(f::FPCA, fd::IrregularFunctionalData) =
    throw(ArgumentError("irregular FPCA prediction is available with PACE in task 2"))

"""
    reconstruct(f::FPCA, scores=scores(f); grid=f.grid)

Return `GridFunctionalData` containing `mean + scores * eigenfunctions'`.
The finite increasing grid must lie inside the work-grid domain. Both the mean
and eigenfunctions are linearly interpolated. Subject ids are retained when
the supplied scores are the training score object; other score matrices have
no inferred ids. Scores must be finite with one column per selected component.
"""
function reconstruct(f::FPCA, projected::AbstractMatrix{<:Real}=scores(f);
        grid::AbstractVector{<:Real}=f.grid)
    Base.require_one_based_indexing(projected, grid)
    size(projected, 1) >= 1 || throw(ArgumentError("reconstruction requires at least one score row"))
    size(projected, 2) == ncomponents(f) || throw(DimensionMismatch("reconstruction score columns must equal ncomponents"))
    all(isfinite, projected) || throw(ArgumentError("reconstruction scores must be finite"))
    quadrature_weights(grid)
    functions = basismatrix(f.basis, grid)
    mu = evaluate(f.mean, grid, PiecewiseLinear())
    reconstructed = projected * transpose(functions) .+ mu
    GridFunctionalData(reconstructed, grid; ids=projected === f.scores ? f.ids : nothing)
end

function Base.show(io::IO, f::FPCA)
    print(io, "FPCA(method=", f.method, ", design=", f.design, ", n=", nobs(f),
        ", grid=", length(f.grid), " points, ncomponents=", ncomponents(f),
        ", selection=", f.selection, ", presmooth=", f.presmooth,
        ", sign_rule=", f.sign_rule, ", settings=", f.settings,
        ", dropped=", f.dropped, ", noise_variance=", f.noise_variance, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::FPCA) = show(io, f)
