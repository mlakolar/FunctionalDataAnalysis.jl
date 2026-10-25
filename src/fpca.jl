abstract type _ComponentSelector end

const _FPCA_EIGEN_TOL = 1e-10
const _FPCA_SIGN_TOL = 1e-12

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

Akaike information criterion component selector for PACE. Minimize minus twice
the Gaussian log likelihood plus `2K`, following Yao, Müller, and Wang (2005),
section 2.5. Selection stops at the first increase, following fdapace 0.6.0;
the full criterion curve and global minimizer are also recorded. Dense FPCA
throws for this selector.
"""
struct AIC <: _ComponentSelector end

"""
    BIC()

Bayesian information criterion component selector for PACE. Minimize minus
twice the Gaussian log likelihood plus `K*log(nobs)`, following Yao, Müller,
and Wang (2005), section 2.5. Selection stops at the first increase, following
fdapace 0.6.0; the full criterion curve and global minimizer are also recorded.
Dense FPCA throws for this selector.
"""
struct BIC <: _ComponentSelector end

Base.show(io::IO, rule::FVE) = print(io, "FVE(", rule.threshold, ")")

"""
    fit(FPCA, fd; method=:auto, grid=nothing, ncomponents=FVE(0.99),
        maxK=nothing, presmooth=nothing, mean=nothing, covariance=nothing,
        noise_variance=nothing, mean_bandwidth=nothing,
        covariance_bandwidth=nothing, diagonal_bandwidth=nothing, kernel=nothing,
        weighting=:obs, binning=:auto, face=(;), eigen_tol=1e-10, sign_tol=1e-12)

Univariate functional principal component analysis. Complete grid data use the
dense route: cross-sectional mean and sample covariance with denominator `n-1`
(R's `cov`, also used by fdapace 0.6.0 `GetCovDense`). `:auto` chooses this
route for `sampling_design(fd) == :regular`; irregular or masked data use
PACE. `method=:pace` forces PACE on either data representation, while
`method=:dense` requires complete `GridFunctionalData`. Pooled smoothing
keywords are accepted only on the PACE route.

`BasisFunctionalData` dispatches to exact-basis roughness-penalized FPCA;
see `fit(FPCA, fd::BasisFunctionalData)` for its lambda, penalty, centering,
normalization, and leave-one-curve-out selection settings.

`method=:face` explicitly selects fast covariance estimation for common-grid
data, including masked observations (Xiao et al., 2016). Its `face` named tuple
accepts `knots=35`, `p=3`, `m=2`, `lambda=nothing`, `alpha=1`, `center=true`,
`loglambda=range(-20,20; length=100)`, and `mean_smoother=nothing`. These follow
refund 0.1-40 `fpca.face`, except for the mean smoother described below.
The audited `knots` counts equal intervals (34 strictly interior knots), and
the simple knot sequence extends `p` intervals beyond each endpoint.
`p` is spline degree, `m` is coefficient-difference order. Rank-deficient
designs throw instead of refund's undocumented ridge. `lambda=nothing`
minimizes refund's closed-form GCV over `exp.(loglambda)`; ties use the first
candidate. A finite nonnegative fixed lambda is also accepted. The full
candidate grid and criterion are stored even for a fixed lambda. `maxK`
defaults to `min(nobs, knots+p)`; FVE uses the retained positive spectrum.
Fixed K, FVE, numerical eigenvalue cutoff and sign rule reuse the other routes;
AIC and BIC throw. Refund uses strict `FVE > threshold`, while the package
consistently uses `>=` and supports `FVE(1)`.

FACE's default mean reuses [`PenalizedSpline`](@ref) with a clamped cubic basis
of `min(38, grid_length)` functions, integrated second-derivative penalty,
and [`GCV`](@ref) over `exp.(range(-20,20; length=100))*domain_width^3`.
This bounded basis preserves FACE's scaling and reuses the package smoother;
refund uses `stats::smooth.spline(all.knots=TRUE)`. Override with an explicit
`face=(; mean_smoother=PenalizedSpline(...))`, or supply `mean` as for PACE.
A supplied mean cannot be combined with a mean smoother or `center=false`.
Turning off centering uses a zero mean.

Masked FACE curves initialize with R 4.5 `smooth.spline`'s default reduced
knot scheme and GCV (including its legacy `sgram` quadrature), then follow
refund's iteration: at most 100 passes, relative change of smoothed coordinates
at most 0.02, lambda fixed after the first pass. Fully missing curves throw;
incomplete curves need four observations for initialization. Estimating a mean
requires at least one observation per grid column. Iterations, convergence,
changes, initialization tuning, and completed centered data are stored.
Training scores and uncertainties use the completed data as refund does.

FACE covariance storage is low rank; fitting never forms the grid-square
covariance. Weighted eigenanalysis gives the same stored-quadrature L² scale
as dense/PACE. On rectangular weights Δ, refund's eigenvalues, eigenfunctions,
and scores convert by Δ, `1/sqrt(Δ)`, and `sqrt(Δ)`, respectively; trapezoidal
weights require diagonalizing the small weighted operator. Raw noise variance is
`mean(observed_centered_y²) - tr(smoothed_covariance)/grid_length` on the
measurement scale. Each completion iteration uses `max(raw_noise_variance,
0.025V)`, where `V` is the maximum diagonal of its full smoothed covariance
on the grid. This relative floor departs from refund's zero truncation to
stabilize conditional scores, including partial predictions.
Scores use Gaussian shrinkage under the selected
covariance, computed by the low-rank conditional-expectation solver. FACE
[`fitted_bands`](@ref) provides pointwise posterior variance consistent with
those scores, correcting refund's use of eigenvalues in place of their
inverses. Simultaneous FACE bands are unsupported. See the FPCA documentation
for the source audit and exact scale and variance formulas.

For `IrregularFunctionalData`, `method=:face` fits sparse FACE (Xiao, Li,
Checkley and Crainiceanu, 2018), following face 0.1-8 `face.sparse`.
`:auto` still selects PACE. Sparse `face` defaults are `knots=7` equal
intervals (six strictly interior knots), `p=3`, `m=2`, `center=true`,
`lambda=nothing`, and `loglambda=range(-3,10; length=14)`. Knots span the
observed range expanded by 0.1% at each end. `grid=nothing` uses 100 equally
spaced points across the observed range; an override must cover all observations
and stay within that range, with full spline rank. `maxK` defaults to `knots+p`.
The reference's `gcv` approximates leave-one-subject-out squared prediction
error to first order, omitting the constant sum of squared raw products.
It fits the covariance and noise jointly, including diagonal raw products.
Its explicit covariance ridge is `1e-6*maximum(eigvals(X'X))`, recorded in
`settings.covariance_ridge`. Estimated noise is `max(raw_noise_variance, 0.025V)`,
where `V` is the maximum fitted variance of the full positive spline covariance
over the work grid and all training observation times. This scale-invariant
floor deliberately replaces face 0.1-8's absolute `1e-6` threshold.

Sparse FACE's estimated mean uses the shared P-spline building block with the
same interval count, cubic degree, second differences, and observation weights
`1/N_i`. Its independent `lambda_mean=nothing` selects among
`exp.(mean_loglambda)`, where `mean_loglambda=range(-20,20; length=100)`;
these defaults come from face's `pspline`. A fixed `lambda_mean` or a supplied
PACE-style `mean` is accepted. The mean's reference ridge rule adds `1e-6`
when its Gram matrix has an eigenvalue at most `1e-7`; its value is stored.
Sparse scores use exact spline evaluations and the full positive spline
covariance in the shared conditional-expectation solver, as in
`predict.face.sparse`. Both data representations can be predicted inside the
work grid. Pointwise `fitted_bands` describe the selected-component posterior.
FVE consistently uses weighted functional eigenvalues; face's component count
instead uses coefficient-space eigenvalues. Non-default `m` consistently means
the requested difference order; face's covariance penalty is always second
order despite its `m` argument. `two_step` is unsupported and throws.
All grids, criteria, ridges, signs, and component choices are stored and shown.

PACE implements the conditional-expectation method of Yao, Müller, and Wang
(2005). Its work `grid` defaults to 51 equally spaced points over the observed
time range (fdapace 0.6.0 `nRegGrid`, `SetOptions.R`). An override must cover
every observation time. A missing `mean` estimates [`SmoothedMean`](@ref);
a supplied `SmoothedMean` must have `derivative=0` and is copied, or supply
`(; grid, values)` for a linearly interpolated mean. Its grid must cover the
work grid and every observation. A missing `covariance` estimates
[`SmoothedCovariance`](@ref), using an estimated
or supplied `SmoothedMean`. Alternatively supply that fitted type or a
symmetric `(; grid, surface)` for bilinear interpolation.
`noise_variance=nothing` uses `max(raw_noise_variance, 0.025V)` from the
covariance smoother; a supplied surface requires a finite nonnegative override.
Here `V` is the maximum diagonal of the full positive projected covariance
retained up to `maxK` on the work grid. PACE and both FACE routes use this
relative floor to stabilize conditional-expectation systems. It deliberately
departs from fdapace 0.6.0's negative-only replacement by `1e-6`, face 0.1-8's
absolute `1e-6` floor, and refund 0.1-40's zero truncation. Multiplying observation
units by `a` multiplies both the raw variance and floor by `a²`; changing time
units does not change their ratio. fdapace's optional `rho` regularization is
prior art for stabilizing conditional scores, but this fixed rule does not
reproduce its selector. The factor `0.025` bounds the observation-covariance
condition number by `1+40m` for `m` observations whose fitted variances are at
most `V`. At a target with fitted variance at most `V`, the Euclidean gain
from the observation residual vector to the conditional reconstructed deviation
is at most `sqrt(10)`. Sparse spline predictions at unseen times can have larger
fitted variance and are not covered by that uniform bound.
A supplied `noise_variance`, including zero, is used unchanged. The fit stores
`settings.raw_noise_variance` before zero truncation, `settings.noise_floor`
(`0.025V` when estimated, `nothing` when supplied), `settings.noise_floor_relative`
(`0.025`), `settings.noise_floor_scale` (`V`), and
`settings.noise_floor_scale_source` (`:maximum_fitted_pointwise_variance`).
These three scale fields are `nothing` for supplied noise.
`settings.noise_floor_applied` is true only when the raw estimate is below
the floor. `settings.noise_variance_source` and `settings.noise_truncation`
(`:positive_floor` when estimated, `:none` when supplied) retain provenance.
These settings are shown with the fit; [`noise_variance`](@ref) returns the
value used for scoring and prediction. Standalone `SmoothedCovariance` retains
its zero truncation.
Supplying a fitted mean together with `mean_bandwidth`, or a covariance with
`covariance_bandwidth` or `diagonal_bandwidth`, throws rather than ignoring
the requested tuning of an input that is not refitted.
With a supplied covariance, `binning` must remain `:auto`; the fit records
`binning=nothing` and `binning_rule=:not_applicable`. When both inputs are
supplied, `kernel` must remain `nothing` and `weighting` must remain `:obs`;
their active settings are recorded as `nothing` because no smoother runs.
Supplied fitted inputs retain their original smoothing settings.

PACE uses local linear degree one (Yao, Müller, and Wang, 2005). `kernel=nothing`
resolves to Gaussian, and `weighting=:obs` follows fdapace 0.6.0 `SetOptions.R`;
`:subj` is also available (neither scheme uniformly dominates: Zhang and Wang,
2016, doi:10.1214/16-AOS1446). The standalone smoothers retain Epanechnikov.
`mean_bandwidth=nothing` and `covariance_bandwidth=nothing` resolve respectively
to 0.05 and 0.10 times the observed time range (fdapace `methodBwMu="Default"`,
`methodBwCov="Default"`). Explicit numbers, [`CurveCV`](@ref), and mean
KernelSmoothing selectors pass unchanged to the existing smoothers.
`diagonal_bandwidth=nothing` uses the selected covariance bandwidth, following
fdapace and milestone 2; override it with a number or univariate selector.
[`pace_bandwidth_candidates`](@ref) supplies fdapace's optional candidate grids.

`binning=:auto` follows fdapace `GetBinNum`: use the median curve count for
fdapace's Sparse classification and the maximum otherwise. Classification
uses `IsRegular`'s pooled observed-time coverage, as described in
[`pace_bandwidth_candidates`](@ref), independently of the FPCA route.
Counts at most 20 never bin;
counts above 400 initially use 400 bins. If `n>5000`, set
`mstar=max(20,400+19*(5000-n)/2250)` and use `ceil(mstar)` bins only when
the unrounded `mstar` is below the count.
Otherwise no bins are selected. Override with `nothing` or an integer >=2.
The selected count is passed to the existing covariance smoother, which
preserves raw pair weights and snaps covariance/diagonal training times;
the mean and output grid are unchanged. This differs from fdapace's raw-curve
averaging and possible output-grid reduction. Both the request and resolved
count are stored. With `P=sum(N_i*(N_i-1))` pairs and `M=length(grid)`, fixed
covariance smoothing costs `O(M^2*P)` time. A 2D `CurveCV` with `H` candidates
costs `O(H*P^2 + H*F*P)` for `F` folds, before the final fit; binning reduces
`P` to the number of occupied per-curve cells (at most `n*binning^2`).
`maxK=nothing` resolves to `max(1,min(20,length(grid)-2,nobs(fd)-2))` for
PACE, following fdapace 0.6.0's smoothed route. An explicit PACE cap may be
at most `length(grid)`.

The full positive-semidefinite covariance projection up to `maxK` is used
in each curve's Cholesky score solve. The selected eigenvalues and
eigenfunctions define the score numerator and conditional covariance.
PACE also supports [`AIC`](@ref) and [`BIC`](@ref). Their criteria use the
K-component covariance, including the Gaussian normalization constant, for
every candidate `K=1:maxK`. Beyond the retained positive rank the covariance
has zero additional variance, while the penalty still increases. Selection
stops at the first increase as fdapace does; the global minimizer is retained too.
Scoring adds no ridge beyond the recorded estimated-noise floor. A
non-positive-definite observation covariance throws with the curve index and
a suggestion to supply `noise_variance`.

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
`method=:pace`. A [`LocalPolynomial`](@ref) or [`PenalizedSpline`](@ref)
is applied through `smooth_curves`, evaluated on the original data grid, and
recorded with its selected tuning metadata. Prediction applies the same fitted
preprocessing; a spline GCV choice is fixed at its training-selected lambda.
All choices are stored and shown. `training_data` retains a copy of the original
observations (before presmoothing) for [`fitted_curve_data`](@ref).
Inputs are copied. Dense noise variance and
score covariances are `nothing`.
"""
struct FPCA{T<:AbstractFloat,M,C,N,SC,P,S,D,R,O,B,TD}
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
    basis::B
    selection::S
    dropped::D
    sign_rule::R
    scores::Matrix{T}
    score_covariances::SC
    ids::Union{Nothing,Vector{Any}}
    presmooth::P
    settings::O
    training_data::TD
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
        mean::AbstractVector{<:Real}; maxK=length(weights), eigen_tol=_FPCA_EIGEN_TOL,
        sign_tol=_FPCA_SIGN_TOL, normalization_weights=weights)
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
    for j in axes(functions, 2)
        phi = view(functions, :, j)
        phi ./= norm(normalization_root .* phi)
    end
    _fpca_orient!(functions, mean, sign_tol)
    scaled_values = values ./ scale
    cumulative = cumsum(scaled_values) ./ sum(scaled_values)
    cumulative[end] = one(T)
    (; eigenvalues=values, eigenfunctions=functions, fve=cumulative, dropped)
end

function _fpca_orient!(functions, mean, sign_tol)
    m = length(mean)
    mean_scale = maximum(abs, mean)
    T = eltype(functions)
    scaled_mean = mean_scale == 0 ? zeros(T, m) : mean ./ mean_scale
    for j in axes(functions, 2)
        phi = view(functions, :, j)
        scaled_phi = phi ./ maximum(abs, phi)
        alignment = dot(scaled_phi, scaled_mean)
        nearzero = abs(alignment) <= sign_tol * norm(scaled_phi) * norm(scaled_mean)
        orient = nearzero ? phi[argmax(abs.(phi))] : alignment
        orient < 0 && (phi .*= -one(T))
    end
    functions
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
    throw(ArgumentError("AIC and BIC component selection require method=:pace"))
_fpca_selection(rule, curve, maxK) =
    throw(ArgumentError("ncomponents must be a positive integer, FVE(threshold), or (for PACE) AIC() or BIC()"))

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

Base.@constprop :aggressive function fit(::Type{FPCA}, fd::GridFunctionalData; method=:auto, grid=nothing,
        ncomponents=FVE(0.99), maxK=nothing, presmooth=nothing,
        mean=nothing, covariance=nothing, noise_variance=nothing,
        mean_bandwidth=nothing, covariance_bandwidth=nothing, diagonal_bandwidth=nothing,
        kernel=nothing, weighting=:obs, binning=:auto, face=(;),
        eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL)
    method in (:auto, :dense, :pace, :face) || throw(ArgumentError("FPCA method must be :auto, :dense, :pace, or :face"))
    face isa NamedTuple || throw(ArgumentError("face must be a named tuple of FACE settings"))
    if method == :face
        (grid === nothing && presmooth === nothing && covariance === nothing &&
            noise_variance === nothing && mean_bandwidth === nothing &&
            covariance_bandwidth === nothing && diagonal_bandwidth === nothing &&
            kernel === nothing && weighting == :obs && binning === :auto) ||
            throw(ArgumentError("FACE uses the data grid and its own smoothing and noise estimates; grid, presmooth and PACE tuning are unsupported"))
        return _fit_face(fd; ncomponents, maxK, mean, eigen_tol, sign_tol, face...)
    end
    isempty(face) || throw(ArgumentError("face settings require method=:face"))
    design = sampling_design(fd)
    if method == :pace || (method == :auto && design != :regular)
        presmooth === nothing || throw(ArgumentError("presmooth is only supported by dense FPCA; PACE smooths the pooled mean and covariance"))
        return _fit_pace(fd; method, grid, ncomponents, maxK, mean, covariance,
            noise_variance, mean_bandwidth, covariance_bandwidth, diagonal_bandwidth, kernel,
            weighting, binning, eigen_tol, sign_tol)
    end
    fd.observed === nothing || throw(ArgumentError("dense FPCA requires complete grid data; use method=:pace"))
    (grid === nothing && mean === nothing && covariance === nothing &&
        noise_variance === nothing && mean_bandwidth === nothing &&
        covariance_bandwidth === nothing && diagonal_bandwidth === nothing &&
        binning === :auto && weighting == :obs && kernel === nothing) ||
        throw(ArgumentError("grid, mean, covariance, noise_variance and pooled smoothing options require method=:pace"))
    n, m = size(fd.values)
    n >= 2 || throw(ArgumentError("dense FPCA requires at least two curves"))
    maxK = maxK === nothing ? max(1, min(m-2, n-2)) : maxK
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
    covariance_functions = system.eigenfunctions .* transpose(sqrt.(system.eigenvalues))
    full_projected_covariance = covariance_functions * transpose(covariance_functions)
    all(isfinite, full_projected_covariance) || throw(ArgumentError(
        "dense FPCA fitted covariance must be finite; rescale the observations"))
    settings = (; requested_method=method, grid_source=:data, ncomponents=deepcopy(ncomponents),
        maxK=Int(maxK), eigen_tol, sign_tol,
        covariance_denominator=n-1, quadrature=:stored_trapezoidal,
        preprocessing, observations=:copied_original, full_projected_covariance)
    sign_rule = (; method=:mean_alignment, fallback=:largest_absolute_entry, tolerance=sign_tol)
    FPCA(:dense, design, grid, weights, GridFunctionalData(reshape(mu, 1, :), grid),
        covariance, nothing, system.eigenvalues[1:k], system.eigenvalues,
        system.fve, basis, selection, system.dropped, sign_rule, projected,
        nothing, deepcopy(fd.ids), deepcopy(presmooth), settings, _convert_component(T, fd))
end

Base.@constprop :aggressive function fit(::Type{FPCA}, fd::IrregularFunctionalData; method=:auto, grid=nothing,
        ncomponents=FVE(0.99), maxK=nothing, presmooth=nothing,
        mean=nothing, covariance=nothing, noise_variance=nothing,
        mean_bandwidth=nothing, covariance_bandwidth=nothing, diagonal_bandwidth=nothing,
        kernel=nothing, weighting=:obs, binning=:auto, face=(;),
        eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL)
    face isa NamedTuple || throw(ArgumentError("face must be a named tuple of FACE settings"))
    if method == :face
        (presmooth === nothing && covariance === nothing && noise_variance === nothing &&
            mean_bandwidth === nothing && covariance_bandwidth === nothing &&
            diagonal_bandwidth === nothing && kernel === nothing && weighting == :obs &&
            binning === :auto) || throw(ArgumentError("sparse FACE uses its own smoothing and noise estimates; presmooth and PACE tuning are unsupported"))
        return _fit_face_sparse(fd; grid, ncomponents, maxK, mean, eigen_tol, sign_tol, face...)
    end
    isempty(face) || throw(ArgumentError("face settings require method=:face"))
    method in (:auto, :dense, :pace) || throw(ArgumentError("FPCA method must be :auto, :dense, :pace, or :face"))
    method == :dense && throw(ArgumentError("dense FPCA requires complete GridFunctionalData"))
    presmooth === nothing || throw(ArgumentError("presmooth is only supported by dense FPCA; PACE smooths the pooled mean and covariance"))
    _fit_pace(fd; method, grid, ncomponents, maxK, mean, covariance,
        noise_variance, mean_bandwidth, covariance_bandwidth, diagonal_bandwidth, kernel,
        weighting, binning, eigen_tol, sign_tol)
end

"""
    eigenfunctions(f::FPCA)

Return the selected eigenfunctions as an [`FPCBasis`](@ref) for grid/PACE
fits, or an exact linear combination of the input basis for basis-data fits.
Both implement `basismatrix`, `gram`, `penalty`, `domain`, and `nbasis`.
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

Return conditional score covariance matrices of size `K × K × n` for PACE or FACE,
quantifying each curve's remaining score uncertainty given its observations.
These omit uncertainty from estimating the mean, covariance, or noise.
Dense and basis FPCA return `nothing`. Use a copy for independent array storage.
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

Return the measurement-error variance used by the fit. PACE and both FACE
routes floor estimated variance at `0.025` times the maximum fitted pointwise
variance; explicitly supplied PACE variance, including zero, is unchanged.
The raw estimate, floor, variance scale, relative factor and application flag are in
`f.settings`. Dense and basis FPCA return `nothing`; `SmoothedCovariance`
returns its stored nonnegative variance estimate without the FPCA floor.
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
    predict(f::FPCA, newdata::Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData})

Project new dense curves using the training mean, basis, weights, and fitted
preprocessing. Return an `n × K` score matrix. Data must be complete and use
exactly the training grid. This extends `StatsAPI.predict`, as in
MultivariateStats.jl's PCA projection interface.

PACE accepts either representation and returns `(scores, covariances)` as a
named tuple, with an `n × K` matrix and `K × K × n` array. It uses the training
mean, eigenfunctions, full projected covariance up to `maxK`, and noise
variance. Every observation time must lie inside the work grid. Predicting
the training data reproduces [`scores`](@ref) and [`score_covariances`](@ref).

Common-grid FACE returns the same named tuple, requires grid data on exactly the
training grid, and uses the selected-K posterior. New masked curves use their
observed points; training predictions reproduce the stored completed-data
scores. No mean or covariance is refitted.
Sparse FACE accepts either representation within its work grid, evaluates
the fitted splines at observation times, and uses the full positive spline
covariance in its BLUP and selected-score covariance.

Basis FPCA requires `BasisFunctionalData` in the same basis as training. It
returns exact L² inner-product scores against the stored normalized harmonics.
"""
function predict(f::FPCA, fd::GridFunctionalData)
    _fpca_predict(f, fd, f.noise_variance)
end

_fpca_predict(f, fd::GridFunctionalData, ::Real) =
    f.method === :face ? (f.training_data isa IrregularFunctionalData ?
        _face_sparse_predict(f, fd) : _face_predict(f, fd)) : _pace_predict(f, fd)

function _fpca_predict(f, fd::GridFunctionalData, ::Nothing)
    f.method === :basis && throw(ArgumentError("basis FPCA prediction requires BasisFunctionalData in the training basis"))
    length(fd.grid) == length(f.grid) || throw(DimensionMismatch("prediction grid length must match the FPCA training grid"))
    fd.grid == f.grid || throw(ArgumentError("prediction grid must exactly match the FPCA training grid"))
    fd.observed === nothing || throw(ArgumentError("dense FPCA prediction requires complete grid data"))
    values, _ = _fpca_preprocess(fd, _fpca_prediction_method(f))
    projected = (values .- f.mean.values) * (f.weights .* f.basis.values)
    all(isfinite, projected) || throw(ArgumentError("FPCA prediction scores must be finite; rescale the observations"))
    projected
end

function predict(f::FPCA, fd::IrregularFunctionalData)
    if f.method === :face
        f.training_data isa IrregularFunctionalData || throw(ArgumentError("common-grid FACE prediction requires GridFunctionalData on the training grid"))
        return _face_sparse_predict(f, fd)
    end
    f.noise_variance === nothing && throw(ArgumentError("irregular prediction requires a PACE FPCA fit"))
    _pace_predict(f, fd)
end

"""
    reconstruct(f::FPCA, scores=scores(f); grid=f.grid)

Return `GridFunctionalData` containing `mean + scores * eigenfunctions'`.
The finite increasing grid must lie inside the work-grid domain. Both the mean
and eigenfunctions are linearly interpolated for grid/PACE/FACE fits; basis
fits evaluate their exact basis expansions. Subject ids are retained when
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
    mu = _fpca_mean_values(f.mean, grid)
    reconstructed = projected * transpose(functions) .+ mu
    GridFunctionalData(reconstructed, grid; ids=projected === f.scores ? f.ids : nothing)
end

_fpca_mean_values(mu::GridFunctionalData, grid) = evaluate(mu, grid, PiecewiseLinear())
_fpca_mean_values(mu::BasisFunctionalData, grid) = evaluate(mu, grid)

function Base.show(io::IO, f::FPCA)
    print(io, "FPCA(method=", f.method, ", design=", f.design, ", n=", nobs(f),
        ", grid=", length(f.grid), " points, ncomponents=", ncomponents(f),
        ", selection=", f.selection, ", presmooth=", f.presmooth,
        ", sign_rule=", f.sign_rule, ", settings=", _show_summary(f.settings),
        ", dropped=", f.dropped, ", noise_variance=", f.noise_variance, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::FPCA) = show(io, f)

# Large arrays in settings (projected covariance surfaces) are shown by size.
_show_summary(x::AbstractMatrix) = "$(join(size(x), "×")) $(nameof(typeof(x)))"
_show_summary(x::NamedTuple) = map(_show_summary, x)
_show_summary(x) = x
