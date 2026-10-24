"""
    CurveCV(bandwidths; folds=10, rng=nothing)
    CurveCV(; pilot, multipliers=2.0 .^ range(-1, 2; length=10), folds=10, rng=nothing)

Select a pooled-mean, covariance, or noise-diagonal bandwidth by leaving out
whole curves. Candidates must be finite, positive, and distinct. Supply an `AbstractRNG` for shuffled, balanced
K-fold splits; `folds == nobs(fd)` uses deterministic leave-one-curve-out and
needs no RNG. Inputs, including the RNG state, are copied. Each fit starts from
that copied state. Exact score ties select the first candidate.

For the mean or noise diagonal, `pilot` is a regression
`KernelSmoothing.BandwidthSelector`. At fit time its global bandwidth multiplies
`multipliers` to form the candidates. The default ten multipliers range from
0.5 to 4 on a logarithmic grid. The pilot uses all pooled observations
and the fit's weights. Selection records `pilot`, `h_pilot`, `pilot_details`,
`multipliers`, `bandwidths`, `errors`, and `assignments`. The 2D covariance
requires explicit candidates. Pilot rules assume independent observations.
That is asymptotically adequate for sparse designs but ignores within-curve
correlation in dense ones; use `CurveCV` for the final choice.
"""
struct CurveCV{T<:AbstractFloat,R,P}
    bandwidths::Vector{T}
    folds::Int
    rng::R
    pilot::P
    multipliers::Vector{T}
    function CurveCV(bandwidths::AbstractVector{<:Real}; folds=10, rng=nothing)
        Base.require_one_based_indexing(bandwidths)
        isempty(bandwidths) && throw(ArgumentError("CurveCV bandwidths must be nonempty"))
        all(h -> isfinite(h) && h > 0, bandwidths) ||
            throw(ArgumentError("CurveCV bandwidths must be finite and positive"))
        T = _floattype(eltype(bandwidths))
        hs = Vector{T}(bandwidths)
        all(isfinite, hs) && length(unique(hs)) == length(hs) ||
            throw(ArgumentError("CurveCV bandwidths must remain finite and distinct after conversion"))
        folds isa Integer && 2 <= folds <= typemax(Int) ||
            throw(ArgumentError("CurveCV folds must be an integer of at least two"))
        rng === nothing || rng isa AbstractRNG ||
            throw(ArgumentError("CurveCV rng must be an AbstractRNG or nothing"))
        new{T,typeof(rng),Nothing}(hs, Int(folds), deepcopy(rng), nothing, T[])
    end
    function CurveCV(; pilot, multipliers=2.0 .^ range(-1, 2; length=10),
            folds=10, rng=nothing)
        pilot isa KernelSmoothing.BandwidthSelector ||
            throw(ArgumentError("CurveCV pilot must be a KernelSmoothing.BandwidthSelector"))
        validated = try
            CurveCV(multipliers; folds, rng)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError(replace(err.msg, "bandwidths" => "multipliers")))
        end
        T = eltype(validated.bandwidths)
        new{T,typeof(validated.rng),typeof(pilot)}(T[], validated.folds,
            validated.rng, deepcopy(pilot), validated.bandwidths)
    end
end

# Validate polynomial settings before data-dependent selection without inventing
# a smoothing bandwidth: this placeholder is used only for validation.
_initial_bandwidth(h) = h
_initial_bandwidth(::KernelSmoothing.BandwidthSelector) = 1.0
_initial_bandwidth(cv::CurveCV) = cv.pilot === nothing ? first(cv.bandwidths) : 1.0

function _regression_bandwidth(rule::KernelSmoothing.BandwidthSelector, t, y, w, kernel, degree)
    data = KernelSmoothing.SmoothingData(t, y; weights=w)
    hasmethod(KernelSmoothing.selectbandwidth,
        Tuple{typeof(rule),typeof(data),typeof(kernel)}, (:degree,)) ||
        throw(ArgumentError("$rule is density-only or does not support regression selection with degree; use a regression rule or CurveCV"))
    selected = KernelSmoothing.selectbandwidth(rule, data, kernel; degree)
    selected.h isa Real || throw(ArgumentError(
        "$rule returns pointwise bandwidths; supply a rule returning one global bandwidth or CurveCV"))
    h = LocalPolynomial(; bandwidth=selected.h, kernel, degree).bandwidth
    (; rule=deepcopy(rule), bandwidth=h, details=deepcopy(selected.details))
end

function _cv_candidates(cv::CurveCV{T,R,Nothing}, t, y, w, kernel, degree) where {T<:AbstractFloat,R}
    cv, NamedTuple()
end
function _cv_candidates(cv::CurveCV, t, y, w, kernel, degree)
    pilot = _regression_bandwidth(cv.pilot, t, y, w, kernel, degree)
    candidates = CurveCV(cv.multipliers .* pilot.bandwidth; folds=cv.folds, rng=cv.rng)
    candidates, (; pilot=pilot.rule, h_pilot=pilot.bandwidth,
        pilot_details=pilot.details, multipliers=copy(cv.multipliers))
end

function _resolve_bandwidth(h, fd, t, y, w, subjects, kernel, degree; kwargs...)
    LocalPolynomial(; bandwidth=h, kernel, degree).bandwidth, nothing
end
function _resolve_bandwidth(rule::KernelSmoothing.BandwidthSelector,
        fd, t, y, w, subjects, kernel, degree; kwargs...)
    selection = _regression_bandwidth(rule, t, y, w, kernel, degree)
    selection.bandwidth, selection
end
function _resolve_bandwidth(cv::CurveCV, fd, t, y, w, subjects, kernel, degree; kwargs...)
    candidates, pilot = _cv_candidates(cv, t, y, w, kernel, degree)
    selection = _mean_selection(candidates, fd, t, y, w, subjects, kernel, degree; kwargs...)
    selection.bandwidths[argmin(selection.errors)], merge(selection, pilot)
end

"""
    fit(SmoothedMean, fd; grid, bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), degree=1, weighting=:obs)

Fit a local polynomial to all observed values of univariate grid or irregular
data. `:obs` gives each observation weight one; `:subj` gives each observation
in curve i weight `1/N_i`. Stored data weights are not used. Empty curves throw.
The explicit work grid must be finite and strictly increasing with at least
two points. The required `bandwidth` is a positive number, [`CurveCV`](@ref),
or a regression `KernelSmoothing.BandwidthSelector` returning one global
bandwidth. Selectors use the pooled observations and the chosen weights.
They assume independent observations: asymptotically adequate for sparse
designs, where each curve contributes about one point per window, but they
ignore within-curve correlation in dense designs. Use `CurveCV` for the
final choice. Density-only and pointwise bandwidth rules raise `ArgumentError`.

The fit stores grid, values, selected bandwidth, kernel, degree, weighting,
sampling design (`design`), `minobs=0`, `binning=:none`, and `folds` (zero for
a fixed or rule-selected bandwidth). Fixed bandwidths store `selection=nothing`.
A rule stores `rule`, `bandwidth`, and
selector `details` in `selection`. CV stores a named tuple of `bandwidths`,
`errors`, and per-curve `assignments`. CV minimizes total weighted squared
prediction error divided by total held-out weight, at actual observation
times. An undefined fit, including any CV prediction, throws with its point
and a suggestion to increase bandwidth. No observation or candidate is dropped.
Use [`evaluate`](@ref) to interpolate linearly within the stored grid range;
queries outside it throw. No confidence bands are provided.
"""
struct SmoothedMean{T<:AbstractFloat,H<:AbstractFloat,K,S}
    grid::Vector{T}
    values::Vector{T}
    bandwidth::H
    kernel::K
    degree::Int
    weighting::Symbol
    design::Symbol
    minobs::Int
    binning::Symbol
    folds::Int
    selection::S
end

function _mean_pool(fd, weighting)
    T = eltype(fd)
    times, values, weights = T[], T[], T[]
    subjects = Int[]
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values for pooled mean"))
        append!(times, t)
        append!(values, y)
        append!(weights, fill(weighting == :obs ? one(T) : one(T)/length(t), length(t)))
        append!(subjects, fill(i, length(t)))
    end
    times, values, weights, subjects
end

function _mean_predict(t, y, w, points, method; label="pooled mean")
    model = KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, t, y;
        bandwidth=method.bandwidth, kernel=method.kernel, degree=method.degree,
        weights=w, points, minobs=0)
    values = KernelSmoothing.coef(model)
    for (point, value) in zip(points, values)
        isfinite(value) || throw(ArgumentError(
            "$label has an undefined estimate at point $point; use a larger bandwidth"))
    end
    values
end

function _mean_selection(selector::CurveCV, fd, t, y, w, subjects, kernel, degree;
        prediction_times=t, label="pooled mean")
    n = nobs(fd)
    selector.folds <= n || throw(ArgumentError("CurveCV folds must not exceed the number of curves ($n)"))
    assignments = collect(1:n)
    if selector.folds != n
        selector.rng === nothing && throw(ArgumentError("CurveCV rng is required unless folds equals the number of curves"))
        order = randperm(deepcopy(selector.rng), n)
        for (j, i) in enumerate(order)
            assignments[i] = mod1(j, selector.folds)
        end
    end
    errors = map(selector.bandwidths) do h
        method = LocalPolynomial(; bandwidth=h, kernel, degree)
        score = zero(promote_type(eltype(t), typeof(h)))
        for fold in 1:selector.folds
            held = assignments[subjects] .== fold
            train = .!held
            predicted = _mean_predict(t[train], y[train], w[train], prediction_times[held], method; label)
            score += sum(w[held] .* abs2.(y[held] - predicted))
        end
        score / sum(w)
    end
    all(isfinite, errors) || throw(ArgumentError("CurveCV errors are nonfinite; rescale the observations"))
    (; bandwidths=copy(selector.bandwidths), errors, assignments)
end

function fit(::Type{SmoothedMean}, fd::Union{GridFunctionalData,IrregularFunctionalData};
        grid::AbstractVector{<:Real}, bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), degree=1, weighting=:obs)
    weighting in (:obs, :subj) || throw(ArgumentError("weighting must be :obs or :subj"))
    x = _evaluation_points(fd, grid)
    quadrature_weights(x)
    # Validate settings before either fitting or selecting a bandwidth.
    method = LocalPolynomial(; bandwidth=_initial_bandwidth(bandwidth),
        kernel, degree)
    t, y, w, subjects = _mean_pool(fd, weighting)
    h, selection = _resolve_bandwidth(bandwidth, fd, t, y, w, subjects, method.kernel, method.degree)
    chosen = LocalPolynomial(; bandwidth=h, kernel=method.kernel, degree=method.degree)
    values = Vector{eltype(x)}(_mean_predict(t, y, w, x, chosen))
    all(isfinite, values) || throw(ArgumentError("pooled mean values must remain finite after conversion"))
    SmoothedMean(x, values, chosen.bandwidth, chosen.kernel, chosen.degree, weighting,
        sampling_design(fd), 0, :none, bandwidth isa CurveCV ? bandwidth.folds : 0, selection)
end

"""
    evaluate(m::SmoothedMean, t)

Linearly interpolate a pooled mean on its stored grid. Accept a real scalar or
vector of finite query times, all within the closed grid range; throw an
`ArgumentError` otherwise. Vector queries return a vector in query order.
"""
function evaluate(m::SmoothedMean, t::AbstractVector{<:Real})
    Base.require_one_based_indexing(t)
    T = promote_type(eltype(m.grid), _floattype(eltype(t)))
    points = Vector{T}(t)
    all(x -> isfinite(x) && first(m.grid) <= x <= last(m.grid), points) ||
        throw(ArgumentError("mean evaluation times must be finite and inside the stored grid range"))
    _evaluate_curve!(Vector{T}(undef, length(points)), m.grid, m.values, points, PiecewiseLinear())
end
evaluate(m::SmoothedMean, t::Real) = only(evaluate(m, [t]))

function Base.show(io::IO, selector::CurveCV)
    print(io, "CurveCV(")
    if selector.pilot === nothing
        print(io, selector.bandwidths, "; ")
    else
        print(io, "; pilot=", selector.pilot, ", multipliers=", selector.multipliers, ", ")
    end
    print(io, "folds=", selector.folds,
        ", rng=", selector.rng, ")")
end
function Base.show(io::IO, m::SmoothedMean)
    print(io, "SmoothedMean(grid=", m.grid, ", bandwidth=", m.bandwidth,
        ", kernel=", m.kernel, ", degree=", m.degree, ", weighting=", m.weighting,
        ", design=", m.design, ", minobs=", m.minobs, ", binning=", m.binning,
        ", folds=", m.folds, ", selection=", m.selection, ")")
end
