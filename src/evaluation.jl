"""
    Linear()

Explicit piecewise-linear interpolation of raw curve observations. Values
outside each curve's observed range are `NaN` during evaluation and missing
during conversion to a grid. A one-point curve is defined only at that point.
"""
struct Linear end

"""
    NoInterpolation()

Exact lookup of raw observations using `==`, with no tolerance. Evaluation
throws at unobserved times. Conversion to a grid instead marks absent
observations missing and requires every source time in the target grid.
"""
struct NoInterpolation end

function _evaluation_points(fd, t::AbstractVector{S}) where {S<:Real}
    Base.require_one_based_indexing(t)
    T = promote_type(eltype(fd), _floattype(S))
    points = Vector{T}(t)
    all(isfinite, points) || throw(ArgumentError("evaluation points must be finite"))
    points
end

function evaluate(fd::BasisFunctionalData, t::AbstractVector{<:Real}; derivative=0)
    fd.coefs * transpose(basismatrix(fd.basis, t; derivative))
end
(fd::BasisFunctionalData)(t::AbstractVector{<:Real}; derivative=0) = evaluate(fd, t; derivative)

evaluate(::Union{GridFunctionalData,IrregularFunctionalData}, ::AbstractVector{<:Real}) =
    throw(ArgumentError("raw data evaluation requires Linear() or NoInterpolation()"))

function _evaluate_curve!(out, times, values, points, ::NoInterpolation)
    for (j, t) in enumerate(points)
        k = searchsortedfirst(times, t)
        k <= length(times) && times[k] == t ||
            throw(ArgumentError("NoInterpolation requires every query time to be observed in each curve"))
        out[j] = values[k]
    end
    out
end

function _evaluate_curve!(out, times, values, points, ::Linear)
    for (j, t) in enumerate(points)
        if isempty(times) || t < first(times) || t > last(times)
            out[j] = NaN
            continue
        end
        k = searchsortedfirst(times, t)
        if k <= length(times) && times[k] == t
            out[j] = values[k]
        else
            left = k - 1
            fraction = (t - times[left]) / (times[k] - times[left])
            out[j] = (1 - fraction) * values[left] + fraction * values[k]
        end
    end
    out
end

function evaluate(fd::Union{GridFunctionalData,IrregularFunctionalData},
        t::AbstractVector{<:Real}, evaluator::Union{Linear,NoInterpolation})
    points = _evaluation_points(fd, t)
    out = Matrix{eltype(points)}(undef, nobs(fd), length(points))
    for i in 1:nobs(fd)
        times, values = fd[i]
        _evaluate_curve!(@view(out[i, :]), times, values, points, evaluator)
    end
    out
end
