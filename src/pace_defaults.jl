# Defaults from fdapace 0.6.0, SetOptions, GetBinNum and GCVLwls*.
function _pace_observed_range(fd)
    a, b = typemax(eltype(fd)), typemin(eltype(fd))
    for i in 1:nobs(fd)
        times, _ = fd[i]
        isempty(times) && throw(ArgumentError("curve $i has no observed values for PACE"))
        a = min(a, first(times))
        b = max(b, last(times))
    end
    isfinite(b-a) && b > a || throw(ArgumentError(
        "PACE requires a finite positive observed time range"))
    a, b
end

# fdapace methodBwMu/methodBwCov = "Default": userBwMu = 0.05 and userBwCov = 0.10
# times the observed range.
_pace_default_mean_bandwidth(a, b) = 0.05*(b-a)
_pace_default_covariance_bandwidth(a, b) = 0.10*(b-a)

_pace_default_grid(fd) = collect(range(_pace_observed_range(fd)...; length=51))

# fdapace IsRegular uses coverage of the pooled observed-time union. This
# classification controls its candidate and bin-count rules, not our route.
function _pace_reference_design(fd)
    times = Set{eltype(fd)}()
    total = 0
    for i in 1:nobs(fd)
        t, _ = fd[i]
        union!(times, t)
        total += length(t)
    end
    coverage = total / length(times) / nobs(fd)
    if coverage == 1
        return length(times) < 8 ? :Sparse : :Dense
    end
    coverage > 0.8 ? :DenseWithMV : :Sparse
end

function _pace_binning(fd, requested)
    design = _pace_reference_design(fd)
    counts = sort(nobs_per_curve(fd))
    n = length(counts)
    statistic = design == :Sparse ? :median : :maximum
    m = if statistic == :median
        isodd(n) ? counts[div(n+1, 2)] : counts[div(n, 2)]/2 + counts[div(n, 2)+1]/2
    else
        last(counts)
    end
    # A fixed metadata type keeps fit inference independent of parity/design.
    rule = (; design, summary_count=Float64(m), count_statistic=statistic, n)
    if requested === :auto
        m <= 20 && return nothing, rule
        bins = m > 400 ? 400 : nothing
        if n > 5000
            mstar = max(20, 400 + 19*(5000-n)/2250)
            bins = mstar < m ? ceil(Int, mstar) : nothing
        end
        return bins, rule
    end
    requested === nothing || (_integer_in(requested, 2)) ||
        throw(ArgumentError("binning must be :auto, nothing, or an integer of at least two"))
    requested === nothing ? (nothing, rule) : (Int(requested), rule)
end

function _pace_geometric_candidates(start, stop)
    isfinite(start) && start > 0 && isfinite(stop) && stop > 0 ||
        throw(ArgumentError("PACE bandwidth candidates require finite positive endpoints"))
    q = (stop/start)^(1/9)
    candidates = [start*q^j for j in 0:9]
    all(isfinite, candidates) || throw(ArgumentError("PACE bandwidth candidates must be finite; rescale the times"))
    sort!(candidates)
end

"""
    pace_bandwidth_candidates(fd; grid=nothing, kernel=KernelSmoothing.GaussianKernel())

Return `(mean, covariance)` vectors of ten geometric bandwidth candidates
from fdapace 0.6.0's `GCVLwls1D1` and `GCVLwls2DV2` (local linear degree one).
`grid=nothing` uses 51 equally spaced points over the observed range, as in
[`FPCA`](@ref); a supplied grid must cover every observation. The covariance
range is `sqrt(2)` times the observed range, as in fdapace, and its starting
bandwidth is multiplied by 0.2 for the default Gaussian kernel. Supply the
same `kernel` to this helper and the fit when overriding that default.

The mean start uses the largest lag-three unique-time gap for fdapace's Sparse
classification, twice the lag-two gap for DenseWithMV, and 1.5 times the
lag-two gap for Dense. Its `IsRegular` classification uses coverage of the
pooled time union: full coverage with at least eight unique times is Dense;
full coverage with fewer times is Sparse; otherwise coverage above 80% is
DenseWithMV, and lower coverage is Sparse. This does not change FPCA's route.
The sparse start is multiplied by 2.5, after reducing its
gap by 0.75 if larger than a quarter of the observed range. The covariance
start is the worst distance to three other unique within-curve time pairs,
including diagonal pairs, measured in the maximum coordinate norm (`BwNN`).
Insufficient unique times or pairs throw. Equal endpoints return repeated
candidates exactly as fdapace does; use `unique` before constructing `CurveCV`
in that degenerate case.

This constructs candidates only: [`CurveCV`](@ref) holds out whole curves,
whereas fdapace covariance GCV/CV uses different losses and raw-pair folds.
For example, `mean_bandwidth=CurveCV(pace_bandwidth_candidates(fd).mean;
folds=nobs(fd))` requests leave-one-curve-out selection. Covariance candidates
cost `O(P^2)` time and `O(P)` memory for `P` unique time pairs. A covariance
`CurveCV` with `H` candidates costs `O(H*P^2)` plus `O(H*F*P)` fold setup for
`F` folds; it can be much more expensive than a fixed-bandwidth fit.
"""
function pace_bandwidth_candidates(fd::Union{GridFunctionalData,IrregularFunctionalData};
        grid=nothing, kernel=KernelSmoothing.GaussianKernel())
    kernel isa KernelSmoothing.AbstractKernel ||
        throw(ArgumentError("kernel must be a KernelSmoothing.AbstractKernel"))
    a, b = _pace_observed_range(fd)
    workgrid = grid === nothing ? _pace_default_grid(fd) : _pace_input_grid(grid, "PACE work")
    workgrid = _evaluation_points(fd, workgrid)
    _pace_observations(fd, workgrid)
    times = sort!(unique!(reduce(vcat, [fd[i][1] for i in 1:nobs(fd)])))
    design = _pace_reference_design(fd)
    lag = design == :Sparse ? 3 : 2
    length(times) > lag || throw(ArgumentError(
        "PACE bandwidth candidates need at least $(lag+1) unique observation times"))
    r = b-a
    gap = maximum(times[(lag+1):end] - times[1:(end-lag)])
    start = if design == :Sparse
        2.5 * (gap > r/4 ? 0.75gap : gap)
    else
        (design == :DenseWithMV ? 2 : 1.5) * gap
    end
    mean = _pace_geometric_candidates(min(start, r), r/4)
    pairs = Set{Tuple{eltype(fd),eltype(fd)}}()
    for i in 1:nobs(fd)
        t, _ = fd[i]
        for s in t, u in t
            push!(pairs, (s, u))
        end
    end
    length(pairs) >= 4 || throw(ArgumentError(
        "PACE covariance candidates need at least four unique time pairs"))
    points = collect(pairs)
    # Keep only the three smallest non-self distances, avoiding O(P^2) storage.
    minimum_bandwidth = zero(eltype(fd))
    for p in points
        nearest = (oftype(r, Inf), oftype(r, Inf), oftype(r, Inf))
        for q in points
            p == q && continue
            distance = max(abs(p[1]-q[1]), abs(p[2]-q[2]))
            if distance < nearest[1]
                nearest = (distance, nearest[1], nearest[2])
            elseif distance < nearest[2]
                nearest = (nearest[1], distance, nearest[2])
            elseif distance < nearest[3]
                nearest = (nearest[1], nearest[2], distance)
            end
        end
        minimum_bandwidth = max(minimum_bandwidth, nearest[3])
    end
    radius = sqrt(2)*r
    start = min(minimum_bandwidth * (kernel isa KernelSmoothing.GaussianKernel ? 0.2 : 1), radius/4)
    stop = start < radius/4 ? radius/4 : radius/2
    covariance = _pace_geometric_candidates(start, stop)
    (; mean, covariance)
end
