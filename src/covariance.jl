"""
    fit(SmoothedCovariance, fd, mean::SmoothedMean; grid, bandwidth, diagonal_bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:obs, binning=nothing)

Smooth all ordered within-curve off-diagonal residual products by bivariate
local linear regression and symmetrize. The required `bandwidth` is positive
or a [`CurveCV`](@ref) with explicit candidates. KernelSmoothing selectors and
pilot-based CurveCV are univariate and cannot select this 2D bandwidth.
`:obs` gives each pair weight one; `:subj` gives weight `1/(N_i*(N_i-1))`.
Every curve must have at least two observations. Stored data weights are unused.
The supplied mean is interpolated at original observation times and held fixed
through CV. Its grid must cover those times.

Optional integer `binning >= 2` snaps times to the nearest of that many equally
spaced points over the observed domain (midpoint ties go upward). Products
still exclude only identical observation indices; multiplicities and original
pair weights are retained. Pairs in each curve and bin cell are aggregated by
summing weights and taking the weighted mean product. Within-cell squared
errors are retained for raw-product CV scores. Residuals are computed before snapping.

`surface` stores the symmetric covariance and `weights` its trapezoidal grid
weights. `noise_variance` estimates measurement error by integrating the
smoothed raw squares minus the surface diagonal over the middle half of the
work-grid domain, multiplying by `2/domain_length`, and truncating at zero.
The required `diagonal_bandwidth` is a positive number, a `CurveCV` (including
pilot-based candidates), or a regression `KernelSmoothing.BandwidthSelector`
returning one global bandwidth, selected
independently of the covariance bandwidth. Diagonal CV predicts held-out raw
squared residuals at their original times, leaving whole curves out with the
mean CV fold and weighting rules. With binning, diagonal training times are
snapped but held-out queries stay at original times. The supplied mean stays
fixed. The diagonal uses local linear degree one and observation weights one or `1/N_i`.
Rules use pooled raw squared residuals, training times (snapped with binning),
and the same observation weights. They assume independent observations, ignore
within-curve correlation, and therefore tend to undersmooth repeated-measures
data; use `CurveCV` for the final choice. Density-only and pointwise rules throw.
`noise_details.selection` stores the rule, bandwidth, and selector details,
or the CV candidates, errors, assignments, and pilot information when used.
`noise_details` records its bandwidth, fold count, selection scores and assignments,
raw estimate, truncation flag, integration bounds and grid, and fitted squares.
Boundary values for integration are linearly interpolated. All settings and
CV scores are retained and shown. No positive-semidefinite projection is made.
"""
struct SmoothedCovariance{T<:AbstractFloat,H,D,K,S,N,B}
    grid::Vector{T}
    surface::Matrix{T}
    weights::Vector{T}
    bandwidth::H
    diagonal_bandwidth::D
    kernel::K
    weighting::Symbol
    binning::B
    degree::Int
    design::Symbol
    folds::Int
    selection::S
    noise_variance::T
    noise_details::N
end

function _covariance_pool(fd, mean, weighting, binning)
    t, y, w, subjects = _mean_pool(fd, weighting)
    residual = y - evaluate(mean, t)
    all(isfinite, residual) || throw(ArgumentError("covariance residuals must be finite"))
    original_times = t
    if binning !== nothing
        a, b = extrema(t)
        b > a || throw(ArgumentError("binning requires a nondegenerate observed domain"))
        knots = collect(range(a, b; length=binning))
        t = [knots[clamp(floor(Int, (v-a)/(b-a)*(binning-1) + 0.5)+1, 1, binning)] for v in t]
    end
    T = eltype(residual)
    x, yy, z, pw, scatter = T[], T[], T[], T[], T[]
    ps = Int[]
    for i in 1:nobs(fd)
        ix = findall(==(i), subjects)
        n = length(ix)
        n >= 2 || throw(ArgumentError("curve $i needs at least two observations for covariance"))
        cells = Dict{Tuple{T,T},Int}()
        for j in ix, k in ix
            j == k && continue
            product = residual[j]*residual[k]
            isfinite(product) || throw(ArgumentError("covariance raw products must be finite; rescale the observations"))
            weight = weighting == :obs ? one(T) : one(T)/(n*(n-1))
            cell = (t[j], t[k])
            index = binning === nothing ? 0 : get(cells, cell, 0)
            if index == 0
                push!(x, t[j]); push!(yy, t[k]); push!(z, product)
                push!(pw, weight); push!(scatter, zero(T)); push!(ps, i)
                binning === nothing || (cells[cell] = length(z))
            else
                # Weighted online mean and centered sum of squares. The latter
                # keeps held-out raw-product loss exact after aggregation.
                total = pw[index] + weight
                delta = product - z[index]
                z[index] += (weight/total)*delta
                scatter[index] += weight*delta*(product-z[index])
                pw[index] = total
            end
        end
    end
    squares = residual.^2
    all(isfinite, z) && all(isfinite, scatter) && all(isfinite, squares) ||
        throw(ArgumentError("covariance raw products must be finite; rescale the observations"))
    (; x, y=yy, z, weights=pw, scatter, subjects=ps, times=t, original_times,
        squares, diagonal_weights=w, diagonal_subjects=subjects)
end

function _covariance_selection(selector::CurveCV, fd, pool, kernel)
    n = nobs(fd)
    selector.folds <= n || throw(ArgumentError("CurveCV folds must not exceed the number of curves ($n)"))
    assignments = collect(1:n)
    if selector.folds != n
        selector.rng === nothing && throw(ArgumentError("CurveCV rng is required unless folds equals the number of curves"))
        for (j, i) in enumerate(randperm(deepcopy(selector.rng), n))
            assignments[i] = mod1(j, selector.folds)
        end
    end
    errors = map(selector.bandwidths) do h
        score = zero(promote_type(eltype(pool.z), typeof(h)))
        for fold in 1:selector.folds
            held = assignments[pool.subjects] .== fold
            train = .!held
            x, y, z, w = pool.x[train], pool.y[train], pool.z[train], pool.weights[train]
            for i in findall(held)
                predicted = _local_linear2d(x, y, z, w, pool.x[i], pool.y[i], h, kernel)
                score += pool.weights[i] * abs2(pool.z[i] - predicted) + pool.scatter[i]
            end
        end
        score / sum(pool.weights)
    end
    all(isfinite, errors) || throw(ArgumentError("CurveCV covariance errors are nonfinite; rescale observations"))
    (; bandwidths=copy(selector.bandwidths), errors, assignments)
end

function fit(::Type{SmoothedCovariance}, fd::Union{GridFunctionalData,IrregularFunctionalData},
        mean::SmoothedMean; grid::AbstractVector{<:Real}, bandwidth, diagonal_bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:obs, binning=nothing)
    weighting in (:obs, :subj) || throw(ArgumentError("weighting must be :obs or :subj"))
    binning === nothing || (binning isa Integer && 2 <= binning <= typemax(Int)) ||
        throw(ArgumentError("binning must be nothing or an integer of at least two"))
    binning = binning === nothing ? nothing : Int(binning)
    grid = _evaluation_points(fd, grid)
    weights = quadrature_weights(grid)
    bandwidth isa KernelSmoothing.BandwidthSelector && throw(ArgumentError(
        "KernelSmoothing bandwidth selectors are univariate; use CurveCV with explicit candidates for the 2D covariance bandwidth"))
    bandwidth isa CurveCV && bandwidth.pilot !== nothing && throw(ArgumentError(
        "pilot CurveCV is univariate; the 2D covariance bandwidth requires CurveCV with explicit candidates"))
    method = LocalPolynomial(; bandwidth=_initial_bandwidth(bandwidth), kernel)
    LocalPolynomial(; bandwidth=_initial_bandwidth(diagonal_bandwidth), kernel=method.kernel)
    pool = _covariance_pool(fd, mean, weighting, binning)
    diagonal_h, diagonal_selection =
        _resolve_bandwidth(diagonal_bandwidth, fd, pool.times, pool.squares,
            pool.diagonal_weights, pool.diagonal_subjects, method.kernel, 1;
            prediction_times=pool.original_times, label="noise variance diagonal smoother")
    selection = bandwidth isa CurveCV ? _covariance_selection(bandwidth, fd, pool, method.kernel) : nothing
    h = selection === nothing ? method.bandwidth : selection.bandwidths[argmin(selection.errors)]
    T = eltype(grid)
    surface = T[_local_linear2d(pool.x, pool.y, pool.z, pool.weights, s, t, h, method.kernel) for s in grid, t in grid]
    surface = (surface + transpose(surface))/2
    diagonal = T.(_mean_predict(pool.times, pool.squares, pool.diagonal_weights, grid,
        LocalPolynomial(; bandwidth=diagonal_h, kernel=method.kernel); label="noise variance diagonal smoother"))
    a, b = first(grid), last(grid)
    lower, upper = a + (b-a)/4, b - (b-a)/4
    integration_grid = vcat(lower, grid[lower .< grid .< upper], upper)
    difference = diagonal - diag(surface)
    integrand = _evaluate_curve!(similar(integration_grid), grid, difference, integration_grid, PiecewiseLinear())
    raw = T(2 * dot(quadrature_weights(integration_grid), integrand)/(b-a))
    all(isfinite, surface) && all(isfinite, diagonal) && isfinite(raw) ||
        throw(ArgumentError("covariance and noise variance must remain finite after conversion"))
    details = (; bandwidth=diagonal_h, degree=1, minobs=0,
        folds=diagonal_bandwidth isa CurveCV ? diagonal_bandwidth.folds : 0,
        selection=diagonal_selection, raw_variance=raw,
        truncated=raw < 0, bounds=(lower, upper), integration_grid, diagonal)
    SmoothedCovariance(grid, surface, weights, h, diagonal_h, method.kernel, weighting, binning, 1,
        sampling_design(fd), bandwidth isa CurveCV ? bandwidth.folds : 0,
        selection, max(zero(T), raw), details)
end

"""
    evaluate(c::SmoothedCovariance, s::Real, t::Real)

Bilinearly interpolate the stored surface at finite coordinates inside its
closed grid range. Queries outside the grid throw `ArgumentError`.
"""
function evaluate(c::SmoothedCovariance, s::Real, t::Real)
    g = c.grid
    all(v -> isfinite(v) && first(g) <= v <= last(g), (s, t)) ||
        throw(ArgumentError("covariance evaluation times must be finite and inside the stored grid range"))
    i, j = min(searchsortedlast(g, s), length(g)-1), min(searchsortedlast(g, t), length(g)-1)
    u, v = (s-g[i])/(g[i+1]-g[i]), (t-g[j])/(g[j+1]-g[j])
    (1-u)*(1-v)*c.surface[i,j] + u*(1-v)*c.surface[i+1,j] +
        (1-u)*v*c.surface[i,j+1] + u*v*c.surface[i+1,j+1]
end

function Base.show(io::IO, c::SmoothedCovariance)
    print(io, "SmoothedCovariance(grid=", c.grid, ", bandwidth=", c.bandwidth,
        ", diagonal_bandwidth=", c.diagonal_bandwidth, ", kernel=", c.kernel,
        ", weighting=", c.weighting, ", binning=", c.binning,
        ", degree=", c.degree, ", design=", c.design, ", folds=", c.folds,
        ", selection=", c.selection, ", noise_variance=", c.noise_variance,
        ", noise_details=", c.noise_details, ")")
end
