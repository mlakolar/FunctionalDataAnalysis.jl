"""
    fit(SmoothedCovariance, fd, mean; grid, bandwidth, diagonal_bandwidth=nothing,
        kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:obs, binning=nothing,
        noise_method=:rotated, noise_interval=:full)

Smooth all ordered within-curve off-diagonal residual products by bivariate
local linear regression and symmetrize. The required `bandwidth` is positive
or a [`CurveCV`](@ref) with explicit candidates. KernelSmoothing selectors and
pilot-based CurveCV are univariate and cannot select this 2D bandwidth.
`:obs` gives each pair weight one; `:subj` gives weight `1/(N_i*(N_i-1))`.
Every curve must have at least two observations. Stored data weights are unused.
The supplied mean (`SmoothedMean` or one complete `GridFunctionalData` curve,
including a multivariate FPCA mean) is interpolated at original observation
times and held fixed through CV. Its grid must cover those times, and a
`SmoothedMean` must have `derivative=0`.

Optional integer `binning >= 2` snaps times to the nearest of that many equally
spaced points over the observed domain (midpoint ties go upward). Products
still exclude only identical observation indices; multiplicities and original
pair weights are retained. Pairs in each curve and bin cell are aggregated by
summing weights and taking the weighted mean product. Within-cell squared
errors are retained for raw-product CV scores. Residuals are computed before snapping.

`surface` stores the symmetric covariance and `weights` its trapezoidal grid
weights. `noise_variance` estimates measurement error by integrating the
smoothed raw squares minus a latent covariance diagonal, and truncating at zero.
The defaults `noise_method=:rotated, noise_interval=:full` were chosen by the
issue #17 benchmark (`benchmark/sigma2/results/summary.md`, 48 designs × 500
replicates on USC Discovery): the rotated diagonal over the full interval had
the lowest MSE in 41 of 48 cells and removed most of the upward bias of the
surface diagonal.
`noise_method=:surface_diagonal` uses the unrotated surface diagonal (the
milestone 2 estimator); `noise_interval=:middle_half` integrates over the
middle half of the domain as fdapace does, while `:full` integrates over the
whole domain (Li and Hsing, 2010). The surface method divides by the
integration interval's length and interpolates its exact boundaries.

`noise_method=:rotated` follows fdapace 0.6.0 `PC_CovE` and
`RotateLwls2DV2` (Yao, Müller and Wang, 2005). Rotate pairs by 45 degrees;
fit columns `(1, across^2, along)` with the product kernel and the selected
covariance bandwidth in both rotated axes. The raw-square bandwidth remains
`diagonal_bandwidth`. Select existing work-grid points inside the middle half
or full **observed** domain, with fdapace's floating-point boundary buffer;
do not insert endpoints. Divide by the nominal interval length, even if the
retained grid spans less. At least two retained points are required. Singular
windows throw rather than using fdapace's one-point fallback. Subject weighting
and binning retain the package conventions above. The covariance surface is
unaffected. See the [Discovery results](@ref sigma2-discovery-results) for the
default-selection study.
By default (`diagonal_bandwidth=nothing`) the diagonal uses the covariance
bandwidth, after any CV selection, as fdapace does; theory only requires the
diagonal bandwidth to be of no larger order than the covariance bandwidth (Li
and Hsing, 2010, Corollary 3.5). To choose it separately, pass a positive
number, a `CurveCV` (including pilot-based candidates), or a regression
`KernelSmoothing.BandwidthSelector` returning one global bandwidth; it is then
selected independently of the covariance bandwidth.
`noise_details.diagonal_source` is `:covariance` or `:user`. Diagonal CV predicts held-out raw
squared residuals at their original times, leaving whole curves out with the
mean CV fold and weighting rules. With binning, diagonal training times are
snapped but held-out queries stay at original times. The supplied mean stays
fixed. The diagonal uses local linear degree one and observation weights one or `1/N_i`.
Rules use pooled raw squared residuals, training times (snapped with binning),
and the same observation weights. They assume independent observations, which is
asymptotically adequate for sparse designs but ignores within-curve correlation
in dense ones; use `CurveCV` for the final choice. Density-only and pointwise rules throw.
`noise_details.selection` stores the rule, bandwidth, and selector details,
or the CV candidates, errors, assignments, and pilot information when used.
`noise_details` records its bandwidth, fold count, selection scores and assignments,
raw estimate, truncation flag, method, interval, domain, nominal integration
length, boundary rule, integration grid, fitted squares, latent diagonal on
that integration grid, and rotation settings (`applied` indicates their use).
All settings and
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

function _covariance_mean_values(mean::SmoothedMean, t)
    _require_value_mean(mean)
    evaluate(mean, t)
end
function _covariance_mean_values(mean::GridFunctionalData, t)
    nobs(mean) == 1 && mean.observed === nothing || throw(ArgumentError(
        "a covariance mean must contain one complete grid curve"))
    all(v -> first(mean.grid) <= v <= last(mean.grid), t) || throw(ArgumentError(
        "the mean grid must cover every observation time"))
    vec(evaluate(mean, t, PiecewiseLinear()))
end

function _covariance_snap(t, binning)
    binning === nothing && return t
    a, b = extrema(t)
    b > a || throw(ArgumentError("binning requires a nondegenerate observed domain"))
    knots = collect(range(a, b; length=binning))
    [knots[clamp(floor(Int, (v-a)/(b-a)*(binning-1) + 0.5)+1, 1, binning)] for v in t]
end

# Shared weight-preserving aggregation for covariance and cross-covariance.
function _covariance_add_pair!(pool, cells, binning, s, t, product, weight, subject)
    isfinite(product) || throw(ArgumentError(
        "covariance raw products must be finite; rescale the observations"))
    cell = (s, t)
    index = binning === nothing ? 0 : get(cells, cell, 0)
    if index == 0
        push!(pool.x, s); push!(pool.y, t); push!(pool.z, product)
        push!(pool.weights, weight); push!(pool.scatter, zero(product))
        push!(pool.subjects, subject)
        binning === nothing || (cells[cell] = length(pool.z))
    else
        total = pool.weights[index] + weight
        delta = product - pool.z[index]
        pool.z[index] += (weight/total)*delta
        pool.scatter[index] += weight*delta*(product-pool.z[index])
        pool.weights[index] = total
    end
    nothing
end

function _covariance_pool(fd, mean, weighting, binning)
    t, y, w, subjects = _mean_pool(fd, weighting)
    residual = y - _covariance_mean_values(mean, t)
    all(isfinite, residual) || throw(ArgumentError("covariance residuals must be finite"))
    original_times = t
    t = _covariance_snap(t, binning)
    T = eltype(residual)
    x, yy, z, pw, scatter = T[], T[], T[], T[], T[]
    ps = Int[]
    pool = (; x, y=yy, z, weights=pw, scatter, subjects=ps)
    for i in 1:nobs(fd)
        ix = findall(==(i), subjects)
        n = length(ix)
        n >= 2 || throw(ArgumentError("curve $i needs at least two observations for covariance"))
        cells = Dict{Tuple{T,T},Int}()
        for j in ix, k in ix
            j == k && continue
            product = residual[j]*residual[k]
            weight = weighting == :obs ? one(T) : one(T)/(n*(n-1))
            _covariance_add_pair!(pool, cells, binning, t[j], t[k], product, weight, i)
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
    # Leaving out a fold of curves is leaving out a group of pairs. The within-cell
    # scatter of binned products adds the same constant to every candidate.
    hs = Float64.(selector.bandwidths)
    cv = KernelSmoothing.select_bandwidth(KernelSmoothing.CrossValidation(), (pool.x, pool.y), pool.z;
        kernel, weights=pool.weights, groups=assignments[pool.subjects], bandwidths=hs)
    scatter = sum(pool.scatter) / sum(pool.weights)
    T = promote_type(eltype(pool.z), eltype(selector.bandwidths))
    errors = T[cv.details.criterion[searchsortedfirst(cv.details.grid, h)] + scatter for h in hs]
    for (h, e) in zip(selector.bandwidths, errors)
        isfinite(e) || throw(ArgumentError(
            "covariance has a singular or undefined local linear fit with curves held out at bandwidth $h; use larger CurveCV bandwidths"))
    end
    (; bandwidths=copy(selector.bandwidths), errors, assignments)
end

function fit(::Type{SmoothedCovariance}, fd::Union{GridFunctionalData,IrregularFunctionalData},
        mean::Union{SmoothedMean,GridFunctionalData}; grid::AbstractVector{<:Real}, bandwidth, diagonal_bandwidth=nothing,
        kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:obs, binning=nothing,
        noise_method=:rotated, noise_interval=:full)
    mean isa SmoothedMean && _require_value_mean(mean)
    weighting in (:obs, :subj) || throw(ArgumentError("weighting must be :obs or :subj"))
    noise_method isa Symbol && noise_method in (:surface_diagonal, :rotated) || throw(ArgumentError(
        "noise_method must be :surface_diagonal or :rotated"))
    noise_interval isa Symbol && noise_interval in (:middle_half, :full) || throw(ArgumentError(
        "noise_interval must be :middle_half or :full"))
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
    diagonal_bandwidth === nothing ||
        LocalPolynomial(; bandwidth=_initial_bandwidth(diagonal_bandwidth), kernel=method.kernel)
    pool = _covariance_pool(fd, mean, weighting, binning)
    selection = bandwidth isa CurveCV ? _covariance_selection(bandwidth, fd, pool, method.kernel) : nothing
    h = selection === nothing ? method.bandwidth : selection.bandwidths[argmin(selection.errors)]
    diagonal_h, diagonal_selection = if diagonal_bandwidth === nothing
        h, nothing
    else
        _resolve_bandwidth(diagonal_bandwidth, fd, pool.times, pool.squares,
            pool.diagonal_weights, pool.diagonal_subjects, method.kernel, 1;
            prediction_times=pool.original_times, label="noise variance diagonal smoother")
    end
    T = eltype(grid)
    surface = T.(_local_linear_surface(pool, grid, grid, h, method.kernel))
    surface = (surface + transpose(surface))/2
    diagonal = T.(_mean_predict(pool.times, pool.squares, pool.diagonal_weights, grid,
        LocalPolynomial(; bandwidth=diagonal_h, kernel=method.kernel); label="noise variance diagonal smoother"))
    integration = _covariance_noise_integral(pool, grid, surface, diagonal, h,
        method.kernel, noise_method, noise_interval)
    raw = integration.raw_variance
    all(isfinite, surface) && all(isfinite, diagonal) && isfinite(raw) ||
        throw(ArgumentError("covariance and noise variance must remain finite after conversion"))
    details = (; bandwidth=diagonal_h, degree=1, minobs=0,
        folds=diagonal_bandwidth isa CurveCV ? diagonal_bandwidth.folds : 0,
        selection=diagonal_selection, diagonal_source=diagonal_bandwidth === nothing ? :covariance : :user,
        raw_variance=raw, truncated=raw < 0, diagonal,
        noise_method, noise_interval, integration.domain, integration.bounds,
        integration.nominal_length, integration.boundary_rule, integration.integration_grid,
        integration.latent_diagonal,
        rotation=(; applied=noise_method == :rotated, angle=T(pi/4), bandwidth=h,
            across_degree=2, along_degree=1))
    SmoothedCovariance(grid, surface, weights, h, diagonal_h, method.kernel, weighting, binning, 1,
        sampling_design(fd), bandwidth isa CurveCV ? bandwidth.folds : 0,
        selection, max(zero(T), raw), details)
end

# Shared by the fit and the benchmark's integrated estimators. The
# surface estimator preserves the original arithmetic and clipped trapezoids.
function _covariance_noise_integral(pool, grid::AbstractVector{T}, surface, diagonal, h, kernel,
        noise_method, noise_interval) where {T<:AbstractFloat}
    rotated = noise_method == :rotated
    a, b = rotated ? extrema(pool.original_times) : (first(grid), last(grid))
    lower, upper = noise_interval == :middle_half ? (a+(b-a)/4, b-(b-a)/4) : (a, b)
    fraction = noise_interval == :middle_half ? T(0.5) : one(T)
    nominal_length = (b-a)*fraction
    isfinite(nominal_length) && nominal_length > 0 || throw(ArgumentError(
        "noise integration requires a finite positive domain length"))
    # Keep both branches concrete on Julia 1.10 even for runtime method choices.
    local integration_grid::Vector{T}, latent_diagonal::Vector{T}, integrand::Vector{T}
    integration_grid, latent_diagonal, integrand = if rotated
        buffer = 10eps(T)*max(abs(a), abs(b))
        keep = (grid .> lower-buffer) .& (grid .< upper+buffer)
        points = grid[keep]
        length(points) >= 2 || throw(ArgumentError(
            "rotated noise integration requires at least two work-grid points inside noise_interval"))
        latent = Vector{T}(_rotated_covariance_diagonal(pool, points, h, kernel))
        points, latent, diagonal[keep] - latent
    else
        points = vcat(lower, grid[lower .< grid .< upper], upper)
        latent = _evaluate_curve!(similar(points), grid, diag(surface), points, PiecewiseLinear())
        difference = diagonal - diag(surface)
        values = _evaluate_curve!(similar(points), grid, difference, points, PiecewiseLinear())
        points, latent, values
    end
    integral = dot(quadrature_weights(integration_grid), integrand)
    raw_variance::T = noise_interval == :middle_half ? 2integral/(b-a) : integral/(b-a)
    isfinite(raw_variance) && all(isfinite, latent_diagonal) || throw(ArgumentError(
        "noise variance diagonal and integral must be finite"))
    (; raw_variance, domain=(T(a), T(b)), bounds=(T(lower), T(upper)),
        nominal_length=T(nominal_length), integration_grid, latent_diagonal,
        boundary_rule=rotated ? :retained_grid : :interpolated_endpoints)
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
