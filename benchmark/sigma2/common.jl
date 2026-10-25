# Benchmark implementation, not package API. Julia >= 1.10; FDA 0.1.0.
module Sigma2Benchmark

using FunctionalDataAnalysis
using LinearAlgebra
using Random
import KernelSmoothing
import StatsAPI

const FDA = FunctionalDataAnalysis
const ESTIMATORS = (:surface_middle_half, :surface_full, :rotated_middle_half,
    :rotated_full, :projected_fgm)

# Exact duplicate grouping, never time snapping. Reuse the package's weighted
# product aggregator, including within-cell RSS, and its local smoother.
function collapse_pairs(pool, indices=eachindex(pool.z))
    grouped = (; x=Float64[], y=Float64[], z=Float64[], weights=Float64[],
        scatter=Float64[], subjects=Int[])
    cells = Dict{Tuple{Float64,Float64},Int}()
    for i in indices
        FDA._covariance_add_pair!(grouped, cells, true, pool.x[i], pool.y[i],
            pool.z[i], pool.weights[i], 1)
        grouped.scatter[cells[(pool.x[i], pool.y[i])]] += pool.scatter[i]
    end
    grouped
end

# Same whole-curve CurveCV loss as the package, using sufficient statistics
# at exact duplicate coordinates within each training/held-out fold. On dense
# data this avoids predicting the same pair once per subject. No smoother is
# implemented here; every prediction uses KernelSmoothing.local_polynomial.
function grouped_cv(selector::CurveCV, fd, pool, kernel, kind)
    assignments = collect(1:nobs(fd))
    if selector.folds != nobs(fd)
        for (j, i) in enumerate(randperm(deepcopy(selector.rng), nobs(fd)))
            assignments[i] = mod1(j, selector.folds)
        end
    end
    errors = zeros(length(selector.bandwidths))
    for fold in 1:selector.folds
        held_mask = assignments[pool.subjects] .== fold
        train = collapse_pairs(pool, findall(.!held_mask))
        held = collapse_pairs(pool, findall(held_mask))
        for (j, h) in enumerate(selector.bandwidths)
            predicted = cv_predictions(train, held, h, kernel, kind)
            errors[j] += sum(held.weights .* abs2.(held.z - predicted) + held.scatter)
        end
    end
    errors ./= sum(pool.weights)
    all(isfinite, errors) || throw(ArgumentError("benchmark covariance CV errors must be finite"))
    (; bandwidths=copy(selector.bandwidths), errors, assignments)
end

cv_predictions(train, held, h, kernel, ::Val{:covariance}) =
    KernelSmoothing.local_polynomial((train.x, train.y), train.z, hcat(held.x, held.y);
        bandwidth=h, kernel, weights=train.weights)
cv_predictions(train, held, h, kernel, ::Val{:mean}) =
    FDA._mean_predict(train.x, train.z, train.weights, held.x,
        LocalPolynomial(; bandwidth=h, kernel))
covariance_cv(selector, fd, pool, kernel) = grouped_cv(selector, fd, pool, kernel, Val(:covariance))

function mean_cv(selector, fd, weighting, kernel)
    t, y, w, subjects = FDA._mean_pool(fd, weighting)
    pool = (; x=t, y=t, z=y, weights=w, scatter=zeros(length(y)), subjects)
    grouped_cv(selector, fd, pool, kernel, Val(:mean))
end

function simulation(config, replicate)
    rng = MersenneTwister(config.seed + replicate - 1)
    sampling = config.design == :sparse ? SparseSampling((0., 1.), 2:6) :
        RegularSampling(collect(range(0., 1.; length=50)))
    eigenvalues = [1., .5, .25]
    sigma2 = config.ratio*sum(eigenvalues) # variance ratio; domain has length one
    data = simulate_kl(rng, config.n; mean=t -> 1+t,
        eigenfunctions=fourier_eigenfunctions((0., 1.), 3), eigenvalues,
        noise_sd=sqrt(sigma2), sampling).data
    data, sigma2
end

function prepare(config, replicate, fd)
    a, b = FDA._pace_observed_range(fd)
    grid = collect(range(a, b; length=51))
    kernel = KernelSmoothing.GaussianKernel()
    fold_seed = config.seed + replicate - 1 + 1000000
    mean_selection = if config.bandwidth == :default
        nothing
    else
        mean_cv(CurveCV((b-a).*[.025, .05, .1]; folds=5,
            rng=MersenneTwister(fold_seed)), fd, config.weighting, kernel)
    end
    mean_bandwidth = mean_selection === nothing ? FDA._pace_default_mean_bandwidth(a, b) :
        mean_selection.bandwidths[argmin(mean_selection.errors)]
    mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=mean_bandwidth,
        kernel, weighting=config.weighting)
    rawpool = FDA._covariance_pool(fd, mean, config.weighting, nothing)
    selection = if config.bandwidth == :default
        nothing
    else
        covariance_cv(CurveCV((b-a).*[.05, .1, .2]; folds=5,
            rng=MersenneTwister(fold_seed)), fd, rawpool, kernel)
    end
    h = selection === nothing ? FDA._pace_default_covariance_bandwidth(a, b) :
        selection.bandwidths[argmin(selection.errors)]
    pool = merge(rawpool, collapse_pairs(rawpool))
    surface = FDA._local_linear_surface(pool, grid, grid, h, kernel)
    surface = (surface + surface')/2
    diagonal = FDA._mean_predict(rawpool.times, rawpool.squares, rawpool.diagonal_weights,
        grid, LocalPolynomial(; bandwidth=h, kernel); label="noise variance diagonal smoother")
    (; grid, kernel, mean, pool, surface, diagonal, h, selection, mean_selection)
end

function estimate(prepared, estimator)
    (; grid, kernel, mean, pool, surface, diagonal, h) = prepared
    if estimator == :projected_fgm
        # Benchmark-only FGM-inspired reference: use the package eigen step,
        # all positive components, and FGM's relative cutoff, not an FPCA K cap.
        system = FDA._fpca_eigen(surface, quadrature_weights(grid), mean.values;
            max_k=length(grid), eigen_tol=100eps(Float64), sign_tol=1e-12)
        surface = (system.eigenfunctions .* system.eigenvalues') * system.eigenfunctions'
        surface = (surface + surface')/2
        # FGM bsplinekit 6f0eac3, src/doubly.jl:223-226: arithmetic mean on
        # these index bounds, rather than a trapezoidal middle-half integral.
        q = length(grid)
        interior = max(1, ceil(Int, q/4)):min(q, floor(Int, 3q/4))
        raw = sum(diagonal[interior] - diag(surface)[interior])/length(interior)
        isfinite(raw) || throw(ArgumentError("projected reference variance must be finite"))
        return (; raw, value=max(0., raw))
    end
    method = estimator in (:rotated_middle_half, :rotated_full) ? :rotated : :surface_diagonal
    interval = estimator in (:surface_full, :rotated_full) ? :full : :middle_half
    details = FDA._covariance_noise_integral(pool, grid, surface, diagonal, h,
        kernel, method, interval)
    (; raw=details.raw_variance, value=max(0., details.raw_variance))
end

function read_config(path, index)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("configuration CSV is empty"))
    split(first(lines), ',') == ["id", "design", "n", "ratio", "weighting", "bandwidth", "replicates", "seed"] ||
        throw(ArgumentError("unexpected configuration CSV header"))
    1 <= index < length(lines) || throw(ArgumentError("configuration index is outside the CSV"))
    fields = split(lines[index+1], ',')
    length(fields) == 8 || throw(ArgumentError("configuration row must have eight columns"))
    id, design, n, ratio, weighting, bandwidth, replicates, seed = fields
    c = (; id=parse(Int, id), design=Symbol(design), n=parse(Int, n), ratio=parse(Float64, ratio),
        weighting=Symbol(weighting), bandwidth=Symbol(bandwidth),
        replicates=parse(Int, replicates), seed=parse(Int, seed))
    c.id == index || throw(ArgumentError("configuration id must match its one-based row index"))
    c.design in (:sparse, :dense) || throw(ArgumentError("design must be sparse or dense"))
    c.n in (100, 400) || throw(ArgumentError("n must be 100 or 400"))
    c.ratio in (.05, .2, .5) || throw(ArgumentError("ratio must be .05, .2, or .5"))
    c.weighting in (:obs, :subj) || throw(ArgumentError("weighting must be obs or subj"))
    c.bandwidth in (:default, :cv) || throw(ArgumentError("bandwidth must be default or cv"))
    c.replicates >= 2 || throw(ArgumentError("replicates must be at least two"))
    c.seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    c
end

csvcell(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
csvrow(io, values) = println(io, join(csvcell.(values), ','))

end
