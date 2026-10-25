# Benchmark implementation, not package API. Julia >= 1.10; FDA 0.1.0.
module ScalingBenchmark

using FunctionalDataAnalysis
using LinearAlgebra
using Random
import StatsAPI: fit

const CONFIG_HEADER = ["id", "design", "n", "m", "ratio", "min_points", "max_points", "replicates", "seed"]
const RAW_HEADER = ["replicate", "seed", "route", "status", "wall_seconds", "allocated_bytes",
    "process_peak_rss_bytes", "eigenvalue_relative_error", "eigenfunction_l2_error",
    "sigma2_error", "ncomponents", "reconstruction_error", "grid_start", "grid_end",
    "truth_eigenvalues", "settings"]
const FAILURE_HEADER = ["replicate", "seed", "route", "stage", "error_type", "error"]
const VALUES = [1.0, 0.5, 0.25]
const FUNCTIONS = fourier_eigenfunctions((0.0, 1.0), 3)

routes(config) = config.design == :dense ? (:dense, :face) : (:pace, :face_sparse)

function read_config(path, index)
    lines = readlines(path)
    !isempty(lines) && split(first(lines), ',') == CONFIG_HEADER ||
        throw(ArgumentError("unexpected scaling configuration CSV header"))
    1 <= index < length(lines) || throw(ArgumentError("configuration index is outside the CSV"))
    fields = split(lines[index+1], ',')
    length(fields) == length(CONFIG_HEADER) || throw(ArgumentError("configuration row must have nine columns"))
    id, design, n, m, ratio, lo, hi, replicates, seed = fields
    c = (; id=parse(Int, id), design=Symbol(design), n=parse(Int, n), m=parse(Int, m),
        ratio=parse(Float64, ratio), min_points=parse(Int, lo), max_points=parse(Int, hi),
        replicates=parse(Int, replicates), seed=parse(Int, seed))
    c.id == index || throw(ArgumentError("configuration id must match its one-based row index"))
    c.design in (:dense, :sparse) || throw(ArgumentError("design must be dense or sparse"))
    c.n >= 3 || throw(ArgumentError("n must be at least three"))
    c.m >= (c.design == :dense ? 38 : 10) || throw(ArgumentError("m is too small for the default FACE spline rank"))
    isfinite(c.ratio) && c.ratio > 0 || throw(ArgumentError("noise variance ratio must be finite and positive"))
    if c.design == :dense
        c.min_points == c.max_points == c.m || throw(ArgumentError("dense observation counts must equal m"))
    else
        c.min_points == 2 && c.max_points == 10 || throw(ArgumentError("sparse counts must range from 2 to 10"))
    end
    c.replicates >= 1 || throw(ArgumentError("replicates must be positive"))
    0 <= c.seed <= typemax(Int)-c.replicates || throw(ArgumentError("seed range must be nonnegative and fit in Int"))
    c
end

function simulation(config, replicate)
    sampling = config.design == :dense ?
        RegularSampling(collect(range(0.0, 1.0; length=config.m))) :
        SparseSampling((0.0, 1.0), config.min_points:config.max_points)
    simulated = simulate_kl(MersenneTwister(config.seed+replicate-1), config.n;
        mean=t -> 1+t, eigenfunctions=FUNCTIONS, eigenvalues=VALUES,
        noise_sd=sqrt(config.ratio*sum(VALUES)), sampling)
    grid = config.design == :dense ? copy(simulated.data.grid) :
        collect(range(extrema(simulated.data.times)...; length=config.m))
    (; simulated, grid)
end

function fit_route(route, data, grid)
    route == :dense && return fit(FPCA, data; method=:dense)
    route == :face && return fit(FPCA, data; method=:face)
    route == :pace && return fit(FPCA, data; method=:pace, grid)
    route == :face_sparse && return fit(FPCA, data; method=:face, grid)
    throw(ArgumentError("unknown scaling route $route"))
end

# A 3x3 eigensystem gives the exact population target under the same stored
# quadrature, including restriction to the sparse sample's observed interval.
# Never form a grid-square covariance merely to compute benchmark errors.
function population(grid)
    weights = quadrature_weights(grid)
    phi = [f(t) for t in grid, f in FUNCTIONS]
    factor = phi .* transpose(sqrt.(VALUES))
    system = eigen(Symmetric(transpose(factor)*(weights .* factor)))
    order = sortperm(system.values; rev=true)
    values = system.values[order]
    functions = (factor*system.vectors[:, order]) ./ transpose(sqrt.(values))
    (; weights, phi, values, functions)
end

function accuracy(fitted, prepared, truth, sigma2)
    k = ncomponents(fitted)
    compared = min(k, length(VALUES))
    estimated = zeros(length(VALUES))
    estimated[1:compared] = eigvals(fitted)[1:compared]
    eigenvalue_error = norm(estimated-truth.values)/norm(truth.values)
    functions = basismatrix(eigenfunctions(fitted), prepared.grid)
    distances = zeros(length(VALUES))
    for j in eachindex(distances)
        target = view(truth.functions, :, j)
        if j > k
            distances[j] = sqrt(sum(truth.weights .* abs2.(target)))
        else
            candidate = view(functions, :, j)
            orientation = dot(target, truth.weights .* candidate) < 0 ? -1 : 1
            distances[j] = sqrt(sum(truth.weights .* abs2.(orientation .* candidate-target)))
        end
    end
    latent = prepared.simulated.scores*transpose(truth.phi) .+ transpose(1 .+ prepared.grid)
    reconstructed = reconstruct(fitted; grid=prepared.grid).values
    reconstruction_error = sum(abs2.(reconstructed-latent) .* transpose(truth.weights))/size(latent, 1)
    noise = noise_variance(fitted)
    metrics = (; eigenvalue_relative_error=eigenvalue_error,
        eigenfunction_l2_error=sum(distances)/length(distances),
        sigma2_error=noise === nothing ? nothing : noise-sigma2,
        ncomponents=k, reconstruction_error)
    all(x -> x === nothing || isfinite(x), metrics) || throw(ArgumentError("benchmark accuracy metrics must be finite"))
    metrics
end

# Keep selected tuning values, not full covariance surfaces, in every raw row.
function tuning(fitted)
    s = fitted.settings
    keys = (:maxK, :ncomponents, :eigen_tol, :sign_tol, :quadrature, :score_method,
        :knots, :p, :m, :lambda, :lambda_source, :alpha, :center,
        :mean_bandwidth, :covariance_bandwidth, :diagonal_bandwidth,
        :weighting, :binning, :binning_rule, :covariance_ridge,
        :raw_noise_variance, :noise_floor, :noise_floor_applied,
        :noise_floor_relative, :noise_floor_scale, :noise_floor_scale_source)
    settings = String["method=$(fitted.method)", "design=$(fitted.design)"]
    for key in keys
        hasproperty(s, key) && push!(settings, "$key=$(repr(getproperty(s, key)))")
    end
    if hasproperty(s, :mean)
        mean = s.mean
        if hasproperty(mean, :smoothing) && mean.smoothing !== nothing
            mean = mean.smoothing
        end
        for key in (:source, :knots, :lambda, :lambda_source)
            hasproperty(mean, key) && push!(settings, "mean_$key=$(repr(getproperty(mean, key)))")
        end
    end
    if hasproperty(s, :covariance_settings)
        cov = s.covariance_settings
        push!(settings, "resolved_diagonal_bandwidth=$(cov.diagonal_bandwidth)")
        push!(settings, "kernel=$(repr(cov.kernel))")
    end
    join(settings, ';')
end

fit_failure(err) = err isa Exception && !(err isa Union{InterruptException,OutOfMemoryError})

# Only the first twelve fields of our own raw schema are read here. They are
# quoted labels/numbers with no commas; later settings remain opaque CSV text.
# The Python summary independently validates the complete CSV and all counts.
function aggregates(path, route)
    names = (:wall_seconds, :allocated_bytes, :eigenvalue_relative_error,
        :eigenfunction_l2_error, :sigma2_error, :ncomponents, :reconstruction_error)
    columns = (5, 6, 8, 9, 10, 11, 12)
    totals, counts = zeros(length(names)), zeros(Int, length(names))
    successes, failures, peak = 0, 0, 0
    for line in Iterators.drop(eachline(path), 1)
        fields = strip.(split(line, ','; limit=13)[1:12], '"')
        fields[3] == string(route) || continue
        if fields[4] == "success"
            successes += 1
        else
            failures += 1
        end
        peak = max(peak, parse(Int, fields[7]))
        for (j, column) in enumerate(columns)
            isempty(fields[column]) && continue
            totals[j] += parse(Float64, fields[column])
            counts[j] += 1
        end
    end
    headers = ["$(route)_successes", "$(route)_failures", "$(route)_process_peak_rss_bytes"]
    row = Any[successes, failures, peak]
    for j in eachindex(names)
        append!(headers, ["$(route)_$(names[j])_count", "$(route)_$(names[j])_mean"])
        append!(row, (counts[j], counts[j] == 0 ? nothing : totals[j]/counts[j]))
    end
    headers, row
end

csvcell(::Nothing) = "\"\""
csvcell(x) = "\"" * replace(replace(string(x), '\n' => "\\n", '\r' => "\\r"), "\"" => "\"\"") * "\""
csvrow(io, values) = println(io, join(csvcell.(values), ','))

end
