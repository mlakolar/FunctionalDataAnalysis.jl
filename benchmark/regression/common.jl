# Study implementation, not package API. Julia >= 1.10; FDA 0.1.0.
module RegressionStudy

using FunctionalDataAnalysis
using LinearAlgebra
using Random
import StatsAPI: fit, predict

const CONFIG_HEADER = ["id", "design", "beta", "n", "r2", "n_test", "replicates", "seed"]
const ROUTES = (:fpc, :fpc_bic, :pfr)
const BETAS = (:leading, :trailing, :smooth)
const METRIC_NAMES = (:beta_relative_l2_error, :prediction_relative_error,
    :pointwise_coverage, :whole_curve_coverage, :band_mean_width, :ncomponents, :term_dof)
const RAW_HEADER = ["replicate", "seed", "route", "status", "wall_seconds", "allocated_bytes",
    "process_peak_rss_bytes", string.(METRIC_NAMES)..., "lambda", "band_covariance",
    "grid_start", "grid_end", "signal_variance", "settings"]
const FAILURE_HEADER = ["replicate", "seed", "route", "stage", "error_type", "error"]

# Covariates: nine orthonormal Fourier functions on [0,1] (constant, then
# sine/cosine pairs up to frequency 4) with eigenvalues 1/k^2. In population,
# FVE(0.99) keeps the first eight; the ninth carries 0.8% of the variance.
const FUNCTIONS = fourier_eigenfunctions((0.0, 1.0), 9)
const VALUES = [1 / k^2 for k in 1:9]
const MEAN = t -> 1 + t
const COVARIATE_NOISE_RATIO = 0.05
const COVARIATE_NOISE_SD = sqrt(COVARIATE_NOISE_RATIO * sum(VALUES))
const DENSE_GRID = collect(range(0.0, 1.0; length=101))
const SPARSE_POINTS = 2:10
const ALPHA = 0.05

# Reiss and Ogden (2007) style smooth coefficient, not built from the FPCs.
smooth_beta(t) = exp(-(t - 0.3)^2 / (2 * 0.1^2))

function beta_function(design::Symbol)
    design == :leading && return t -> FUNCTIONS[1](t) + FUNCTIONS[2](t)
    design == :trailing && return t -> FUNCTIONS[9](t)
    design == :smooth && return smooth_beta
    throw(ArgumentError("unknown coefficient design $design"))
end

# Composite trapezoidal integration on a fine uniform grid; for these smooth
# integrands its error is below 1e-8, far under the Monte Carlo error.
const FINE_GRID = collect(range(0.0, 1.0; length=20001))
const FINE_WEIGHTS = quadrature_weights(FINE_GRID)
fine_integral(f) = sum(FINE_WEIGHTS .* f.(FINE_GRID))

"""Exact population quantities for one coefficient design."""
function truth(design::Symbol)
    beta = beta_function(design)
    projection = [fine_integral(t -> FUNCTIONS[k](t) * beta(t)) for k in eachindex(FUNCTIONS)]
    norm2 = fine_integral(t -> beta(t)^2)
    (; design, beta, projection, offset=fine_integral(t -> MEAN(t) * beta(t)),
        signal_variance=sum(VALUES .* abs2.(projection)), beta_norm=sqrt(norm2),
        unidentified_norm=sqrt(max(norm2 - sum(abs2, projection), 0.0)))
end

function read_config(path, index)
    lines = readlines(path)
    !isempty(lines) && split(first(lines), ',') == CONFIG_HEADER ||
        throw(ArgumentError("unexpected regression configuration CSV header"))
    1 <= index < length(lines) || throw(ArgumentError("configuration index is outside the CSV"))
    fields = split(lines[index+1], ',')
    length(fields) == length(CONFIG_HEADER) || throw(ArgumentError("configuration row must have eight columns"))
    id, design, beta, n, r2, n_test, replicates, seed = fields
    c = (; id=parse(Int, id), design=Symbol(design), beta=Symbol(beta), n=parse(Int, n),
        r2=parse(Float64, r2), n_test=parse(Int, n_test), replicates=parse(Int, replicates),
        seed=parse(Int, seed))
    c.id == index || throw(ArgumentError("configuration id must match its one-based row index"))
    c.design in (:dense, :sparse) || throw(ArgumentError("design must be dense or sparse"))
    c.beta in BETAS || throw(ArgumentError("beta must be one of $BETAS"))
    c.n >= 20 || throw(ArgumentError("n must be at least 20"))
    0 < c.r2 < 1 || throw(ArgumentError("r2 must lie strictly between 0 and 1"))
    c.n_test >= 1 || throw(ArgumentError("n_test must be positive"))
    c.replicates >= 1 || throw(ArgumentError("replicates must be positive"))
    0 <= c.seed <= typemax(Int) - c.replicates || throw(ArgumentError("seed range must be nonnegative and fit in Int"))
    c
end

"""Training sample, held-out subjects and responses for one replicate."""
function simulation(config, replicate, population=truth(config.beta))
    rng = MersenneTwister(config.seed + replicate - 1)
    draw(n, sampling) = simulate_kl(rng, n; mean=MEAN, eigenfunctions=FUNCTIONS,
        eigenvalues=VALUES, noise_sd=COVARIATE_NOISE_SD, sampling)
    if config.design == :dense
        train = draw(config.n, RegularSampling(DENSE_GRID))
        test = draw(config.n_test, RegularSampling(DENSE_GRID))
        grid = copy(DENSE_GRID)
    else
        train = draw(config.n, SparseSampling((0.0, 1.0), SPARSE_POINTS))
        # Sparse fits (and PACE prediction) cover only the observed training
        # range, so held-out times are drawn there and beta is scored there.
        observed = extrema(train.data.times)
        test = draw(config.n_test, SparseSampling(observed, SPARSE_POINTS))
        grid = collect(range(observed...; length=101))
    end
    noise_sd = sqrt(population.signal_variance * (1 - config.r2) / config.r2)
    y = population.offset .+ train.scores * population.projection .+ noise_sd .* randn(rng, config.n)
    test_mean = population.offset .+ test.scores * population.projection
    (; train=train.data, test=test.data, y, test_mean, grid)
end

function fit_route(route, design, data, y)
    route == :fpc && return fit(FPCRegression, data, y)
    route == :fpc_bic && return fit(FPCRegression, data, y; selection=BIC())
    route == :pfr && return design == :dense ? fit(PenalizedFunctionalRegression, data, y) :
        fit(PenalizedFunctionalRegression, data, y; presmooth=:fpca)
    throw(ArgumentError("unknown regression route $route"))
end

term_dof(f::FPCRegression) = only(f.counts)
term_dof(f) = only(f.edf_terms)
fitted_components(f::FPCRegression) = only(f.counts)
fitted_components(f) = nothing
fitted_lambda(f::FPCRegression) = nothing
fitted_lambda(f) = only(f.lambda)

function accuracy(fitted, prepared, population)
    band = only(coefficient_bands(fitted; grid=prepared.grid, alpha=ALPHA))
    target = population.beta.(prepared.grid)
    weights = quadrature_weights(prepared.grid)
    error = sqrt(sum(weights .* abs2.(band.estimate - target)) / sum(weights .* abs2.(target)))
    covered = band.lower .<= target .<= band.upper
    predictions = predict(fitted, prepared.test)
    prediction_error = sum(abs2, predictions - prepared.test_mean) /
        (length(predictions) * population.signal_variance)
    metrics = (; beta_relative_l2_error=error, prediction_relative_error=prediction_error,
        pointwise_coverage=sum(covered) / length(covered),
        # Joint coverage of the pointwise band over the grid; neither method
        # offers a simultaneous band, so this is not a simultaneous coverage claim.
        whole_curve_coverage=all(covered) ? 1 : 0,
        band_mean_width=sum(weights .* (band.upper - band.lower)) / sum(weights),
        ncomponents=fitted_components(fitted), term_dof=term_dof(fitted))
    all(x -> x === nothing || isfinite(x), metrics) || throw(ArgumentError("study metrics must be finite"))
    (; metrics, lambda=fitted_lambda(fitted), band_covariance=string(band.settings.covariance))
end

# Selected tuning values for every retained replicate.
function tuning(f::FPCRegression)
    m = only(f.fpca)
    s = m.settings
    settings = String["ncomponents=$(only(f.counts))", "selection=$(f.selection.rule)",
        "selection_cap=$(only(f.selection.cap))", "fpca_method=$(m.method)",
        "fpca_method_source=$(only(f.settings.fpca_method_source))",
        "fpca_knots=$(something(only(f.settings.fpca_knots), "none"))",
        "fpca_design=$(m.design)", "covariance=$(f.settings.covariance)"]
    for key in (:ncomponents, :maxK, :mean_bandwidth, :covariance_bandwidth, :raw_noise_variance,
        :noise_floor_applied)
        hasproperty(s, key) && push!(settings, "fpca_$key=$(repr(getproperty(s, key)))")
    end
    join(settings, ';')
end

function tuning(f)
    s = f.settings
    join(String["criterion=$(s.criterion)", "lambda=$(repr(only(f.lambda)))",
        "lambda_status=$(only(s.lambda_status))", "edf=$(repr(only(f.edf_terms)))",
        "k=$(only(s.k))", "integration=$(only(s.integration))", "presmooth=$(repr(only(s.presmooth)))",
        "corrected_covariance=$(s.corrected_covariance)"], ';')
end

fit_failure(err) = err isa Exception && !(err isa Union{InterruptException,OutOfMemoryError})

# Only our own numeric raw fields are read here (no commas inside them).
# The Python summary independently validates the complete CSV and all counts.
function aggregates(path, route)
    names = (:wall_seconds, :allocated_bytes, METRIC_NAMES...)
    columns = (5, 6, (8:7+length(METRIC_NAMES))...)
    totals, counts = zeros(length(names)), zeros(Int, length(names))
    successes, failures, peak = 0, 0, 0
    for line in Iterators.drop(eachline(path), 1)
        fields = strip.(split(line, ','; limit=length(RAW_HEADER))[1:last(columns)], '"')
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
        append!(row, (counts[j], counts[j] == 0 ? nothing : totals[j] / counts[j]))
    end
    headers, row
end

csvcell(::Nothing) = "\"\""
csvcell(x) = "\"" * replace(replace(string(x), '\n' => "\\n", '\r' => "\\r"), "\"" => "\"\"") * "\""
csvrow(io, values) = println(io, join(csvcell.(values), ','))

end
