# One cell or route/replicate chunk -> one result row and retained companions.
# Each selected route has its own process, keeping competitor RSS separate.
include("common.jl")
using .RegressionStudy
using LinearAlgebra
import FunctionalDataAnalysis
import KernelSmoothing

function work_selection(config, first_replicate, last_replicate, route)
    last_replicate = something(last_replicate, config.replicates)
    1 <= first_replicate <= last_replicate <= config.replicates || throw(ArgumentError(
        "replicate range must satisfy 1 <= FIRST <= LAST <= $(config.replicates)"))
    selected_routes = route === nothing ? RegressionStudy.ROUTES : (route,)
    all(in(RegressionStudy.ROUTES), selected_routes) || throw(ArgumentError(
        "route must be one of $(RegressionStudy.ROUTES)"))
    first_replicate:last_replicate, selected_routes
end

const EMPTY_METRICS = map(_ -> nothing, NamedTuple{RegressionStudy.METRIC_NAMES}(RegressionStudy.METRIC_NAMES))

function run_attempt(route, design, prepared, population;
    fit_function=RegressionStudy.fit_route, metric_function=RegressionStudy.accuracy,
    tuning_function=RegressionStudy.tuning)
    measured = @timed try
        fit_function(route, design, prepared.train, prepared.y)
    catch err
        RegressionStudy.fit_failure(err) || rethrow()
        err
    end
    fitted = measured.value
    status, stage, failure = "success", "fit", nothing
    result = (; metrics=EMPTY_METRICS, lambda=nothing, band_covariance=nothing)
    settings = ""
    if fitted isa Exception
        status, failure = "failure", fitted
    else
        try
            result = metric_function(fitted, prepared, population)
            settings = tuning_function(fitted)
        catch err
            RegressionStudy.fit_failure(err) || rethrow()
            status, stage, failure = "failure", "metrics", err
        end
    end
    if failure !== nothing
        result = (; metrics=EMPTY_METRICS, lambda=nothing, band_covariance=nothing)
        settings = ""
    end
    (; measured, status, stage, failure, result..., settings)
end

function run_worker(config_path, index, output, route, first_replicate=1, last_replicate=nothing)
    config = RegressionStudy.read_config(config_path, index)
    replicates, _ = work_selection(config, first_replicate, last_replicate, route)
    BLAS.set_num_threads(1)
    population = RegressionStudy.truth(config.beta)
    # Compile on tiny data outside scored replicates. A failed warmup is reported
    # but does not prevent retaining the configured attempts and their failures.
    warm_config = merge(config, (; n=30, n_test=5, r2=0.8, seed=20261026))
    warmup = RegressionStudy.simulation(warm_config, 1, population)
    try
        warm_fit = RegressionStudy.fit_route(route, config.design, warmup.train, warmup.y)
        RegressionStudy.accuracy(warm_fit, warmup, population)
    catch err
        RegressionStudy.fit_failure(err) || rethrow()
        @warn "Unscored warmup failed; proceeding to configured replicates" route exception=(err, catch_backtrace())
    end
    warmup = nothing
    GC.gc()
    raw_path = output[1:end-4] * ".replicates.csv"
    failure_path = output[1:end-4] * ".failures.csv"
    open(raw_path, "a") do rawio
        open(failure_path, "a") do failureio
            for replicate in replicates
                seed = config.seed + replicate - 1
                prepared = RegressionStudy.simulation(config, replicate, population)
                GC.gc()
                attempt = run_attempt(route, config.design, prepared, population)
                if attempt.failure !== nothing
                    failure = attempt.failure
                    RegressionStudy.csvrow(failureio, (replicate, seed, route, attempt.stage,
                        string(typeof(failure)), sprint(showerror, failure)))
                end
                RegressionStudy.csvrow(rawio, (replicate, seed, route, attempt.status,
                    attempt.measured.time, attempt.measured.bytes, Sys.maxrss(),
                    values(attempt.metrics)..., attempt.lambda, attempt.band_covariance,
                    first(prepared.grid), last(prepared.grid), population.signal_variance,
                    attempt.settings))
                flush(rawio)
                flush(failureio)
                println("configuration ", config.id, " ", route, ": ", replicate, "/",
                    config.replicates, " ", attempt.status)
                # No prior fit remains live at the next timed fit.
                attempt = prepared = nothing
            end
        end
    end
end

function run_configuration(config_path, index, output, first_replicate=1, last_replicate=nothing, route=nothing)
    config = RegressionStudy.read_config(config_path, index)
    replicates, selected_routes = work_selection(config, first_replicate, last_replicate, route)
    endswith(output, ".csv") || throw(ArgumentError("output must end in .csv"))
    raw_path = output[1:end-4] * ".replicates.csv"
    failure_path = output[1:end-4] * ".failures.csv"
    any(ispath, (output, raw_path, failure_path)) && throw(ArgumentError(
        "output or companion files already exist; use a new output path"))
    project = Base.active_project()
    project === nothing && throw(ArgumentError("activate an instantiated FDA project with --project"))
    mkpath(dirname(output))
    open(io -> RegressionStudy.csvrow(io, RegressionStudy.RAW_HEADER), raw_path, "w")
    open(io -> RegressionStudy.csvrow(io, RegressionStudy.FAILURE_HEADER), failure_path, "w")
    started = time()
    for selected_route in selected_routes
        command = `$(Base.julia_cmd()) --project=$(dirname(project)) $(@__FILE__) --worker $config_path $index $output $selected_route $(first(replicates)) $(last(replicates))`
        process = run(ignorestatus(addenv(command, "JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1")))
        success(process) || error("$selected_route worker exited with code $(process.exitcode); companion files retained, no final result row written")
    end
    population = RegressionStudy.truth(config.beta)
    header = [string.(keys(config))..., "first_replicate", "last_replicate", "routes",
        "signal_variance", "beta_norm", "unidentified_beta_norm",
        "julia_version", "fda_version", "kernel_smoothing_version", "elapsed_seconds"]
    row = Any[values(config)..., first(replicates), last(replicates), join(selected_routes, ';'),
        population.signal_variance, population.beta_norm, population.unidentified_norm,
        VERSION, pkgversion(FunctionalDataAnalysis), pkgversion(KernelSmoothing), time() - started]
    for selected_route in selected_routes
        route_header, route_row = RegressionStudy.aggregates(raw_path, selected_route)
        append!(header, route_header)
        append!(row, route_row)
    end
    open(output, "w") do io
        RegressionStudy.csvrow(io, header)
        RegressionStudy.csvrow(io, row)
    end
    println("Wrote ", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 7 && ARGS[1] == "--worker"
        run_worker(ARGS[2], parse(Int, ARGS[3]), ARGS[4], Symbol(ARGS[5]), parse(Int, ARGS[6]), parse(Int, ARGS[7]))
    else
        length(ARGS) in (3, 5, 6) || throw(ArgumentError(
            "usage: run.jl CONFIG.csv INDEX OUTPUT.csv [FIRST LAST [ROUTE]]"))
        first_replicate = length(ARGS) >= 5 ? parse(Int, ARGS[4]) : 1
        last_replicate = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
        route = length(ARGS) == 6 ? Symbol(ARGS[6]) : nothing
        run_configuration(ARGS[1], parse(Int, ARGS[2]), ARGS[3], first_replicate, last_replicate, route)
    end
end
