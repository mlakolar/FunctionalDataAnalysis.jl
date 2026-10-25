# One cell or route/replicate chunk -> one result row and retained companions.
# Each selected route has its own process, keeping competitor RSS separate.
include("common.jl")
using .ScalingBenchmark
using LinearAlgebra
import FunctionalDataAnalysis
import KernelSmoothing

function work_selection(config, first_replicate, last_replicate, route)
    last_replicate = something(last_replicate, config.replicates)
    1 <= first_replicate <= last_replicate <= config.replicates || throw(ArgumentError(
        "replicate range must satisfy 1 <= FIRST <= LAST <= $(config.replicates)"))
    selected_routes = route === nothing ? ScalingBenchmark.routes(config) : (route,)
    all(r -> r in ScalingBenchmark.routes(config), selected_routes) || throw(ArgumentError(
        "route must match the $(config.design) configuration: $(ScalingBenchmark.routes(config))"))
    first_replicate:last_replicate, selected_routes
end

function run_attempt(route, prepared, truth, sigma2;
    fit_function=ScalingBenchmark.fit_route, metric_function=ScalingBenchmark.accuracy,
    tuning_function=ScalingBenchmark.tuning)
    measured = @timed try
        fit_function(route, prepared.simulated.data, prepared.grid)
    catch err
        ScalingBenchmark.fit_failure(err) || rethrow()
        err
    end
    fitted = measured.value
    status, stage, failure = "success", "fit", nothing
    metrics = (; eigenvalue_relative_error=nothing, eigenfunction_l2_error=nothing,
        sigma2_error=nothing, ncomponents=nothing, reconstruction_error=nothing)
    settings = ""
    if fitted isa Exception
        status, failure = "failure", fitted
    else
        try
            metrics = metric_function(fitted, prepared, truth, sigma2)
            settings = tuning_function(fitted)
        catch err
            ScalingBenchmark.fit_failure(err) || rethrow()
            status, stage, failure = "failure", "metrics", err
        end
    end
    if failure !== nothing
        metrics = map(_ -> nothing, metrics)
        settings = ""
    end
    (; measured, status, stage, failure, metrics, settings)
end

function run_worker(config_path, index, output, route, first_replicate=1, last_replicate=nothing)
    config = ScalingBenchmark.read_config(config_path, index)
    replicates, _ = work_selection(config, first_replicate, last_replicate, route)
    BLAS.set_num_threads(1)
    # Compile on tiny data outside scored replicates. A failed warmup is reported
    # but does not prevent retaining the configured attempts and their failures.
    warm_config = merge(config, (; n=12, m=config.design == :dense ? 40 : 21,
        min_points=config.design == :dense ? 40 : 2,
        max_points=config.design == :dense ? 40 : 10, ratio=0.2, seed=20261025))
    warmup = ScalingBenchmark.simulation(warm_config, 1)
    try
        ScalingBenchmark.fit_route(route, warmup.simulated.data, warmup.grid)
    catch err
        ScalingBenchmark.fit_failure(err) || rethrow()
        @warn "Unscored warmup failed; proceeding to configured replicates" route exception=(err, catch_backtrace())
    end
    warmup = nothing
    GC.gc()
    raw_path = output[1:end-4]*".replicates.csv"
    failure_path = output[1:end-4]*".failures.csv"
    open(raw_path, "a") do rawio
        open(failure_path, "a") do failureio
            for replicate in replicates
                seed = config.seed+replicate-1
                prepared = ScalingBenchmark.simulation(config, replicate)
                truth = ScalingBenchmark.population(prepared.grid)
                GC.gc()
                attempt = run_attempt(route, prepared, truth, config.ratio*sum(ScalingBenchmark.VALUES))
                if attempt.failure !== nothing
                    failure = attempt.failure
                    ScalingBenchmark.csvrow(failureio, (replicate, seed, route, attempt.stage,
                        string(typeof(failure)), sprint(showerror, failure)))
                end
                ScalingBenchmark.csvrow(rawio, (replicate, seed, route, attempt.status, attempt.measured.time,
                    attempt.measured.bytes, Sys.maxrss(), values(attempt.metrics)...,
                    first(prepared.grid), last(prepared.grid), join(truth.values, ';'), attempt.settings))
                flush(rawio)
                flush(failureio)
                println("configuration ", config.id, " ", route, ": ", replicate, "/", config.replicates, " ", attempt.status)
                # No prior fits or reconstruction matrices remain live at the
                # next timed fit. RSS remains a cumulative process high-water mark.
                attempt = prepared = truth = nothing
            end
        end
    end
end

function run_configuration(config_path, index, output, first_replicate=1, last_replicate=nothing, route=nothing)
    config = ScalingBenchmark.read_config(config_path, index)
    replicates, selected_routes = work_selection(config, first_replicate, last_replicate, route)
    endswith(output, ".csv") || throw(ArgumentError("output must end in .csv"))
    raw_path = output[1:end-4]*".replicates.csv"
    failure_path = output[1:end-4]*".failures.csv"
    any(ispath, (output, raw_path, failure_path)) && throw(ArgumentError(
        "output or companion files already exist; use a new output path"))
    project = Base.active_project()
    project === nothing && throw(ArgumentError("activate an instantiated FDA project with --project"))
    mkpath(dirname(output))
    open(io -> ScalingBenchmark.csvrow(io, ScalingBenchmark.RAW_HEADER), raw_path, "w")
    open(io -> ScalingBenchmark.csvrow(io, ScalingBenchmark.FAILURE_HEADER), failure_path, "w")
    started = time()
    for selected_route in selected_routes
        command = `$(Base.julia_cmd()) --project=$(dirname(project)) $(@__FILE__) --worker $config_path $index $output $selected_route $(first(replicates)) $(last(replicates))`
        process = run(ignorestatus(addenv(command, "JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1")))
        success(process) || error("$selected_route worker exited with code $(process.exitcode); companion files retained, no final result row written")
    end
    header = [string.(keys(config))..., "first_replicate", "last_replicate", "routes",
        "sigma2", "julia_version", "fda_version", "kernel_smoothing_version", "elapsed_seconds"]
    row = Any[values(config)..., first(replicates), last(replicates), join(selected_routes, ';'),
        config.ratio*sum(ScalingBenchmark.VALUES), VERSION, pkgversion(FunctionalDataAnalysis),
        pkgversion(KernelSmoothing), time()-started]
    for selected_route in selected_routes
        route_header, route_row = ScalingBenchmark.aggregates(raw_path, selected_route)
        append!(header, route_header)
        append!(row, route_row)
    end
    open(output, "w") do io
        ScalingBenchmark.csvrow(io, header)
        ScalingBenchmark.csvrow(io, row)
    end
    println("Wrote ", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) in (5, 7) && ARGS[1] == "--worker"
        first_replicate = length(ARGS) == 7 ? parse(Int, ARGS[6]) : 1
        last_replicate = length(ARGS) == 7 ? parse(Int, ARGS[7]) : nothing
        run_worker(ARGS[2], parse(Int, ARGS[3]), ARGS[4], Symbol(ARGS[5]), first_replicate, last_replicate)
    else
        length(ARGS) in (3, 5, 6) || throw(ArgumentError(
            "usage: run.jl CONFIG.csv INDEX OUTPUT.csv [FIRST LAST [ROUTE]]"))
        first_replicate = length(ARGS) >= 5 ? parse(Int, ARGS[4]) : 1
        last_replicate = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
        route = length(ARGS) == 6 ? Symbol(ARGS[6]) : nothing
        run_configuration(ARGS[1], parse(Int, ARGS[2]), ARGS[3], first_replicate, last_replicate, route)
    end
end
