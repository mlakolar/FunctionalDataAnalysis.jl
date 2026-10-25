# One configuration -> one summary row plus per-replicate/failure companions.
# Julia >= 1.10; FunctionalDataAnalysis 0.1.0; no benchmark-only dependencies.
include("common.jl")
using .Sigma2Benchmark
using LinearAlgebra
import KernelSmoothing
import FunctionalDataAnalysis

function run_configuration(config_path, index, output)
    config = Sigma2Benchmark.read_config(config_path, index)
    endswith(output, ".csv") || throw(ArgumentError("output must end in .csv"))
    raw_path = output[1:end-4]*".replicates.csv"
    failure_path = output[1:end-4]*".failures.csv"
    any(ispath, (output, raw_path, failure_path)) && throw(ArgumentError(
        "output or companion files already exist; use a new output path"))
    mkpath(dirname(output))
    BLAS.set_num_threads(1)
    started = time()
    estimators = Sigma2Benchmark.ESTIMATORS
    successes = zeros(Int, length(estimators))
    means, m2, squared_errors = zeros(5), zeros(5), zeros(5)
    truncated = zeros(Int, 5)
    sigma2 = config.ratio*1.75
    open(raw_path, "w") do rawio
        open(failure_path, "w") do failureio
            Sigma2Benchmark.csvrow(rawio, ["replicate", "seed", "mean_bandwidth", "covariance_bandwidth",
                "mean_cv_errors", "covariance_cv_errors", "fold_assignments",
                [string(e)*suffix for e in estimators for suffix in ("_raw", "_estimate")]...])
            Sigma2Benchmark.csvrow(failureio, ("replicate", "estimator", "error"))
            for replicate in 1:config.replicates
                values = fill(NaN, 5)
                raw_values = fill(NaN, 5)
                mean_h, h = NaN, NaN
                mean_errors, covariance_errors, assignments = "", "", ""
                try
                    fd, truth = Sigma2Benchmark.simulation(config, replicate)
                    truth == sigma2 || error("simulation sigma2 does not match configuration")
                    prepared = Sigma2Benchmark.prepare(config, replicate, fd)
                    mean_h, h = prepared.mean.bandwidth, prepared.h
                    if prepared.selection !== nothing
                        mean_errors = join(prepared.mean_selection.errors, ';')
                        covariance_errors = join(prepared.selection.errors, ';')
                        assignments = join(prepared.selection.assignments, ';')
                    end
                    for (j, estimator) in enumerate(estimators)
                        try
                            result = Sigma2Benchmark.estimate(prepared, estimator)
                            raw_values[j], values[j] = result.raw, result.value
                        catch err
                            err isa ArgumentError || rethrow()
                            Sigma2Benchmark.csvrow(failureio, (replicate, estimator, sprint(showerror, err)))
                        end
                    end
                catch err
                    err isa ArgumentError || rethrow()
                    Sigma2Benchmark.csvrow(failureio, (replicate, "shared_fit", sprint(showerror, err)))
                end
                for j in eachindex(estimators)
                    isfinite(values[j]) || continue
                    successes[j] += 1
                    delta = values[j]-means[j]
                    means[j] += delta/successes[j]
                    m2[j] += delta*(values[j]-means[j])
                    squared_errors[j] += abs2(values[j]-sigma2)
                    truncated[j] += raw_values[j] < 0
                end
                Sigma2Benchmark.csvrow(rawio, (replicate, config.seed+replicate-1, mean_h, h,
                    mean_errors, covariance_errors, assignments,
                    Iterators.flatten(zip(raw_values, values))...))
                flush(rawio)
                flush(failureio)
                println("configuration ", config.id, ": ", replicate, "/", config.replicates)
            end
        end
    end
    headers = [string.(keys(config))..., "sigma2", "julia_version", "fda_version",
        "kernel_smoothing_version", "elapsed_seconds"]
    row = Any[Base.values(config)..., sigma2, VERSION, pkgversion(FunctionalDataAnalysis),
        pkgversion(KernelSmoothing), time()-started]
    for (j, estimator) in enumerate(estimators)
        append!(headers, [string(estimator)*suffix for suffix in
            ("_successes", "_mean", "_m2", "_squared_error_sum", "_truncated")])
        append!(row, (successes[j], successes[j] == 0 ? NaN : means[j], m2[j],
            squared_errors[j], truncated[j]))
    end
    open(output, "w") do io
        Sigma2Benchmark.csvrow(io, headers)
        Sigma2Benchmark.csvrow(io, row)
    end
    println("Wrote ", output, "; successful fits per estimator: ", successes)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || throw(ArgumentError("usage: run.jl CONFIG.csv INDEX OUTPUT.csv"))
    run_configuration(ARGS[1], parse(Int, ARGS[2]), ARGS[3])
end
