# CPU-time estimate of a full work list from a probe run (configs.jl --probe).
# Run it on the machine that ran the probe.
# Usage: julia --project=. estimate.jl CONFIG.csv PROBE_CONFIG.csv PROBE_DIR [--block-size N]
isdefined(@__MODULE__, :OutlierStudy) || include("common.jl")
isdefined(@__MODULE__, :read_configurations) || include("work_units.jl")
isdefined(@__MODULE__, :read_rows) || include("summarize.jl")
using .OutlierStudy
using Printf
using Statistics

cell_key(config) = (config.study, config.design, config.n)
timing_key(config, route) = (cell_key(config)..., route)

"""
Seconds per replicate that a worker spends outside the timed fit: the simulation
and the full garbage collection before each fit (run.jl), measured here.
"""
function replicate_overhead(config; replicates=2)
    OutlierStudy.simulation(config, 1)
    GC.gc()
    mean(@elapsed((OutlierStudy.simulation(config, r); GC.gc())) for r in 1:replicates)
end

"""
From a complete probe run: mean fit seconds per (study, design, n, route), the
per-replicate overhead per (study, design, n) from `replicate_seconds`, and the
mean per-worker startup (process start, package loading and warmup), which is a
probe cell's elapsed time less its fit time and per-replicate overhead, per route.
"""
function probe_timings(probe_configs, directory; replicate_seconds=replicate_overhead)
    chunks = result_chunks(probe_configs, directory)
    fits = Dict{Tuple{Symbol,Symbol,Int,Symbol},Vector{Float64}}()
    per_replicate = Dict{Tuple{Symbol,Symbol,Int},Float64}()
    startups = Float64[]
    for config in probe_configs
        cell = chunks[config.id]
        length(cell) == 1 || study_error("$directory: probe cell $(config.id) needs exactly one whole-cell result")
        path, result = only(cell)
        selected = Symbol.(split(result["routes"], ';'))
        Set(selected) == Set(OutlierStudy.routes(config)) || study_error("$path: probe cell must run every route")
        records = read_rows(path[1:end-4] * ".replicates.csv", OutlierStudy.RAW_HEADER)
        length(records) == length(selected) * config.replicates || study_error("$path: incomplete probe records")
        fit_total = 0.0
        for record in records
            seconds = finite_field(record["wall_seconds"], "$path: wall_seconds"; minimum=0)
            push!(get!(fits, timing_key(config, Symbol(record["route"])), Float64[]), seconds)
            fit_total += seconds
        end
        overhead = get!(() -> replicate_seconds(config), per_replicate, cell_key(config))
        elapsed = finite_field(result["elapsed_seconds"], "$path: elapsed_seconds"; minimum=0)
        push!(startups, max(elapsed - fit_total - length(records) * overhead, 0.0) / length(selected))
    end
    (; fit_seconds=Dict(k => mean(v) for (k, v) in fits), per_replicate, startup=mean(startups))
end

"""Estimated CPU seconds of every work unit of `configs` with the given block size."""
function estimate(configs, probe_configs, directory; block_size=200, replicate_seconds=replicate_overhead)
    timings = probe_timings(probe_configs, directory; replicate_seconds)
    units = work_units(configs, block_size)
    seconds = map(units) do u
        config = configs[u.cell]
        key = timing_key(config, u.route)
        haskey(timings.fit_seconds, key) || study_error("probe has no timing for $key")
        timings.startup + (u.last - u.first + 1) * (timings.fit_seconds[key] + timings.per_replicate[cell_key(config)])
    end
    (; units, seconds, timings...)
end

function write_estimate(io, configs, result)
    total = sum(result.seconds)
    @printf(io, "Work units: %d; estimated CPU time %.2f hours (%.0f s), of which worker startup %.2f hours\n",
        length(result.units), total / 3600, total, length(result.units) * result.startup / 3600)
    @printf(io, "Per-worker startup (process, packages, warmup): %.1f s; longest unit %.0f s\n",
        result.startup, maximum(result.seconds))
    println(io, "\nPer-replicate overhead outside the fit (simulation, garbage collection), seconds:")
    for key in sort!(collect(keys(result.per_replicate)))
        @printf(io, "  %-10s %-6s n=%-4d %8.3f\n", key..., result.per_replicate[key])
    end
    println(io, "\nMean fit seconds from the probe:")
    for key in sort!(collect(keys(result.fit_seconds)))
        @printf(io, "  %-10s %-6s n=%-4d %-14s %8.3f\n", key..., result.fit_seconds[key])
    end
    println(io, "\nEstimated CPU hours by study, design and n:")
    groups = Dict{Tuple{Symbol,Symbol,Int},Float64}()
    for (u, s) in zip(result.units, result.seconds)
        key = cell_key(configs[u.cell])
        groups[key] = get(groups, key, 0.0) + s
    end
    for key in sort!(collect(keys(groups)))
        @printf(io, "  %-10s %-6s n=%-4d %7.2f\n", key..., groups[key] / 3600)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    usage = "usage: estimate.jl CONFIG.csv PROBE_CONFIG.csv PROBE_DIR [--block-size N]"
    length(ARGS) in (3, 5) && (length(ARGS) == 3 || ARGS[4] == "--block-size") || throw(ArgumentError(usage))
    configs = read_configurations(ARGS[1])
    block_size = length(ARGS) == 5 ? positive_integer(ARGS[5]) : 200
    write_estimate(stdout, configs, estimate(configs, read_configurations(ARGS[2]), ARGS[3]; block_size))
end
