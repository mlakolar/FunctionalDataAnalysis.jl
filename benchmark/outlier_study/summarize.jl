# Merge complete whole-cell or chunked study results from retained replicates.
# Usage: julia --project=. summarize.jl CONFIG.csv RESULTS_DIR OUTPUT_PREFIX
isdefined(@__MODULE__, :OutlierStudy) || include("common.jl")
isdefined(@__MODULE__, :read_configurations) || include("work_units.jl")
using .OutlierStudy
using Printf
using Statistics

const RESOURCE_METRICS = ("wall_seconds", "allocated_bytes", "process_peak_rss_bytes")
const QUALITY_METRICS = string.(OutlierStudy.METRIC_NAMES)
const OUTLIER_METRICS = ("true_positive_rate", "false_positive_rate", "flagged")
const CLUSTER_METRICS = ("adjusted_rand_index",)
const AGGREGATE_METRICS = ("wall_seconds", "allocated_bytes", QUALITY_METRICS...)
const VERSION_FIELDS = ("julia_version", "fda_version", "kernel_smoothing_version")
const CHUNK_FIELDS = ("first_replicate", "last_replicate", "routes")

study_error(message) = throw(ArgumentError(message))

"""Rows of a file written by `csvrow`, as dictionaries keyed by its unique header."""
function read_rows(path, required=())
    lines = readlines(path)
    isempty(lines) && study_error("$path: missing header")
    header = OutlierStudy.parse_csvrow(first(lines))
    allunique(header) || study_error("$path: duplicate column names")
    issubset(required, header) || study_error("$path: missing required columns")
    map(lines[2:end]) do line
        cells = OutlierStudy.parse_csvrow(line)
        length(cells) == length(header) || study_error("$path: malformed CSV row")
        Dict(zip(header, cells))
    end
end

function integer_field(value, label; minimum=0)
    number = tryparse(Int, value)
    number !== nothing && string(number) == value && number >= minimum ||
        study_error("$label: expected an integer >= $minimum, got $(repr(value))")
    number
end

function finite_field(value, label; minimum=-Inf, maximum=Inf)
    number = tryparse(Float64, value)
    number !== nothing && isfinite(number) && minimum <= number <= maximum ||
        study_error("$label: invalid numeric value $(repr(value))")
    number
end

configuration_text(x::Nothing) = ""
configuration_text(x) = string(x)

route_set(config) = Set(string.(OutlierStudy.routes(config)))

"""Status of a successful fit from its settings, or `missing` when it has none."""
function fit_status(settings)
    for item in split(settings, ';')
        startswith(item, "status=") && return item[8:end]
        item == "converged=true" && return "converged"
        item == "converged=false" && return "maxiter"
    end
    missing
end

function validate_aggregates(result, path, route, successes, failures, metrics)
    for (name, expected) in (("successes", successes), ("failures", failures))
        key = "$(route)_$name"
        integer_field(get(result, key, ""), "$path: $key") == expected ||
            study_error("$path: $key differs from retained records")
    end
    key = "$(route)_process_peak_rss_bytes"
    finite_field(get(result, key, ""), "$path: $key"; minimum=0) == maximum(metrics["process_peak_rss_bytes"]) ||
        study_error("$path: $key differs from retained records")
    for metric in AGGREGATE_METRICS
        values = metrics[metric]
        key = "$(route)_$(metric)_count"
        integer_field(get(result, key, ""), "$path: $key") == length(values) ||
            study_error("$path: $key differs from retained records")
        key = "$(route)_$(metric)_mean"
        if isempty(values)
            get(result, key, nothing) == "" || study_error("$path: $key must be blank when its count is zero")
        else
            isapprox(finite_field(get(result, key, ""), "$path: $key"), mean(values); rtol=1e-12, atol=1e-12) ||
                study_error("$path: $key differs from retained records")
        end
    end
end

function chunk_scope(result, path, config)
    issubset(CHUNK_FIELDS, keys(result)) || study_error("$path: incomplete chunk metadata")
    lo = integer_field(result["first_replicate"], "$path: first_replicate"; minimum=1)
    hi = integer_field(result["last_replicate"], "$path: last_replicate"; minimum=1)
    lo <= hi <= config.replicates || study_error("$path: replicate range is outside the configuration")
    selected = split(result["routes"], ';')
    !isempty(selected) && allunique(selected) && issubset(selected, route_set(config)) ||
        study_error("$path: invalid chunk routes $(repr(result["routes"]))")
    for key in keys(result)
        # Longest matching prefix, so that no route can claim another's columns.
        owners = [r for r in string.(OutlierStudy.ALL_ROUTES) if startswith(key, r * "_")]
        isempty(owners) && continue
        owner = owners[argmax(length.(owners))]
        owner in selected || study_error("$path: aggregates for unselected route $owner")
    end
    lo, hi, String.(selected)
end

"""Finalized result rows, grouped by their declared cell id."""
function result_chunks(configs, directory)
    grouped = Dict(config.id => Tuple{String,Dict{String,String}}[] for config in configs)
    required = (OutlierStudy.CONFIG_HEADER..., CHUNK_FIELDS..., "elapsed_seconds", VERSION_FIELDS...)
    for name in sort(readdir(directory))
        startswith(name, "cell-") && endswith(name, ".csv") || continue
        (endswith(name, ".replicates.csv") || endswith(name, ".failures.csv")) && continue
        path = joinpath(directory, name)
        rows = read_rows(path, required)
        length(rows) == 1 || study_error("$path: expected exactly one result row")
        id = integer_field(only(rows)["id"], "$path: id"; minimum=1)
        haskey(grouped, id) || study_error("$path: unexpected configuration id $id")
        push!(grouped[id], (path, only(rows)))
    end
    grouped
end

empty_metrics(selected) = Dict(r => Dict(m => Float64[] for m in (RESOURCE_METRICS..., QUALITY_METRICS...))
    for r in selected)

"""Validate one chunk and its diagnostics before admitting its observations."""
function read_chunk(path, result, config)
    lo, hi, selected = chunk_scope(result, path, config)
    stem = path[1:end-4]
    sidecar = stem * ".replicates.csv"
    expected = Set((r, route) for r in lo:hi for route in selected)
    seen, failed = Set{Tuple{Int,String}}(), Set{Tuple{Int,String}}()
    metrics = empty_metrics(selected)
    successes, failures = Dict(r => 0 for r in selected), Dict(r => 0 for r in selected)
    statuses = Dict(r => Dict{String,Int}() for r in selected)
    outlier = config.study == :outlier
    applicable = outlier ? OUTLIER_METRICS : CLUSTER_METRICS
    for record in read_rows(sidecar, OutlierStudy.RAW_HEADER)
        replicate = integer_field(record["replicate"], "$sidecar: replicate"; minimum=1)
        route = record["route"]
        key = (replicate, route)
        key in expected && !(key in seen) || study_error("$sidecar: duplicate or unexpected replicate/route $key")
        push!(seen, key)
        integer_field(record["seed"], "$sidecar: seed") == config.seed + replicate - 1 ||
            study_error("$sidecar: incorrect seed for replicate $replicate")
        status = record["status"]
        status in ("success", "failure") || study_error("$sidecar: unrecognized fit status $(repr(status))")
        if status == "failure"
            failures[route] += 1
            push!(failed, key)
            if any(!isempty(record[m]) for m in (QUALITY_METRICS..., "positives", "settings"))
                study_error("$sidecar: failed fit has quality metrics or settings")
            end
        else
            successes[route] += 1
            isempty(record["settings"]) && study_error("$sidecar: successful fit has no recorded settings")
            for m in QUALITY_METRICS
                (m in applicable) == !isempty(record[m]) ||
                    study_error("$sidecar: metric $m does not match the $(config.study) study")
            end
            positives = outlier ? string(OutlierStudy.outlier_count(config)) : ""
            record["positives"] == positives || study_error("$sidecar: wrong number of true outliers")
            fit = fit_status(record["settings"])
            fit === missing || (statuses[route][fit] = get(statuses[route], fit, 0) + 1)
        end
        for metric in RESOURCE_METRICS
            push!(metrics[route][metric], finite_field(record[metric], "$sidecar: $metric"; minimum=0))
        end
        status == "success" || continue
        for metric in applicable
            value = record[metric]
            number = if metric == "flagged"
                integer_field(value, "$sidecar: flagged") <= config.n ? parse(Float64, value) :
                    study_error("$sidecar: more curves flagged than observed")
            elseif metric == "adjusted_rand_index"
                finite_field(value, "$sidecar: $metric"; minimum=-1, maximum=1)
            else
                finite_field(value, "$sidecar: $metric"; minimum=0, maximum=1)
            end
            push!(metrics[route][metric], number)
        end
    end
    seen == expected || study_error("$sidecar: missing $(length(setdiff(expected, seen))) declared replicate/route records")
    failure_path = stem * ".failures.csv"
    logged = Set{Tuple{Int,String}}()
    for record in read_rows(failure_path, OutlierStudy.FAILURE_HEADER)
        key = (integer_field(record["replicate"], "$failure_path: replicate"; minimum=1), record["route"])
        key in failed && !(key in logged) || study_error("$failure_path: duplicate or unexpected failure $key")
        push!(logged, key)
        integer_field(record["seed"], "$failure_path: seed") == config.seed + key[1] - 1 ||
            study_error("$failure_path: incorrect failure seed")
        record["stage"] in ("fit", "metrics") && !isempty(record["error"]) && !isempty(strip(record["error_type"])) ||
            study_error("$failure_path: failed fit needs its stage, type and error")
    end
    logged == failed || study_error("$failure_path: missing failed-fit diagnostics")
    for route in selected
        validate_aggregates(result, path, route, successes[route], failures[route], metrics[route])
    end
    (; seen, metrics, successes, failures, statuses)
end

moments(values) = (isempty(values) ? NaN : mean(values), length(values) > 1 ? std(values) : NaN)

function summarize(configs, directory)
    summaries = Dict{String,Any}[]
    common_versions = nothing
    chunks = result_chunks(configs, directory)
    for config in configs
        cell_chunks = chunks[config.id]
        isempty(cell_chunks) && study_error("$directory: missing result chunks for cell $(config.id)")
        cell_routes = String.(string.(OutlierStudy.routes(config)))
        expected = Set((r, route) for r in 1:config.replicates for route in cell_routes)
        seen = Set{Tuple{Int,String}}()
        metrics = empty_metrics(cell_routes)
        successes, failures = Dict(r => 0 for r in cell_routes), Dict(r => 0 for r in cell_routes)
        statuses = Dict(r => Dict{String,Int}() for r in cell_routes)
        for (path, result) in cell_chunks
            for (field, value) in zip(OutlierStudy.CONFIG_HEADER, values(config))
                result[field] == configuration_text(value) ||
                    study_error("$path: configuration differs from the requested grid")
            end
            versions = Tuple(result[key] for key in VERSION_FIELDS)
            all(!isempty, versions) || study_error("$path: missing software versions")
            common_versions = something(common_versions, versions)
            versions == common_versions || study_error("$path: software versions differ across chunks/cells")
            finite_field(result["elapsed_seconds"], "$path: elapsed_seconds"; minimum=0)
            chunk = read_chunk(path, result, config)
            isempty(intersect(seen, chunk.seen)) || study_error("$path: duplicate replicate/route across chunks")
            union!(seen, chunk.seen)
            for route in keys(chunk.metrics)
                successes[route] += chunk.successes[route]
                failures[route] += chunk.failures[route]
                mergewith!(+, statuses[route], chunk.statuses[route])
                for (metric, values) in chunk.metrics[route]
                    append!(metrics[route][metric], values)
                end
            end
        end
        seen == expected || study_error("$directory: cell $(config.id) is missing " *
            "$(length(setdiff(expected, seen))) configured replicate/route records")
        for route in cell_routes
            row = Dict{String,Any}(field => configuration_text(value)
                for (field, value) in zip(OutlierStudy.CONFIG_HEADER, values(config)))
            row["outlier_type"] = config.study == :outlier ? string(OutlierStudy.outlier_type(config.scenario)) : ""
            row["route"] = route
            row["successes"], row["failures"] = successes[route], failures[route]
            nonconverged = sum(count for (status, count) in statuses[route] if status != "converged"; init=0)
            row["nonconverged"] = isempty(statuses[route]) ? "" : nonconverged
            row["statuses"] = join(("$s=$c" for (s, c) in sort!(collect(statuses[route]))), ';')
            for metric in (RESOURCE_METRICS..., QUALITY_METRICS...)
                values = metrics[route][metric]
                row["$(metric)_count"] = length(values)
                if metric == "process_peak_rss_bytes"
                    row["$(metric)_max"] = isempty(values) ? NaN : maximum(values)
                    continue
                end
                row["$(metric)_mean"], row["$(metric)_sd"] = moments(values)
                metric in ("adjusted_rand_index", "true_positive_rate", "false_positive_rate") &&
                    (row["$(metric)_median"] = isempty(values) ? NaN : median(values))
            end
            for (field, value) in zip(VERSION_FIELDS, common_versions)
                row[field] = value
            end
            push!(summaries, row)
        end
    end
    summaries
end

const SUMMARY_FIELDS = vcat(OutlierStudy.CONFIG_HEADER, ["outlier_type", "route", "successes", "failures", "nonconverged", "statuses"],
    ["$(m)_$(s)" for m in (RESOURCE_METRICS..., QUALITY_METRICS...)
        for s in (m == "process_peak_rss_bytes" ? ("count", "max") :
            m in ("adjusted_rand_index", "true_positive_rate", "false_positive_rate") ?
                ("count", "mean", "sd", "median") : ("count", "mean", "sd"))],
    collect(VERSION_FIELDS))

fmt(x::AbstractFloat) = isnan(x) ? "" : @sprintf("%.3g", x)
fmt(x) = string(x)

function write_markdown(io, rows)
    println(io, "# Outlier detection and clustering study\n")
    println(io, "Every configured replicate is retained for every route; failures are counted and ",
        "never replaced. Quality summaries are conditional on success. True and false positive rates ",
        "are the fractions of true outliers and of typical curves flagged; ARI is the adjusted Rand ",
        "index of the partition against the true groups. \"Not converged\" counts successful fits ",
        "whose recorded status is not `converged`. Peak RSS is the cumulative route-worker process peak.\n")
    println(io, "## Outlier detection\n")
    println(io, "| Cell | Model | Outlier type | Contamination | Design | n | Route | OK | Failures | TPR (mean) | ",
        "FPR (mean) | Flagged (mean) | Seconds (mean) |")
    println(io, "|---:|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows
        row["study"] == "outlier" || continue
        println(io, "| ", join((row["id"], row["scenario"], row["outlier_type"], row["contamination"], row["design"], row["n"],
            row["route"], "$(row["successes"])/$(row["replicates"])", row["failures"],
            fmt(row["true_positive_rate_mean"]), fmt(row["false_positive_rate_mean"]),
            fmt(row["flagged_mean"]), fmt(row["wall_seconds_mean"])), " | "), " |")
    end
    println(io, "\n## Clustering\n")
    println(io, "| Cell | Groups differ in | Groups | Design | n | Route | OK | Failures | Not converged | ",
        "ARI (median) | ARI (mean) | ARI (SD) | Seconds (mean) |")
    println(io, "|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows
        row["study"] == "clustering" || continue
        println(io, "| ", join((row["id"], row["scenario"], row["groups"], row["design"], row["n"],
            row["route"], "$(row["successes"])/$(row["replicates"])", row["failures"], row["nonconverged"],
            fmt(row["adjusted_rand_index_median"]), fmt(row["adjusted_rand_index_mean"]),
            fmt(row["adjusted_rand_index_sd"]), fmt(row["wall_seconds_mean"])), " | "), " |")
    end
    println(io, "\n## Resources (all recorded attempts)\n")
    println(io, "| Cell | Study | Design | n | Route | Wall seconds (mean) | Allocated MB (mean) | ",
        "Max process peak RSS MB |")
    println(io, "|---:|---|---|---:|---|---:|---:|---:|")
    for row in rows
        println(io, "| ", join((row["id"], row["study"], row["design"], row["n"], row["route"],
            fmt(row["wall_seconds_mean"]), fmt(row["allocated_bytes_mean"] / 1e6),
            fmt(row["process_peak_rss_bytes_max"] / 1e6)), " | "), " |")
    end
end

function write_summary(config_path, directory, prefix)
    configs = read_configurations(config_path)
    rows = summarize(configs, directory)
    csv_path, md_path = prefix * ".csv", prefix * ".md"
    (ispath(csv_path) || ispath(md_path)) && study_error("summary outputs already exist; use a new prefix")
    mkpath(dirname(abspath(prefix)))
    open(csv_path, "w") do io
        println(io, join(SUMMARY_FIELDS, ','))
        for row in rows
            println(io, join((csvtext(row[field]) for field in SUMMARY_FIELDS), ','))
        end
    end
    open(io -> write_markdown(io, rows), md_path, "w")
    println("Wrote ", csv_path, " and ", md_path, "; ", length(configs), " configurations, ",
        length(rows), " route rows")
    rows
end

csvtext(x::AbstractFloat) = isnan(x) ? "" : repr(x)
csvtext(x::AbstractString) = occursin(r"[,\"\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x
csvtext(x) = string(x)

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || throw(ArgumentError("usage: summarize.jl CONFIG.csv RESULTS_DIR OUTPUT_PREFIX"))
    write_summary(ARGS...)
end
