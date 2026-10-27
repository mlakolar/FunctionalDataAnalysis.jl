# Create or validate complete route/replicate work lists for GNU Parallel.
# Usage: julia --project=. work_units.jl CONFIG.csv UNITS.tsv [--block-size N | --check]
isdefined(@__MODULE__, :OutlierStudy) || include("common.jl")
using .OutlierStudy

const UNIT_FIELDS = ["cell", "route", "first_replicate", "last_replicate", "output"]

function read_configurations(path)
    lines = readlines(path)
    length(lines) >= 2 || throw(ArgumentError("configuration grid is empty"))
    [OutlierStudy.read_config(path, index) for index in 1:length(lines)-1]
end

unit_output(cell, route, first, last) =
    "cell-$(lpad(cell, 3, '0'))-$(route)-$(lpad(first, 4, '0'))-$(lpad(last, 4, '0')).csv"

"""Each configured (cell, route, replicate) exactly once, in order."""
function work_units(configs, block_size::Integer)
    block_size >= 1 || throw(ArgumentError("block size must be a positive integer"))
    units = NamedTuple[]
    for config in configs, route in OutlierStudy.routes(config), lo in 1:block_size:config.replicates
        hi = min(lo + block_size - 1, config.replicates)
        push!(units, (; cell=config.id, route, first=lo, last=hi, output=unit_output(config.id, route, lo, hi)))
    end
    units
end

function write_work_units(path, units)
    ispath(path) && throw(ArgumentError("$path already exists; use a new output path"))
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(UNIT_FIELDS, '\t'))
        for u in units
            println(io, join((u.cell, u.route, u.first, u.last, u.output), '\t'))
        end
    end
    length(units)
end

function positive_integer(field)
    value = tryparse(Int, field)
    value !== nothing && value >= 1 && string(value) == field ||
        throw(ArgumentError("expected a positive integer, got $(repr(field))"))
    value
end

"""Validate a work list before launch without reading or changing results."""
function check_work_units(path, configs)
    lines = readlines(path)
    !isempty(lines) && split(first(lines), '\t') == UNIT_FIELDS ||
        throw(ArgumentError("unexpected work-list TSV header"))
    observed = Dict{Tuple{Int,Symbol},Set{Int}}()
    for line in lines[2:end]
        fields = split(line, '\t')
        length(fields) == length(UNIT_FIELDS) || throw(ArgumentError("malformed work-list row"))
        cell = positive_integer(fields[1])
        cell <= length(configs) || throw(ArgumentError("work-list cell $cell is outside the configuration grid"))
        config = configs[cell]
        route = Symbol(fields[2])
        lo, hi = positive_integer(fields[3]), positive_integer(fields[4])
        route in OutlierStudy.routes(config) && lo <= hi <= config.replicates ||
            throw(ArgumentError("work-list route or replicate range does not match cell $cell"))
        fields[5] == unit_output(cell, route, lo, hi) ||
            throw(ArgumentError("work-list output must be the canonical chunk basename"))
        seen = get!(observed, (cell, route), Set{Int}())
        attempted = Set(lo:hi)
        isempty(intersect(seen, attempted)) ||
            throw(ArgumentError("duplicate work-list replicates for cell $cell, route $route"))
        union!(seen, attempted)
    end
    for config in configs, route in OutlierStudy.routes(config)
        get(observed, (config.id, route), Set{Int}()) == Set(1:config.replicates) ||
            throw(ArgumentError("missing work-list replicates for cell $(config.id), route $route"))
    end
    length(lines) - 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    usage = "usage: work_units.jl CONFIG.csv UNITS.tsv [--block-size N | --check]"
    length(ARGS) in (2, 3, 4) || throw(ArgumentError(usage))
    configs = read_configurations(ARGS[1])
    if length(ARGS) == 3
        ARGS[3] == "--check" || throw(ArgumentError(usage))
        println("Validated ", check_work_units(ARGS[2], configs), " work units")
    else
        block_size = length(ARGS) == 4 ? (ARGS[3] == "--block-size" ? positive_integer(ARGS[4]) :
            throw(ArgumentError(usage))) : 200
        println("Wrote ", write_work_units(ARGS[2], work_units(configs, block_size)), " work units to ", ARGS[2])
    end
end
