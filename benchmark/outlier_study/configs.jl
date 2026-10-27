# Generate the outlier-detection and clustering study grid (no FDA dependency).
# Usage: julia configs.jl OUTPUT.csv [--smoke | --probe]

const CONFIG_FIELDS = ("id", "study", "scenario", "design", "n", "contamination", "groups",
    "replicates", "seed")
const GRID_KINDS = (:full, :smoke, :probe)

"""
Deterministic cells; all routes in a cell share replicate seeds. `:smoke` is two
small cells that exercise the harness; `:probe` is one cell per study, design
and `n` at full size, with two replicates, for the cost estimate (estimate.jl).
"""
function configurations(; kind=:full)
    kind in GRID_KINDS || throw(ArgumentError("grid kind must be one of $GRID_KINDS"))
    cells = if kind == :smoke
        [("outlier", "model6", "sparse", 40, "0.1", ""),
         ("clustering", "covariance", "dense", 40, "", "2")]
    elseif kind == :probe
        vcat([("outlier", "model1", design, n, "0.1", "") for design in ("dense", "sparse") for n in (100, 500)],
             [("clustering", "covariance", design, n, "", "3") for design in ("dense", "sparse") for n in (100, 500)])
    else
        outlier = [("outlier", scenario, design, n, contamination, "")
            for scenario in ("model$m" for m in 1:9)
            for contamination in ("0.05", "0.1") for design in ("dense", "sparse") for n in (100, 500)]
        clustering = [("clustering", scenario, design, n, "", groups)
            for scenario in ("mean", "covariance") for groups in ("2", "3")
            for design in ("dense", "sparse") for n in (100, 500)]
        vcat(outlier, clustering)
    end
    [(index, cell..., kind == :full ? 200 : 2, 20261027 + 1000index) for (index, cell) in enumerate(cells)]
end

function write_configurations(path; kind=:full)
    ispath(path) && throw(ArgumentError("$path already exists; use a new output path"))
    rows = configurations(; kind)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(CONFIG_FIELDS, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
    path
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (1, 2) && (length(ARGS) == 1 || ARGS[2] in ("--smoke", "--probe")) ||
        throw(ArgumentError("usage: configs.jl OUTPUT.csv [--smoke | --probe]"))
    write_configurations(ARGS[1]; kind=length(ARGS) == 1 ? :full : Symbol(ARGS[2][3:end]))
    println("Wrote ", ARGS[1])
end
