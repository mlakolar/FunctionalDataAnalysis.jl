# Study implementation, not package API. Julia >= 1.10; FDA 0.1.0.
module OutlierStudy

using FunctionalDataAnalysis
using LinearAlgebra
using Random
import StatsAPI: fit

const CONFIG_HEADER = ["id", "study", "scenario", "design", "n", "contamination", "groups",
    "replicates", "seed"]
const STUDIES = (:outlier, :clustering)
const DESIGNS = (:dense, :sparse)
const OUTLIER_SCENARIOS = Tuple(Symbol("model$m") for m in 1:9)
const CLUSTER_SCENARIOS = (:mean, :covariance)
const OUTLIER_ROUTES = (:boxplot, :outliergram, :msplot)
const CLUSTER_ROUTES = (:kmeans_curves, :kmeans_scores, :kcfc, :mixture)
const ALL_ROUTES = (OUTLIER_ROUTES..., CLUSTER_ROUTES...)
const METRIC_NAMES = (:true_positive_rate, :false_positive_rate, :flagged, :adjusted_rand_index)
const RAW_HEADER = ["replicate", "seed", "route", "status", "wall_seconds", "allocated_bytes",
    "process_peak_rss_bytes", string.(METRIC_NAMES)..., "positives", "settings"]
const FAILURE_HEADER = ["replicate", "seed", "route", "stage", "error_type", "error"]

# Both studies: dense curves on 50 equally spaced points of [0,1] (fdaoutlier's
# simulation default), sparse curves at 5-10 uniform times per curve.
const DOMAIN = (0.0, 1.0)
const GRID = collect(range(0.0, 1.0; length=50))
const SPARSE_POINTS = 5:10

"""Routes run in a cell. k-means of curves needs complete curves, so dense only."""
function routes(config)
    config.study == :outlier && return OUTLIER_ROUTES
    config.design == :dense ? CLUSTER_ROUTES : CLUSTER_ROUTES[2:end]
end

function parse_optional(T, field)
    isempty(field) ? nothing : parse(T, field)
end

function read_config(path, index)
    lines = readlines(path)
    !isempty(lines) && split(first(lines), ',') == CONFIG_HEADER ||
        throw(ArgumentError("unexpected outlier-study configuration CSV header"))
    1 <= index < length(lines) || throw(ArgumentError("configuration index is outside the CSV"))
    fields = split(lines[index+1], ',')
    length(fields) == length(CONFIG_HEADER) || throw(ArgumentError("configuration row must have nine columns"))
    id, study, scenario, design, n, contamination, groups, replicates, seed = fields
    c = (; id=parse(Int, id), study=Symbol(study), scenario=Symbol(scenario), design=Symbol(design),
        n=parse(Int, n), contamination=parse_optional(Float64, contamination),
        groups=parse_optional(Int, groups), replicates=parse(Int, replicates), seed=parse(Int, seed))
    c.id == index || throw(ArgumentError("configuration id must match its one-based row index"))
    c.study in STUDIES || throw(ArgumentError("study must be one of $STUDIES"))
    c.design in DESIGNS || throw(ArgumentError("design must be dense or sparse"))
    c.n >= 20 || throw(ArgumentError("n must be at least 20"))
    if c.study == :outlier
        c.scenario in OUTLIER_SCENARIOS || throw(ArgumentError("outlier scenario must be one of $OUTLIER_SCENARIOS"))
        c.contamination !== nothing && 0 < c.contamination < 0.5 ||
            throw(ArgumentError("outlier cells need a contamination in (0, 0.5)"))
        c.groups === nothing || throw(ArgumentError("outlier cells have no groups"))
        1 <= outlier_count(c) < c.n || throw(ArgumentError("contamination must give at least one outlier"))
    else
        c.scenario in CLUSTER_SCENARIOS || throw(ArgumentError("clustering scenario must be one of $CLUSTER_SCENARIOS"))
        c.groups in (2, 3) || throw(ArgumentError("clustering cells need 2 or 3 groups"))
        c.contamination === nothing || throw(ArgumentError("clustering cells have no contamination"))
    end
    c.replicates >= 1 || throw(ArgumentError("replicates must be positive"))
    0 <= c.seed <= typemax(Int) ÷ 16 - c.replicates || throw(ArgumentError("seed range must be nonnegative and fit in Int"))
    c
end

outlier_count(config) = round(Int, config.contamination * config.n)

# --- Curves -------------------------------------------------------------------

function sample_times(rng, design)
    design == :dense && return copy(GRID)
    count = rand(rng, SPARSE_POINTS)
    for _ in 1:1000
        times = sort!(rand(rng, count))
        allunique(times) && return times
    end
    throw(ArgumentError("could not draw distinct sparse times"))
end

function assemble(design, times, values)
    design == :dense && return GridFunctionalData(reduce(vcat, transpose.(values)), copy(GRID))
    IrregularFunctionalData(times, values; domain=DOMAIN)
end

# Outlier study: fdaoutlier 0.2.1's published simulation models
# (`simulation_model1` ... `simulation_model9`, R/simulation_models.R) at their
# default parameters, so that detection rates are comparable with the
# literature. Every model is X(t) = mean(t) + e(t) with e a zero-mean Gaussian
# process with covariance α exp(-β|s - t|^ν) (fdaoutlier `covfunexp`). Curve
# parameters are drawn from `rng` in our own order; the deterministic parts
# reproduce fdaoutlier's output (reference/fdaoutlier_models.R and check.jl).
# The outlier type of each model, in the taxonomy of Hubert, Rousseeuw and
# Segaert (2015) and Dai and Genton (2018):
const OUTLIER_TYPES = (model1=:shift, model2=:magnitude, model3=:magnitude, model4=:shape,
    model5=:shape, model6=:mixed, model7=:shape, model8=:shape, model9=:amplitude)
outlier_type(scenario) = OUTLIER_TYPES[scenario]

const DEFAULT_COVARIANCE = (1.0, 1.0, 1.0)            # (α, β, ν)
covariance_parameters(scenario, outlier) =
    scenario == :model4 ? (0.3, 1 / 0.3, 1.0) :       # model 4: cov_alpha = 0.3, cov_beta = 1/0.3
    scenario == :model5 && outlier ? (5.0, 2.0, 0.5) : # model 5 outliers: cov_alpha2, cov_beta2, cov_nu2
    DEFAULT_COVARIANCE
gp_covariance(t, (alpha, beta, nu)=DEFAULT_COVARIANCE) = [alpha * exp(-beta * abs(a - b)^nu) for a in t, b in t]
const DENSE_FACTORS = Dict(c => cholesky(Symmetric(gp_covariance(GRID, c))).L
    for c in (DEFAULT_COVARIANCE, (0.3, 1 / 0.3, 1.0), (5.0, 2.0, 0.5)))
gp_factor(t, c=DEFAULT_COVARIANCE) = t == GRID ? DENSE_FACTORS[c] : cholesky(Symmetric(gp_covariance(t, c))).L

uniform(rng, a, b) = a + (b - a) * rand(rng)
sign_draw(rng) = rand(rng, Bool) ? 1.0 : -1.0          # P(k = 1) = kprob = 0.5

"""Random parameters of one curve (fdaoutlier defaults); empty when the model has none."""
function curve_parameters(rng, scenario, outlier)
    if scenario == :model9
        outlier || return (uniform(rng, 3, 8), uniform(rng, 3, 8))      # ai = c(3, 8)
        high = rand(rng, Bool)                                          # kprob = 0.5
        a, b = high ? (9.0, 10.5) : (1.5, 2.5)                          # ci = c(9, 10.5), bi = c(1.5, 2.5)
        return (uniform(rng, a, b), uniform(rng, a, b), Float64(high))
    end
    outlier || return ()
    scenario == :model1 && return (sign_draw(rng),)
    scenario in (:model2, :model3) && return (sign_draw(rng), uniform(rng, 0.1, 0.9))   # a = 0.1, b = 0.9
    scenario == :model6 && return (Float64(rand(rng, Bool)), uniform(rng, 0.25, 0.75))  # u; a = 0.25, b = 0.75
    scenario == :model7 && return (uniform(rng, 0.25, 0.75),)                           # θ; a = 0.25, b = 0.75
    ()
end

"""Deterministic part of a curve at `t`, given its parameters."""
function model_mean(scenario, outlier, params, t)
    if scenario == :model4
        return outlier ? 30 * (1 - t) * t^1.5 : 30 * t * (1 - t)^1.5              # mu = 30, m = 3/2
    elseif scenario == :model8
        return outlier ? 2sin(15pi * t + 2) : 2sin(15pi * t)                      # k = 2, r = 15, v = 2
    elseif scenario == :model9
        return params[1] * sin(2pi * t) + params[2] * cos(2pi * t)
    end
    mu = 4t                                                                       # mu = 4
    outlier || return mu
    if scenario == :model1
        return mu + 8 * params[1]                                                 # q = 8
    elseif scenario == :model2
        return mu + (params[2] <= t <= params[2] + 0.05 ? 8 * params[1] : 0.0)    # q = 8, l = 0.05
    elseif scenario == :model3
        return mu + (t >= params[2] ? 6 * params[1] : 0.0)                        # q = 6
    elseif scenario == :model6
        s, v = params[1] == 1 ? -1.0 : 1.0, params[2]                            # (-1)^u; q = 1.8, r = 0.02, z = 50, w = 2
        return mu + s * 1.8 - s * exp(-50 * (t - v)^2) / sqrt(0.02pi)
    elseif scenario == :model7
        return mu + 2sin(4pi * (t + params[1]))                                   # k = 2, r = 4
    end
    mu                                                                            # model 5
end

"""One outlier-study sample: `(; data, outliers)`, outliers sorted."""
function outlier_sample(config, replicate)
    rng = MersenneTwister(config.seed + replicate - 1)
    outliers = sort!(randperm(rng, config.n)[1:outlier_count(config)])
    flagged = falses(config.n)
    flagged[outliers] .= true
    times, values = Vector{Float64}[], Vector{Float64}[]
    for i in 1:config.n
        t = sample_times(rng, config.design)
        params = curve_parameters(rng, config.scenario, flagged[i])
        e = gp_factor(t, covariance_parameters(config.scenario, flagged[i])) * randn(rng, length(t))
        push!(times, t)
        push!(values, model_mean.(config.scenario, flagged[i], Ref(params), t) .+ e)
    end
    (; data=assemble(config.design, times, values), outliers)
end

# Clustering study: curves 1 + t + group mean + Σ_j ξ_j φ_j(t) + noise with
# φ_j(t) = √2 sin(jπt), j = 1, 2, 3, orthonormal on [0,1].
#   :mean       group means 2.5 apart along φ_1; common score SDs (1, √0.5, 0.5);
#   :covariance common mean; group g has score SD 1 along φ_g and 0.2 along the others.
const CLUSTER_NOISE_SD = 0.1
const MEAN_STEP = 2.5
mode(j, t) = sqrt(2) * sinpi(j * t)
group_offset(scenario, g, G) = scenario == :mean ? MEAN_STEP * (g - (G + 1) / 2) : 0.0
group_sd(scenario, g) = scenario == :mean ? [1.0, sqrt(0.5), 0.5] : [j == g ? 1.0 : 0.2 for j in 1:3]

"""Balanced, interleaved true labels."""
group_labels(n, G) = [(i - 1) % G + 1 for i in 1:n]

"""One clustering-study sample: `(; data, labels)`."""
function cluster_sample(config, replicate)
    rng = MersenneTwister(config.seed + replicate - 1)
    labels = group_labels(config.n, config.groups)
    times, values = Vector{Float64}[], Vector{Float64}[]
    for g in labels
        t = sample_times(rng, config.design)
        xi = group_sd(config.scenario, g) .* randn(rng, 3)
        xi[1] += group_offset(config.scenario, g, config.groups)
        y = [1 + s + sum(xi[j] * mode(j, s) for j in 1:3) for s in t] .+ CLUSTER_NOISE_SD .* randn(rng, length(t))
        push!(times, t)
        push!(values, y)
    end
    (; data=assemble(config.design, times, values), labels)
end

simulation(config, replicate) =
    config.study == :outlier ? outlier_sample(config, replicate) : cluster_sample(config, replicate)

# --- Methods and metrics ------------------------------------------------------

"""Method randomness (MCD subsets, k-means seeds, EM starts), separate from the data stream."""
method_rng(seed, route) = MersenneTwister(16seed + findfirst(==(route), ALL_ROUTES))

# Package defaults throughout. The score-based clustering routes use FVE(0.9),
# the FPCA of kCFC's initial step and of the mixture (fdapace kCFC and FClust).
function fit_route(route, data, config, rng)
    route == :boxplot && return functional_boxplot(data)
    route == :outliergram && return outliergram(data)
    route == :msplot && return magnitude_shape(data; rng)
    k = config.groups
    route == :kmeans_curves && return fit(FunctionalKMeans, data; k, rng)
    route == :kmeans_scores && return fit(FunctionalKMeans, fit(FPCA, data; ncomponents=FVE(0.9)); k, rng)
    route == :kcfc && return fit(KCFC, data; k, rng)
    route == :mixture && return fit(FunctionalGaussianMixture, data; k, rng)
    throw(ArgumentError("unknown route $route"))
end

const EMPTY_METRICS = NamedTuple{METRIC_NAMES}(ntuple(_ -> nothing, length(METRIC_NAMES)))

function accuracy(fitted, prepared, n)
    if hasproperty(prepared, :outliers)
        truth = prepared.outliers
        flagged = fitted.outliers
        true_positives = count(in(truth), flagged)
        metrics = merge(EMPTY_METRICS, (; true_positive_rate=true_positives / length(truth),
            false_positive_rate=(length(flagged) - true_positives) / (n - length(truth)),
            flagged=length(flagged)))
        return (; metrics, positives=length(truth))
    end
    labels = assignments(fitted)
    length(labels) == n || throw(DimensionMismatch("clustering returned $(length(labels)) labels for $n curves"))
    metrics = merge(EMPTY_METRICS, (; adjusted_rand_index=randindex(labels, prepared.labels)[1]))
    all(x -> x === nothing || isfinite(x), metrics) || throw(ArgumentError("study metrics must be finite"))
    (; metrics, positives=nothing)
end

fpca_components(::Nothing) = "none"
fpca_components(f) = string(size(f.scores, 2))

# Selected tuning values and statuses for every retained replicate.
function tuning(b::FunctionalBoxplot)
    join(String["factor=$(b.factor)", "central=$(b.settings.central)", "central_curves=$(length(b.central))",
        "representation=$(b.depth.representation)", "fpca_components=$(fpca_components(b.depth.fpca))"], ';')
end

function tuning(o::Outliergram)
    join(String["threshold=$(repr(o.threshold))", "first_stage=$(length(o.flagged))",
        "shift_check=$(length(o.shifted))", "representation=$(o.representation)",
        "fpca_components=$(fpca_components(o.fpca))"], ';')
end

function tuning(s::MagnitudeShape)
    join(String["cutoff=$(repr(s.cutoff))", "cutoff_rule=$(s.settings.cutoff_rule)",
        "mcd_subsets=$(s.settings.mcd.nsamp)", "mcd_exact=$(s.settings.mcd.exact)",
        "representation=$(s.representation)", "fpca_components=$(fpca_components(s.fpca))"], ';')
end

function tuning(m::FunctionalKMeans)
    join(String["space=$(m.space)", "iterations=$(m.iterations)", "converged=$(m.converged)",
        "fpca_components=$(fpca_components(m.fpca))"], ';')
end

function tuning(m::KCFC)
    join(String["status=$(m.status)", "iterations=$(m.iterations)",
        "fpca_method=$(m.initial_fpca.method)", "fpca_components=$(fpca_components(m.initial_fpca))",
        "cluster_components=$(join(fpca_components.(m.fpcas), '/'))"], ';')
end

function tuning(m::FunctionalGaussianMixture)
    join(String["status=$(m.status)", "iterations=$(m.iterations)",
        "fpca_method=$(m.fpca.method)", "fpca_components=$(fpca_components(m.fpca))"], ';')
end

fit_failure(err) = err isa Exception && !(err isa Union{InterruptException,OutOfMemoryError})

# Only our own numeric raw fields are read here (no commas inside them).
# summarize.jl independently validates the complete CSV and all counts.
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

"""Parse one line written by `csvrow` (every cell quoted, quotes doubled, no raw newlines)."""
function parse_csvrow(line::AbstractString)
    cells = String[]
    i, n = firstindex(line), lastindex(line)
    isempty(line) && return cells
    while true
        line[i] == '"' || throw(ArgumentError("unquoted CSV cell"))
        buffer = IOBuffer()
        i = nextind(line, i)
        while true
            i > n && throw(ArgumentError("unterminated CSV cell"))
            c = line[i]
            if c == '"'
                j = nextind(line, i)
                if j <= n && line[j] == '"'
                    write(buffer, '"')
                    i = nextind(line, j)
                else
                    i = j
                    break
                end
            else
                write(buffer, c)
                i = nextind(line, i)
            end
        end
        push!(cells, String(take!(buffer)))
        i > n && return cells
        line[i] == ',' || throw(ArgumentError("malformed CSV separator"))
        i = nextind(line, i)
        i > n && throw(ArgumentError("trailing CSV separator"))
    end
end

end
