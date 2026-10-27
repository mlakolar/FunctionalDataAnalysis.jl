# Focused harness checks only; no study runs. Usage: julia --project=. check.jl
include("run.jl")
include("work_units.jl")
include("summarize.jl")
include("configs.jl")
include("estimate.jl")
using .OutlierStudy
using FunctionalDataAnalysis
using LinearAlgebra
using Random
using Test
import Clustering

const S = OutlierStudy
const HERE = @__DIR__
const SMOKE = joinpath(HERE, "smoke")

struct StudyTestException <: Exception end
struct StandInClustering <: Clustering.ClusteringResult
    assignments::Vector{Int}
end

config_row(; id=1, study=:outlier, scenario=:model7, design=:dense, n=40, contamination=0.1,
    groups=nothing, replicates=2, seed=20262027) =
    (; id, study, scenario, design, n, contamination, groups, replicates, seed)

function write_config(path, rows)
    open(path, "w") do io
        println(io, join(S.CONFIG_HEADER, ','))
        for r in rows
            println(io, join((x === nothing ? "" : x for x in r), ','))
        end
    end
    path
end

@testset "Configuration grid and routes" begin
    cells = configurations()
    @test length(cells) == 88
    @test allunique(c[end] for c in cells)
    @test allunique(c[2:7] for c in cells)
    @test count(c -> c[2] == "outlier", cells) == 9 * 2 * 2 * 2
    @test all(c -> c[end-1] == 200, cells)
    @test length(configurations(; kind=:smoke)) == 2 && length(configurations(; kind=:probe)) == 8
    @test_throws ArgumentError configurations(; kind=:other)
    mktempdir() do dir
        path = write_configurations(joinpath(dir, "grid.csv"))
        @test read(path, String) == read(joinpath(HERE, "configurations.csv"), String)
        @test_throws ArgumentError write_configurations(path)
        for kind in (:smoke, :probe)
            grid = write_configurations(joinpath(dir, "$kind.csv"); kind)
            @test read(grid, String) == read(joinpath(SMOKE, kind == :smoke ? "" : "probe", "configurations.csv"), String)
            @test length(read_configurations(grid)) == length(configurations(; kind))
        end
    end
    configs = read_configurations(joinpath(HERE, "configurations.csv"))
    @test sum(length(S.routes(c)) for c in configs) == 72 * 3 + 8 * 4 + 8 * 3
    @test S.routes(config_row()) == S.OUTLIER_ROUTES
    @test S.routes(config_row(study=:clustering, contamination=nothing, groups=2)) == S.CLUSTER_ROUTES
    @test S.routes(config_row(study=:clustering, design=:sparse, contamination=nothing, groups=2)) ==
        (:kmeans_scores, :kcfc, :mixture)
    @test S.outlier_count(config_row(n=500, contamination=0.05)) == 25
    mktempdir() do dir
        good = write_config(joinpath(dir, "good.csv"), [values(config_row()),
            values(config_row(id=2, study=:clustering, scenario=:mean, contamination=nothing, groups=3))])
        @test S.read_config(good, 1) == config_row()
        @test S.read_config(good, 2).groups == 3
        @test_throws ArgumentError S.read_config(good, 0)
        @test_throws ArgumentError S.read_config(good, 3)
        for bad in (config_row(id=2), config_row(study=:other), config_row(scenario=:mean),
                config_row(contamination=nothing), config_row(contamination=0.6), config_row(groups=2),
                config_row(n=10), config_row(contamination=0.01),
                config_row(study=:clustering, scenario=:mean, contamination=nothing, groups=4),
                config_row(study=:clustering, scenario=:model1, contamination=nothing, groups=2),
                config_row(scenario=:model10),
                config_row(study=:clustering, scenario=:mean, contamination=0.1, groups=2),
                config_row(design=:irregular), config_row(replicates=0), config_row(seed=-1))
            path = write_config(joinpath(dir, "bad.csv"), [values(bad)])
            @test_throws ArgumentError S.read_config(path, 1)
        end
        open(joinpath(dir, "header.csv"), "w") do io
            println(io, "id,study")
            println(io, "1,outlier")
        end
        @test_throws ArgumentError S.read_config(joinpath(dir, "header.csv"), 1)
    end
end

"""Rows of a golden CSV after its `#` header comments: (header, rows of strings)."""
function golden_rows(file)
    lines = filter(l -> !startswith(l, '#'), readlines(joinpath(HERE, "reference", file)))
    split(first(lines), ','), [split(l, ',') for l in lines[2:end]]
end

@testset "Outlier designs" begin
    @test S.OUTLIER_SCENARIOS == Tuple(Symbol("model$m") for m in 1:9)
    @test keys(S.OUTLIER_TYPES) == S.OUTLIER_SCENARIOS
    @test Set(values(S.OUTLIER_TYPES)) == Set((:shift, :magnitude, :shape, :mixed, :amplitude))
    # Deterministic parts against fdaoutlier 0.2.1 (process switched off, curve
    # parameters replayed from its RNG): relative error below 1e-8.
    header, rows = golden_rows("fdaoutlier_models.csv")
    @test header[1:6] == ["model", "curve", "outlier", "p1", "p2", "p3"] && length(header) == 6 + length(S.GRID)
    @test sort(unique(parse.(Int, first.(rows)))) == 1:9
    for f in rows
        scenario, outlier = Symbol("model", f[1]), f[3] == "1"
        params = Tuple(parse(Float64, x) for x in f[4:6] if !isempty(x))
        reference = parse.(Float64, f[7:end])
        ours = S.model_mean.(scenario, outlier, Ref(params), S.GRID)
        @test all(isapprox.(ours, reference; rtol=1e-8, atol=1e-12))
    end
    header, rows = golden_rows("fdaoutlier_covariances.csv")
    grid5 = collect(range(0.0, 1.0; length=5))
    for c in unique(Tuple(parse.(Float64, f[1:3])) for f in rows)
        K = S.gp_covariance(grid5, c)
        @test all(isapprox(K[parse(Int, f[4]), parse(Int, f[5])], parse(Float64, f[6]); rtol=1e-8)
            for f in rows if Tuple(parse.(Float64, f[1:3])) == c)
    end
    @test S.covariance_parameters(:model4, false) == S.covariance_parameters(:model4, true) == (0.3, 1 / 0.3, 1.0)
    @test S.covariance_parameters(:model5, true) == (5.0, 2.0, 0.5)
    @test S.covariance_parameters(:model5, false) == S.covariance_parameters(:model1, true) == (1.0, 1.0, 1.0)
    for (c, L) in S.DENSE_FACTORS
        @test L * L' ≈ S.gp_covariance(S.GRID, c)
        @test S.gp_factor(S.GRID, c) === L
    end
    # Parameter draws follow fdaoutlier's default ranges, with both signs.
    rng = MersenneTwister(1)
    draws(scenario, outlier) = [S.curve_parameters(rng, scenario, outlier) for _ in 1:400]
    for scenario in S.OUTLIER_SCENARIOS
        scenario == :model9 || @test all(isempty, draws(scenario, false))
    end
    for scenario in (:model4, :model5, :model8)
        @test all(isempty, draws(scenario, true))
    end
    k = draws(:model1, true)
    @test Set(only.(k)) == Set((-1.0, 1.0))
    for scenario in (:model2, :model3)
        d = draws(scenario, true)
        @test Set(first.(d)) == Set((-1.0, 1.0)) && all(p -> 0.1 <= p[2] <= 0.9, d)
    end
    d = draws(:model6, true)
    @test Set(first.(d)) == Set((0.0, 1.0)) && all(p -> 0.25 <= p[2] <= 0.75, d)
    @test all(p -> 0.25 <= only(p) <= 0.75, draws(:model7, true))
    @test all(p -> all(x -> 3 <= x <= 8, p), draws(:model9, false))
    d = draws(:model9, true)
    @test Set(last.(d)) == Set((0.0, 1.0))
    @test all(p -> p[3] == 1 ? all(x -> 9 <= x <= 10.5, p[1:2]) : all(x -> 1.5 <= x <= 2.5, p[1:2]), d)

    dense = config_row(n=100, contamination=0.1, scenario=:model1)
    a, b = S.simulation(dense, 2), S.simulation(dense, 2)
    @test a.data.values == b.data.values && a.outliers == b.outliers
    @test S.simulation(dense, 3).data.values != a.data.values
    @test a.data isa GridFunctionalData && a.data.grid == S.GRID && nobs(a.data) == 100
    @test length(a.outliers) == 10 && issorted(a.outliers) && allunique(a.outliers)
    # Model 1 outliers sit 8 away from typical curves on average (a check, not a golden value).
    typical = setdiff(1:100, a.outliers)
    centre = vec(sum(a.data.values[typical, :]; dims=1)) ./ length(typical)
    @test all(i -> abs(sum(a.data.values[i, :] .- centre) / length(S.GRID)) > 4, a.outliers)
    sparse = config_row(design=:sparse, n=60, contamination=0.05, scenario=:model2)
    s = S.simulation(sparse, 1)
    @test s.data isa IrregularFunctionalData && nobs(s.data) == 60 && length(s.outliers) == 3
    @test all(in(S.SPARSE_POINTS), nobs_per_curve(s.data))
    @test all(0 .<= s.data.times .<= 1)
    for scenario in S.OUTLIER_SCENARIOS
        @test nobs(S.simulation(merge(sparse, (; scenario)), 1).data) == 60
    end
    # Dense residuals have the model's process variance at every grid point:
    # 1 by default, 0.3 in model 4 and 5 for model 5's outliers.
    for (scenario, typical_variance, outlier_variance) in ((:model4, 0.3, 0.3), (:model5, 1.0, 5.0))
        big = S.simulation(config_row(n=4000, contamination=0.4, scenario=scenario), 1)
        out = falses(4000)
        out[big.outliers] .= true
        for (outlier, variance) in ((false, typical_variance), (true, outlier_variance))
            selected = outlier ? out : .!out
            residual = big.data.values[selected, :] .- transpose(S.model_mean.(scenario, outlier, Ref(()), S.GRID))
            @test all(x -> abs(x / variance - 1) < 0.15, vec(sum(abs2, residual; dims=1)) ./ count(selected))
        end
    end
end

@testset "Clustering designs" begin
    t = collect(range(0.0, 1.0; length=20001))
    w = quadrature_weights(t)
    gram = [sum(w .* S.mode.(i, t) .* S.mode.(j, t)) for i in 1:3, j in 1:3]
    @test gram ≈ I(3) atol=1e-8
    @test S.group_labels(7, 3) == [1, 2, 3, 1, 2, 3, 1]
    @test [S.group_offset(:mean, g, 2) for g in 1:2] == [-1.25, 1.25]
    @test [S.group_offset(:mean, g, 3) for g in 1:3] == [-2.5, 0.0, 2.5]
    @test S.group_offset(:covariance, 1, 3) == 0
    @test S.group_sd(:covariance, 2) == [0.2, 1.0, 0.2]
    c = config_row(study=:clustering, scenario=:covariance, contamination=nothing, groups=3, n=60)
    a, b = S.simulation(c, 1), S.simulation(c, 1)
    @test a.data.values == b.data.values && a.labels == S.group_labels(60, 3)
    sparse = S.simulation(merge(c, (; design=:sparse, scenario=:mean)), 1)
    @test sparse.data isa IrregularFunctionalData && all(in(S.SPARSE_POINTS), nobs_per_curve(sparse.data))
end

@testset "Metrics and method randomness" begin
    prepared = (; outliers=[2, 7, 9, 10])
    m = S.accuracy((; outliers=[1, 2, 7]), prepared, 20)
    @test m.metrics.true_positive_rate == 1 / 2
    @test m.metrics.false_positive_rate == 1 / 16
    @test m.metrics.flagged == 3
    @test m.metrics.adjusted_rand_index === nothing
    @test m.positives == 4
    none = S.accuracy((; outliers=Int[]), prepared, 20)
    @test none.metrics.true_positive_rate == 0 && none.metrics.false_positive_rate == 0
    # Hand-computed adjusted Rand indices (Hubert and Arabie, 1985).
    ari(a, b) = S.accuracy(StandInClustering(a), (; labels=b), length(b)).metrics.adjusted_rand_index
    @test ari([1, 1, 2, 2], [1, 1, 1, 2]) ≈ 0 atol=1e-15
    @test ari([1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2]) ≈ 8 / 33
    @test ari([2, 2, 1, 1, 3], [1, 1, 3, 3, 2]) ≈ 1
    @test S.accuracy(StandInClustering([1, 2]), (; labels=[1, 2]), 2).positives === nothing
    @test_throws DimensionMismatch S.accuracy(StandInClustering([1, 2]), (; labels=[1, 2, 1]), 3)
    @test allunique(rand(S.method_rng(5, r)) for r in S.ALL_ROUTES)
    @test rand(S.method_rng(5, :kcfc)) == rand(S.method_rng(5, :kcfc))
    @test rand(S.method_rng(5, :kcfc)) != rand(MersenneTwister(5))
    @test fit_status("status=tiny_cluster;iterations=3") == "tiny_cluster"
    @test fit_status("space=L2;converged=false") == "maxiter"
    @test ismissing(fit_status("factor=1.5"))
end

@testset "Work selection and failures" begin
    config = config_row(replicates=4)
    @test work_selection(config, 1, nothing, nothing) == (1:4, S.OUTLIER_ROUTES)
    @test work_selection(config, 2, 3, :msplot) == (2:3, (:msplot,))
    for (lo, hi) in ((0, 1), (3, 2), (1, 5))
        @test_throws ArgumentError work_selection(config, lo, hi, :boxplot)
    end
    @test_throws ArgumentError work_selection(config, 1, 1, :kcfc)
    sparse = config_row(study=:clustering, design=:sparse, contamination=nothing, groups=2)
    @test_throws ArgumentError work_selection(sparse, 1, 1, :kmeans_curves)
    prepared = (; data=nothing, outliers=[1])
    valid = (; metrics=merge(S.EMPTY_METRICS, (; true_positive_rate=1.0, false_positive_rate=0.0, flagged=1)),
        positives=1)
    success = run_attempt(:boxplot, prepared, config, nothing;
        fit_function=(args...) -> :fit, metric_function=(args...) -> valid, tuning_function=fit -> "settings")
    @test success.status == "success"
    @test success.metrics == valid.metrics && success.positives == 1 && success.settings == "settings"
    # Every Exception is a retained, counted failure, including programming errors.
    for exception in (ArgumentError("bad fit"), DomainError(-1), BoundsError(), ErrorException("fit failed"),
            StudyTestException())
        fit_failure = run_attempt(:boxplot, prepared, config, nothing; fit_function=(args...) -> throw(exception))
        @test fit_failure.status == "failure" && fit_failure.stage == "fit"
        @test fit_failure.failure === exception
        @test all(isnothing, fit_failure.metrics) && fit_failure.positives === nothing
        @test fit_failure.measured.time >= 0
        for stage in (:metrics, :tuning)
            metric_failure = run_attempt(:boxplot, prepared, config, nothing; fit_function=(args...) -> :fit,
                metric_function=(args...) -> stage == :metrics ? throw(exception) : valid,
                tuning_function=fit -> throw(exception))
            @test metric_failure.status == "failure" && metric_failure.stage == "metrics"
            @test all(isnothing, metric_failure.metrics) && isempty(metric_failure.settings)
        end
    end
    for exception in (InterruptException(), OutOfMemoryError())
        @test !S.fit_failure(exception)
        @test_throws typeof(exception) run_attempt(:boxplot, prepared, config, nothing;
            fit_function=(args...) -> throw(exception))
    end
end

@testset "CSV cells and aggregates" begin
    row = ("plain", "with, comma", "with \"quotes\"", "line\nbreak", nothing, 1.5, :symbol)
    line = sprint(S.csvrow, row)
    @test S.parse_csvrow(chomp(line)) == ["plain", "with, comma", "with \"quotes\"", "line\\nbreak", "", "1.5", "symbol"]
    for bad in ("plain", "\"open", "\"a\"x", "\"a\",")
        @test_throws ArgumentError S.parse_csvrow(bad)
    end
    mktemp() do path, io
        S.csvrow(io, S.RAW_HEADER)
        S.csvrow(io, (1, 11, :boxplot, "success", 0.5, 100, 7, 0.5, 0.25, 3, nothing, 2, "a=1"))
        S.csvrow(io, (2, 12, :boxplot, "failure", 1.5, 300, 9, nothing, nothing, nothing, nothing, nothing, ""))
        S.csvrow(io, (1, 11, :msplot, "success", 9.0, 9, 99, 1.0, 0.0, 2, nothing, 2, "b=1"))
        close(io)
        header, values = S.aggregates(path, :boxplot)
        result = Dict(zip(header, values))
        @test result["boxplot_successes"] == 1 && result["boxplot_failures"] == 1
        @test result["boxplot_process_peak_rss_bytes"] == 9
        @test result["boxplot_wall_seconds_count"] == 2 && result["boxplot_wall_seconds_mean"] == 1.0
        @test result["boxplot_true_positive_rate_count"] == 1 && result["boxplot_true_positive_rate_mean"] == 0.5
        @test result["boxplot_adjusted_rand_index_count"] == 0 && result["boxplot_adjusted_rand_index_mean"] === nothing
    end
end

@testset "Work units" begin
    mktempdir() do dir
        rows = [values(config_row(replicates=23)),
            values(config_row(id=2, study=:clustering, scenario=:mean, design=:sparse, contamination=nothing,
                groups=2, replicates=7))]
        path = write_config(joinpath(dir, "configurations.csv"), rows)
        configs = read_configurations(path)
        units = work_units(configs, 10)
        @test [(u.first, u.last) for u in units if u.cell == 1 && u.route == :msplot] == [(1, 10), (11, 20), (21, 23)]
        @test first(units).output == "cell-001-boxplot-0001-0010.csv"
        @test length(units) == 3 * 3 + 3 * 1
        @test !any(u -> u.route == :kmeans_curves, units)
        list = joinpath(dir, "units.tsv")
        @test write_work_units(list, units) == length(units)
        @test check_work_units(list, configs) == length(units)
        @test_throws ArgumentError write_work_units(list, units)
        @test_throws ArgumentError work_units(configs, 0)
        lines = readlines(list)
        for edit in (l -> l[1:end-1],                                    # drop the last unit
                l -> vcat(l, l[end]),                                    # duplicate a unit
                l -> vcat(l[1], replace(l[2], "0010" => "0011")),        # noncanonical name
                l -> vcat(l[1], "2\tkmeans_curves\t1\t7\tcell-002-kmeans_curves-0001-0007.csv", l[2:end]),
                l -> vcat(l[1], "3\tboxplot\t1\t1\tcell-003-boxplot-0001-0001.csv", l[2:end]),
                l -> vcat("cell\troute", l[2:end]))
            bad = joinpath(dir, "bad.tsv")
            write(bad, join(edit(lines), '\n') * "\n")
            @test_throws ArgumentError check_work_units(bad, configs)
        end
    end
end

# Writes a finalized chunk from replicate records, as run.jl does.
function write_chunk(dir, config, lo, hi, route, records, failures; versions=(VERSION, "0.1.0", "0.1.0"),
    elapsed=1.0)
    routes = route isa Symbol ? (route,) : route
    tag = route isa Symbol ? "-$(route)-$(lpad(lo, 4, '0'))-$(lpad(hi, 4, '0'))" : ""
    name = joinpath(dir, "cell-$(lpad(config.id, 3, '0'))$tag")
    open(name * ".replicates.csv", "w") do io
        S.csvrow(io, S.RAW_HEADER)
        foreach(r -> S.csvrow(io, r), records)
    end
    open(name * ".failures.csv", "w") do io
        S.csvrow(io, S.FAILURE_HEADER)
        foreach(r -> S.csvrow(io, r), failures)
    end
    aggregates = [S.aggregates(name * ".replicates.csv", r) for r in routes]
    open(name * ".csv", "w") do io
        S.csvrow(io, vcat(collect(S.CONFIG_HEADER), ["first_replicate", "last_replicate", "routes",
            "julia_version", "fda_version", "kernel_smoothing_version", "elapsed_seconds"], first.(aggregates)...))
        S.csvrow(io, vcat(collect(values(config)), [lo, hi, join(routes, ';'), versions..., elapsed],
            last.(aggregates)...))
    end
    name
end

@testset "Cost estimate" begin
    mktempdir() do dir
        outlier = config_row(n=100, scenario=:model1)
        cluster = config_row(id=2, study=:clustering, scenario=:covariance, design=:sparse, n=100,
            contamination=nothing, groups=3)
        configs = read_configurations(write_config(joinpath(dir, "probe.csv"), [values(outlier), values(cluster)]))
        record(config, r, route, seconds) = (r, config.seed + r - 1, route, "success", seconds, 10, 5,
            (config.study == :outlier ? (1.0, 0.0, 4, nothing) : (nothing, nothing, nothing, 0.5))...,
            config.study == :outlier ? 10 : nothing, "status=converged")
        # Fit seconds: boxplot 1 and 3 (mean 2), outliergram 4, msplot 6; kmeans 1, kcfc 10, mixture 3.
        times = Dict(:boxplot => (1.0, 3.0), :outliergram => (4.0, 4.0), :msplot => (6.0, 6.0),
            :kmeans_scores => (1.0, 1.0), :kcfc => (10.0, 10.0), :mixture => (3.0, 3.0))
        # Elapsed = fit time + 0.5 s per replicate outside the fit + 5 s startup per route process.
        for config in configs
            rs = S.routes(config)
            write_chunk(dir, config, 1, 2, rs, [record(config, r, route, times[route][r]) for route in rs for r in 1:2],
                []; elapsed=sum(sum(times[route]) for route in rs) + length(rs) * (2 * 0.5 + 5))
        end
        half(config) = 0.5
        timings = probe_timings(configs, dir; replicate_seconds=half)
        @test timings.startup ≈ 5
        @test timings.per_replicate == Dict((:outlier, :dense, 100) => 0.5, (:clustering, :sparse, 100) => 0.5)
        @test timings.fit_seconds[(:outlier, :dense, 100, :boxplot)] ≈ 2
        @test timings.fit_seconds[(:clustering, :sparse, 100, :kcfc)] ≈ 10
        full = [merge(c, (; replicates=7)) for c in configs]
        result = estimate(full, configs, dir; block_size=5, replicate_seconds=half)
        @test length(result.units) == 2 * 6
        @test sum(result.seconds) ≈ 12 * 5 + 7 * (2 + 4 + 6 + 1 + 10 + 3) + 6 * 7 * 0.5
        @test maximum(result.seconds) ≈ 5 + 5 * (10 + 0.5)
        @test occursin("Work units: 12", sprint(write_estimate, full, result))
        @test_throws ArgumentError estimate([merge(c, (; n=500)) for c in full], configs, dir; replicate_seconds=half)
        @test_throws ArgumentError probe_timings(configs, mktempdir(); replicate_seconds=half)
        @test 0 < replicate_overhead(outlier; replicates=1) < 60
    end
end

@testset "Summary of smoke and chunked results" begin
    # The committed smoke run is complete and its summary reproduces.
    mktempdir() do dir
        rows = write_summary(joinpath(SMOKE, "configurations.csv"), SMOKE, joinpath(dir, "summary"))
        @test length(rows) == 3 + 4
        @test read(joinpath(dir, "summary.csv"), String) == read(joinpath(SMOKE, "summary.csv"), String)
        @test read(joinpath(dir, "summary.md"), String) == read(joinpath(SMOKE, "summary.md"), String)
        @test_throws ArgumentError write_summary(joinpath(SMOKE, "configurations.csv"), SMOKE, joinpath(dir, "summary"))
    end
    # Chunks of one cell merge; a counted failure stays a failure.
    mktempdir() do dir
        config = config_row(study=:clustering, scenario=:mean, design=:sparse, contamination=nothing,
            groups=2, replicates=3)
        path = write_config(joinpath(dir, "configurations.csv"), [values(config)])
        configs = read_configurations(path)
        ok(r, route, ari) = (r, config.seed + r - 1, route, "success", 0.1, 10, 5, nothing, nothing, nothing, ari,
            nothing, "status=converged")
        bad(r, route) = (r, config.seed + r - 1, route, "failure", 0.2, 20, 6, nothing, nothing, nothing, nothing,
            nothing, "")
        for route in (:kmeans_scores, :mixture)
            write_chunk(dir, config, 1, 3, route, [ok(r, route, 0.5) for r in 1:3], [])
        end
        write_chunk(dir, config, 1, 2, :kcfc, [ok(1, :kcfc, 0.2), bad(2, :kcfc)],
            [(2, config.seed + 1, :kcfc, "fit", "ArgumentError", "tiny initial cluster")])
        final_chunk = write_chunk(dir, config, 3, 3, :kcfc,
            [(3, config.seed + 2, :kcfc, "success", 0.1, 10, 5, nothing, nothing, nothing, 0.4, nothing,
                "status=tiny_cluster")], [])
        rows = summarize(configs, dir)
        kcfc = only(filter(r -> r["route"] == "kcfc", rows))
        @test kcfc["successes"] == 2 && kcfc["failures"] == 1
        @test kcfc["adjusted_rand_index_count"] == 2 && kcfc["adjusted_rand_index_mean"] ≈ 0.3
        @test kcfc["wall_seconds_count"] == 3 && kcfc["nonconverged"] == 1
        @test kcfc["statuses"] == "converged=1;tiny_cluster=1"
        # Adverse result directories are rejected.
        function rejects(edit)
            copy_dir = mktempdir()
            for f in readdir(dir)
                cp(joinpath(dir, f), joinpath(copy_dir, f))
            end
            edit(copy_dir)
            try
                summarize(configs, copy_dir)
                false
            catch err
                err isa ArgumentError
            end
        end
        @test !rejects(identity)
        stem = basename(final_chunk)
        edit_lines(file, f) = d -> write(joinpath(d, file), join(f(readlines(joinpath(d, file))), '\n') * "\n")
        edit_record(file, pattern) = edit_lines(file, l -> [l[1:end-1]; replace(l[end], pattern; count=1)])
        seed3 = string(config.seed + 2)
        @test rejects(d -> rm(joinpath(d, stem * ".csv")))                                   # missing chunk
        @test rejects(edit_lines(stem * ".replicates.csv", l -> l[1:1]))                     # missing record
        @test rejects(edit_lines(stem * ".replicates.csv", l -> vcat(l, l[2])))               # duplicate record
        @test rejects(edit_record(stem * ".replicates.csv", "\"$seed3\"" => "\"1\""))       # wrong seed
        @test rejects(edit_record(stem * ".replicates.csv", "\"0.4\"" => "\"1.4\""))        # ARI out of range
        @test rejects(edit_record(stem * ".replicates.csv", "\"0.4\"" => "\"0.3\""))        # aggregate mismatch
        @test rejects(edit_record(stem * ".replicates.csv", "\"status=tiny_cluster\"" => "\"\""))
        @test rejects(edit_record(stem * ".csv", "\"0.1.0\"" => "\"0.2.0\""))               # versions differ
        @test rejects(edit_record(stem * ".csv", "\"mean\"" => "\"covariance\""))           # wrong configuration
        @test rejects(edit_lines("cell-001-kcfc-0001-0002.failures.csv", l -> l[1:1]))       # undiagnosed failure
        @test rejects(edit_record("cell-001-kcfc-0001-0002.failures.csv", "\"fit\"" => "\"plot\""))
        @test rejects(edit_record("cell-001-kcfc-0001-0002.replicates.csv",                  # failure with metrics
            "\"\",\"\",\"\",\"\"" => "\"\",\"\",\"\",\"0.5\""))
        @test rejects(d -> write_chunk(d, config, 2, 3, :kcfc, [ok(2, :kcfc, 0.4), ok(3, :kcfc, 0.4)], []))
        @test rejects(d -> write_chunk(d, config, 1, 1, :kmeans_curves, [ok(1, :kmeans_curves, 0.4)], []))
        @test rejects(d -> write_chunk(d, config, 4, 4, :kcfc, [ok(4, :kcfc, 0.4)], []))
    end
end
