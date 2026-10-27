using DelimitedFiles: readdlm
import RecipesBase
import StatsAPI

const _OUTLIER_INPUT = readdlm(joinpath(@__DIR__, "data", "m8_outliers_input.csv"), ',', Any; comments=true)
const _OUTLIER_GOLDEN = readdlm(joinpath(@__DIR__, "data", "m8_outliers_golden.csv"), ',', Any; comments=true)

function _outlier_input(name)
    rows = _OUTLIER_INPUT[_OUTLIER_INPUT[:, 1] .== name, :]
    out = zeros(Int(maximum(rows[:, 2])), Int(maximum(rows[:, 3])))
    for row in eachrow(rows)
        out[Int(row[2]), Int(row[3])] = Float64(row[4])
    end
    out
end
# Empty results are stored as one row with index 0.
function _outlier_golden(case, quantity)
    rows = _OUTLIER_GOLDEN[(_OUTLIER_GOLDEN[:, 1] .== case) .& (_OUTLIER_GOLDEN[:, 2] .== quantity), :]
    isempty(rows) && error("missing golden $case/$quantity")
    Float64.(rows[rows[:, 3] .> 0, 4])
end
_outlier_indices(case, quantity) = Int.(_outlier_golden(case, quantity))
_outlier_close(actual, case, quantity) =
    golden_relative_error(vec(actual), _outlier_golden(case, quantity)) < 1e-8

# Inspect recipe series without a plotting backend, as test/plotting.jl does.
function _outlier_series(args...; attrs=Dict{Symbol,Any}())
    result = RecipesBase.RecipeData[]
    function expand(attributes, arguments)
        if !isempty(arguments) && applicable(RecipesBase.apply_recipe, attributes, arguments...)
            for next in RecipesBase.apply_recipe(attributes, arguments...)
                expand(next.plotattributes, next.args)
            end
        else
            push!(result, RecipesBase.RecipeData(attributes, arguments))
        end
    end
    expand(attrs, args)
    result
end

const _OUTLIER_GRID = vec(_outlier_input("grid"))
_outlier_fd(name) = GridFunctionalData(_outlier_input(name), _OUTLIER_GRID)

@testset "functional boxplot goldens against roahd 1.4.3 and fda 6.3.0" begin
    fd = _outlier_fd("uni")
    roahd = ModifiedBandDepth(integration=:uniform)
    b = functional_boxplot(fd; method=roahd, central_ties=:all)
    @test _outlier_close(b.depth.values, "uni", "fbplot_depth")
    @test b.outliers == _outlier_indices("uni", "fbplot_outliers")
    region = only(b.regions)
    @test _outlier_close(region.central_lower, "uni", "fbplot_central_lower")
    @test _outlier_close(region.central_upper, "uni", "fbplot_central_upper")
    @test _outlier_close(region.fence_lower, "uni", "fbplot_fence_lower")
    @test _outlier_close(region.fence_upper, "uni", "fbplot_fence_upper")
    fda = functional_boxplot(fd; method=ModifiedBandDepth(ties=:average, integration=:uniform))
    @test _outlier_close(fda.depth.values, "uni", "fda_depth")
    @test fda.outliers == _outlier_indices("uni", "fda_outliers")
    @test fda.median == only(_outlier_indices("uni", "fda_median"))

    tied = _outlier_fd("tied")
    bt = functional_boxplot(tied; method=roahd, central_ties=:all)
    @test length(bt.central) == only(_outlier_indices("tied", "fbplot_central_size"))
    @test length(bt.central) > cld(nobs(bt), 2)
    @test length(functional_boxplot(tied; method=roahd).central) == cld(nobs(bt), 2)
    @test bt.outliers == _outlier_indices("tied", "fbplot_outliers")
    @test _outlier_close(only(bt.regions).central_lower, "tied", "fbplot_central_lower")
    @test _outlier_close(only(bt.regions).central_upper, "tied", "fbplot_central_upper")
    @test functional_boxplot(tied; method=ModifiedBandDepth(ties=:average, integration=:uniform)).outliers ==
        _outlier_indices("tied", "fda_outliers")

    mfd = MultivariateFunctionalData([_outlier_fd("mv1"), _outlier_fd("mv2")])
    bm = functional_boxplot(mfd; method=roahd, central_ties=:all)
    @test _outlier_close(bm.depth.values, "mv", "fbplot_depth")
    @test bm.outliers == _outlier_indices("mv", "fbplot_outliers")
    for j in 1:2
        @test _outlier_close(bm.regions[j].central_lower, "mv", "fbplot_central_lower$j")
        @test _outlier_close(bm.regions[j].central_upper, "mv", "fbplot_central_upper$j")
    end
end

@testset "outliergram goldens against roahd 1.4.3" begin
    fd = _outlier_fd("uni")
    o = outliergram(fd; method=ModifiedBandDepth(ties=:average, integration=:uniform))
    @test _outlier_close(o.mei, "uni", "outliergram_mei")
    @test _outlier_close(o.mbd, "uni", "outliergram_mbd")
    @test _outlier_close(o.distance, "uni", "outliergram_d")
    @test _outlier_close(collect(o.quartiles), "uni", "outliergram_quartiles")
    @test o.flagged == _outlier_indices("uni", "outliergram_flagged")
    @test o.outliers == _outlier_indices("uni", "outliergram_outliers")
    @test !isempty(o.shifted) && o.shifted == setdiff(o.outliers, o.flagged)
    @test outliergram(fd; method=ModifiedBandDepth(ties=:average, integration=:uniform), shift=false).outliers ==
        o.flagged
end

@testset "magnitude-shape goldens against fdaoutlier 0.2.1 and MASS 7.3-65" begin
    m = length(_OUTLIER_GRID)
    for case in ("uni", "uni2")
        fd = _outlier_fd(case == "uni" ? "uni" : "mv2")
        cutoff = only(_outlier_golden(case, "ms_cutoff"))
        s = magnitude_shape(fd; definition=:fdaoutlier, cutoff)
        @test _outlier_close(s.mean_outlyingness, case, "ms_mo")
        @test _outlier_close(s.variation_outlyingness, case, "ms_vo")
        @test _outlier_close(s.distance, case, "ms_distance")
        @test s.settings.mcd.subset == _outlier_indices(case, "ms_best")
        @test s.settings.mcd.exact && s.settings.estimate == :reweighted
        @test s.outliers == _outlier_indices(case, "ms_outliers")
        @test s.median == only(_outlier_indices(case, "ms_median"))
        # The package definition with equal weights has the same MO and the
        # population variance over the grid; the MCD subset is unchanged.
        d = magnitude_shape(fd; integration=:uniform)
        @test _outlier_close(d.mean_outlyingness, case, "ms_mo")
        @test _outlier_close(d.variation_outlyingness .* m ./ (m - 1), case, "ms_vo")
        @test d.settings.mcd.subset == s.settings.mcd.subset && d.settings.estimate == :raw
    end
    s = magnitude_shape(_outlier_fd("uni"); definition=:fdaoutlier, cutoff=1.0)
    @test _outlier_close(s.center, "uni", "ms_center")
    @test _outlier_close(s.covariance, "uni", "ms_cov")
    raw = magnitude_shape(_outlier_fd("uni"); integration=:uniform)
    @test _outlier_close(raw.center .* [1, m / (m - 1)], "uni", "ms_raw_center")
    @test _outlier_close(raw.covariance .* [1 m/(m-1); m/(m-1) (m/(m-1))^2], "uni", "ms_raw_cov")
    # The flagged magnitude outlier of the second component under the default rule.
    @test 3 in magnitude_shape(_outlier_fd("mv2")).outliers

    mfd = MultivariateFunctionalData([_outlier_fd("mv1"), _outlier_fd("mv2")])
    directions = _outlier_input("directions")
    cutoff = only(_outlier_golden("mv", "ms_cutoff"))
    sm = magnitude_shape(mfd; definition=:fdaoutlier, directions, cutoff)
    @test _outlier_close(sm.mean_outlyingness, "mv", "ms_mo")
    @test _outlier_close(sm.variation_outlyingness, "mv", "ms_vo")
    @test _outlier_close(sm.distance, "mv", "ms_distance")
    @test _outlier_close(sm.center, "mv", "ms_center")
    @test _outlier_close(sm.covariance, "mv", "ms_cov")
    @test sm.settings.mcd.subset == _outlier_indices("mv", "ms_best")
    @test sm.outliers == _outlier_indices("mv", "ms_outliers")
    @test sm.median == only(_outlier_indices("mv", "ms_median"))
    @test sm.settings.directions ≈ directions ./ sqrt.(sum(abs2, directions; dims=2))
    @test !sm.settings.generated && sm.settings.nproj == 50

    # Hardin-Rocke: fdaoutlier's (deterministic) F quantile and the exact c.
    for (n, q) in ((24, 2), (20, 3), (100, 2), (1000, 3))
        hr = FunctionalDataAnalysis._hardin_rocke(n, q)
        @test _outlier_close([hr.quantile], "hardin", "quantile_$(n)_$(q)")
        @test _outlier_close([hr.c], "hardin", "c_$(n)_$(q)")
        @test hr.cutoff ≈ hr.quantile / (hr.c * (hr.m - q + 1) / (q * hr.m))
    end
end

@testset "adjusted boxplot factor against roahd 1.4.3 and robustbase 0.99-7" begin
    x = _outlier_input("uni")
    # robustbase's Qn may return its single-precision search value; this
    # package keeps the exact order statistic (same constants).
    vectors = (x[:, 1], x[1:11, 2], vec(_outlier_input("qn_column")))
    qn = [FunctionalDataAnalysis._qn_scale(v) for v in vectors]
    @test _outlier_close(qn, "adjust", "qn_exact")
    robust = _outlier_golden("adjust", "qn_robustbase")
    @test qn[1:2] == robust[1:2]
    @test 0 < abs(robust[3] - qn[3]) / qn[3] <= 2.0^-23
    @test _outlier_close(FunctionalDataAnalysis._ogk_covariance(x), "adjust", "cov_ogk_exact_qn")
    rate = only(_outlier_golden("adjust", "rate"))
    @test AdjustedFactor().rate ≈ rate rtol=1e-12
    # One trial on R's simulated sample: roahd's uniroot stops within its
    # tolerance of the exact factor, and both flag the same curves.
    sim = _outlier_input("sim")
    method = ModifiedBandDepth(ties=:average, integration=:uniform)
    values = depth(GridFunctionalData(sim, _OUTLIER_GRID), method).values
    exact = FunctionalDataAnalysis._boxplot_trial_factor(sim, values, sortperm(values; rev=true), 0.5, :all, rate)
    root, tol = only(_outlier_golden("adjust", "trial_root")), only(_outlier_golden("adjust", "trial_tol"))
    @test abs(exact - root) <= tol
    flagged(f) = length(functional_boxplot(GridFunctionalData(sim, _OUTLIER_GRID); method, central_ties=:all,
        factor=f).outliers)
    @test flagged(exact) == flagged(root) == only(_outlier_indices("adjust", "trial_flagged"))
    @test flagged(exact) <= rate * size(sim, 1) < flagged(prevfloat(exact))
    # The full adjustment averages the trial factors and records them.
    b = functional_boxplot(GridFunctionalData(x, _OUTLIER_GRID); method, central_ties=:all,
        factor=AdjustedFactor(ntrials=3, trial_size=60, rng=Random.Xoshiro(1)))
    @test b.settings.factor_rule == :adjusted && length(b.settings.trial_factors) == 3
    @test b.factor ≈ sum(b.settings.trial_factors) / 3
    @test b.settings.trial_size == 60 && occursin("adjusted", sprint(show, b))
end

@testset "outlier displays: analytic values" begin
    grid = [0.0, 0.5, 1.0]
    levels = GridFunctionalData(repeat([1.0:9.0; 20.0], 1, 3), grid)
    b = functional_boxplot(levels)
    @test sort(b.central) == [3, 4, 5, 6, 7] && b.median == 5
    r = only(b.regions)
    @test r.central_lower == fill(3.0, 3) && r.central_upper == fill(7.0, 3)
    @test r.fence_lower == fill(-3.0, 3) && r.fence_upper == fill(13.0, 3)
    @test b.outliers == [10]
    @test r.whisker_lower == fill(1.0, 3) && r.whisker_upper == fill(9.0, 3)
    # Ranks 3 and 8 tie in depth: :all takes both.
    ball = functional_boxplot(levels; central_ties=:all)
    @test sort(ball.central) == 3:8 && only(ball.regions).fence_upper == fill(15.5, 3)
    @test functional_boxplot(levels; factor=0).outliers == [1, 2, 8, 9, 10]
    @test functional_boxplot(levels; central=0.9).outliers == Int[]

    # Curves that never cross lie on the outliergram parabola.
    n = 10
    o = outliergram(levels)
    @test maximum(abs, o.distance) < 1e-14
    @test o.mei ≈ [(n - r + 1) / n for r in 1:n]
    @test o.coefficients[3] ≈ n^2 * o.coefficients[1] && o.coefficients[2] ≈ 2 * (n + 1) / (n - 1)
    @test o.threshold == o.quartiles[2] + 1.5 * (o.quartiles[2] - o.quartiles[1])

    # Univariate directional outlyingness (x - median) / mad and its integrals.
    rng = Random.Xoshiro(84)
    g = collect(range(0, 1; length=9)) .^ 2
    x = [sin(3t + a) + a for a in randn(rng, 15), t in g] .+ 0.1 .* randn(rng, 15, 9)
    fd = GridFunctionalData(x, g)
    s = magnitude_shape(fd)
    med = [FunctionalDataAnalysis._r_median(x[:, t]) for t in 1:9]
    mad = [1.4826 * FunctionalDataAnalysis._r_median(abs.(x[:, t] .- med[t])) for t in 1:9]
    o_expected = (x .- transpose(med)) ./ transpose(mad)
    @test s.outlyingness[:, :, 1] ≈ o_expected
    w = quadrature_weights(g) ./ sum(quadrature_weights(g))
    @test s.mean_outlyingness[:, 1] ≈ o_expected * w
    @test s.variation_outlyingness ≈ (abs2.(o_expected .- s.mean_outlyingness[:, 1]) * w)
    @test s.settings.integration == :quadrature && s.settings.estimate == :raw
    @test s.median == argmin(s.distance) && s.outliers == findall(>(s.cutoff), s.distance)
    @test s.cutoff == FunctionalDataAnalysis._hardin_rocke(15, 2).cutoff

    # Two-sided projection outlyingness along the axes is the larger axis outlyingness.
    z = randn(rng, 11, 2)
    axes2 = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
    sdo = FunctionalDataAnalysis._ms_projection(z, axes2, :dai_genton)
    by_axis = [abs.(z[:, j] .- FunctionalDataAnalysis._r_median(z[:, j])) ./
        FunctionalDataAnalysis._r_mad(z[:, j]) for j in 1:2]
    @test sdo ≈ max.(by_axis...)
    # fdaoutlier's one-sided version only looks in the given directions.
    center = sort(z[:, 1])[6]
    @test FunctionalDataAnalysis._ms_projection(z, axes2[[1], :], :fdaoutlier) ≈
        (z[:, 1] .- center) ./ sort(abs.(z[:, 1] .- center))[6]

    # MASS's subsets are enumerated in lexicographic order.
    subsets = [[0, 1, 2]]
    for _ in 2:binomial(5, 3)
        push!(subsets, FunctionalDataAnalysis._mcd_next!(copy(subsets[end]), 5, 3))
    end
    @test subsets == [[i, j, k] for i in 0:4 for j in i+1:4 for k in j+1:4]
end

@testset "outlier displays: representations, settings and display" begin
    rng = Random.Xoshiro(85)
    grid = collect(range(0, 1; length=21))
    values = [a + b * sin(2pi * t) for (a, b) in zip(randn(rng, 16), randn(rng, 16)), t in grid] .+
        0.05 .* randn(rng, 16, 21)
    values[2, :] .+= 4
    fd = GridFunctionalData(values, grid; ids=string.('a':'p'))
    b = functional_boxplot(fd)
    @test b.depth.ids == string.('a':'p') && 2 in b.outliers && nobs(b) == 16
    @test b.settings.central == 0.5 && b.settings.central_ties == :first && b.settings.factor_rule == :fixed
    shown = sprint(show, b)
    @test occursin("FunctionalBoxplot", shown) && occursin("outliers=", shown)
    @test sprint(show, MIME"text/plain"(), b) == shown
    observed = [sort(randperm(rng, 21)[1:9]) for _ in 1:16]
    irregular = IrregularFunctionalData([grid[k] for k in observed], [values[i, observed[i]] for i in 1:16])
    bi = functional_boxplot(irregular; fpca=(; ncomponents=2))
    @test bi.depth.representation == :fpca_reconstruction && bi.depth.fpca isa FPCA
    oi = outliergram(irregular; fpca=(; ncomponents=2))
    @test oi.representation == :fpca_reconstruction && oi.fpca isa FPCA
    @test size(oi.values) == (16, length(oi.grid))
    si = magnitude_shape(irregular; fpca=(; ncomponents=2))
    @test si.representation == :fpca_reconstruction
    basis = BasisFunctionalData(randn(rng, 12, 3), FourierBasis((0.0, 1.0), 3))
    ob = outliergram(basis; grid)
    @test ob.representation == :basis_evaluated && ob.grid == grid
    @test ob.mei ≈ modified_epigraph_index(GridFunctionalData(evaluate(basis, grid), grid))
    o = outliergram(fd)
    @test occursin("Outliergram", sprint(show, o)) && nobs(o) == 16
    @test o.settings.p_check == 0.05 && o.settings.shift && o.settings.integration == :quadrature
    s = magnitude_shape(fd)
    @test occursin("MagnitudeShape", sprint(show, s)) && nobs(s) == 16 && argmax(s.distance) == 2
    @test s.settings.mcd.exact && s.settings.mcd.nsamp == binomial(16, 3)
    # Sampled MCD subsets follow rng.
    s1 = magnitude_shape(fd; nsamp=40, rng=Random.Xoshiro(1))
    s2 = magnitude_shape(fd; nsamp=40, rng=Random.Xoshiro(1))
    @test s1.distance == s2.distance && !s1.settings.mcd.exact && s1.settings.mcd.nsamp == 40
    # Multivariate: generated directions are recorded and unit length.
    mfd = MultivariateFunctionalData([fd, GridFunctionalData(values .^ 2, grid)])
    sm = magnitude_shape(mfd; nproj=30, rng=Random.Xoshiro(2))
    @test size(sm.settings.directions) == (30, 2) && sm.settings.generated
    @test all(r -> sum(abs2, r) ≈ 1, eachrow(sm.settings.directions))
    @test size(sm.mean_outlyingness) == (16, 2) && size(sm.outlyingness) == (16, 21, 2)
    @test magnitude_shape(mfd; nproj=30, rng=Random.Xoshiro(2)).distance == sm.distance
    bm = functional_boxplot(mfd; weights=[0.25, 0.75])
    @test length(bm.regions) == 2 && bm.depth.settings.component_weights == [0.25, 0.75]
end

@testset "outlier displays: plot data and recipes" begin
    rng = Random.Xoshiro(86)
    grid = collect(range(0, 1; length=15))
    values = [a + b * cos(2pi * t) for (a, b) in zip(randn(rng, 14), randn(rng, 14)), t in grid] .+
        0.05 .* randn(rng, 14, 15)
    values[4, :] .+= 5
    fd = GridFunctionalData(values, grid)
    b = functional_boxplot(fd)
    d = boxplot_data(b)
    @test d.outliers == b.outliers == [4] && d.values == values && d.values !== only(b.regions).values
    series = _outlier_series(b)
    @test length(series) == 5 + length(b.outliers)
    @test series[1].plotattributes[:fillrange] == d.central_lower && series[1].args == (grid, d.central_upper)
    @test series[3].args == (grid, values[b.median, :])
    @test series[end].args == (grid, values[4, :]) && series[end].plotattributes[:linestyle] == :dash
    custom = _outlier_series(b; attrs=Dict{Symbol,Any}(:title => "Custom"))
    @test all(s -> s.plotattributes[:title] == "Custom", custom)
    mfd = MultivariateFunctionalData([fd, fd])
    panels = _outlier_series(functional_boxplot(mfd))
    @test [s.plotattributes[:subplot] for s in panels[[1, end]]] == [1, 2]
    @test haskey(boxplot_data(functional_boxplot(mfd)), :nodes)

    o = outliergram(fd)
    od = outliergram_data(o)
    @test od.mei == o.mei && od.mei !== o.mei && length(od.parabola.mei) == 101
    @test od.boundary.mbd ≈ od.parabola.mbd .- o.threshold
    @test od.parabola.mbd[end] ≈ sum(o.coefficients)
    og = _outlier_series(o)
    @test og[end].args == (od.boundary.mei, od.boundary.mbd)
    @test length(og) == 3 + !isempty(o.outliers)

    s = magnitude_shape(fd)
    sd = magnitude_shape_data(s)
    @test sd.magnitude == s.mean_outlyingness[:, 1] && sd.variation == s.variation_outlyingness
    e = sd.ellipse
    @test FunctionalDataAnalysis._mahalanobis(hcat(e.magnitude, e.variation), s.center, s.covariance) ≈
        fill(s.cutoff, 101)
    ms = _outlier_series(s)
    @test ms[end].args == (e.magnitude, e.variation)
    other = GridFunctionalData(values .^ 2 .+ 0.1 .* randn(rng, size(values)), grid)
    sm = magnitude_shape(MultivariateFunctionalData([fd, other]); nproj=20, rng=Random.Xoshiro(1))
    smd = magnitude_shape_data(sm)
    @test smd.ellipse === nothing && smd.multivariate
    @test smd.magnitude ≈ sqrt.(vec(sum(abs2, sm.mean_outlyingness; dims=2)))
    @test _outlier_series(sm)[1].plotattributes[:xlabel] == "‖MO‖"

    # outlier_plot hands the prepared payload to RecipesBase.plot (no backend loaded).
    for (result, kind, data) in ((o, :outliergram, od), (s, :magnitude_shape, sd), (b, :boxplot, d))
        failure = try
            outlier_plot(result; color=:blue)
            nothing
        catch error
            error
        end
        @test failure isa MethodError && failure.args[2] === RecipesBase.plot
        @test last(failure.args) isa FunctionalDataAnalysis._FDAPlot{kind}
        @test isequal(last(failure.args).data, data)
    end
    @test !isdefined(Main, :Plots)
end

@testset "outlier displays: error paths" begin
    rng = Random.Xoshiro(87)
    grid = collect(range(0, 1; length=8))
    values = randn(rng, 12, 8)
    fd = GridFunctionalData(values, grid)
    mfd = MultivariateFunctionalData([fd, GridFunctionalData(randn(rng, 12, 8), grid)])
    for bad in (0, 1, 1.5, true)
        @test_throws ArgumentError functional_boxplot(fd; central=bad)
    end
    for bad in (-1, NaN, Inf, true)
        @test_throws ArgumentError functional_boxplot(fd; factor=bad)
        @test_throws ArgumentError outliergram(fd; factor=bad)
    end
    @test_throws ArgumentError functional_boxplot(fd; central_ties=:last)
    @test_throws ArgumentError functional_boxplot(mfd; factor=AdjustedFactor())
    @test_throws ArgumentError functional_boxplot(1.0)
    @test_throws ArgumentError AdjustedFactor(ntrials=0)
    @test_throws ArgumentError AdjustedFactor(trial_size=2)
    @test_throws ArgumentError AdjustedFactor(rate=1)
    flat = GridFunctionalData(repeat(1.0:12.0, 1, 8), grid)
    @test_throws ArgumentError functional_boxplot(flat; factor=AdjustedFactor())
    # A rate above the non-central share would put the fences inside the central region.
    @test_throws ArgumentError functional_boxplot(fd;
        factor=AdjustedFactor(rate=0.95, ntrials=1, rng=Random.Xoshiro(1)))
    @test_throws ArgumentError outliergram(mfd)
    @test_throws ArgumentError outliergram(fd[1:2])
    @test_throws ArgumentError outliergram(fd; p_check=0)
    @test_throws ArgumentError outliergram(fd; p_check=1)
    @test_throws ArgumentError magnitude_shape(fd; definition=:other)
    @test_throws ArgumentError magnitude_shape(fd; definition=:fdaoutlier, integration=:quadrature)
    @test_throws ArgumentError magnitude_shape(fd; integration=:simpson)
    for bad in (-1, 0, :chisq, Inf, true)
        @test_throws ArgumentError magnitude_shape(fd; cutoff=bad)
    end
    @test_throws ArgumentError magnitude_shape(fd; nproj=10)
    @test_throws ArgumentError magnitude_shape(fd; directions=[1.0 0.0])
    @test_throws ArgumentError magnitude_shape(fd; nsamp=0)
    @test_throws ArgumentError magnitude_shape(fd; nsamp=:all)
    @test_throws ArgumentError magnitude_shape(fd[1:4])
    @test_throws ArgumentError FunctionalDataAnalysis._mass_mcd(hcat(1.0:6.0, zeros(6)), :best, rng)  # IQR 0
    tied = copy(values)
    tied[1:7, 3] .= 0.0
    @test_throws ArgumentError magnitude_shape(GridFunctionalData(tied, grid))  # zero MAD
    @test_throws DimensionMismatch magnitude_shape(mfd; directions=ones(3, 3))
    @test_throws DimensionMismatch magnitude_shape(mfd; nproj=4, directions=ones(3, 2))
    @test_throws ArgumentError magnitude_shape(mfd; directions=[1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError magnitude_shape(mfd; directions=[1.0 NaN])
    @test_throws ArgumentError magnitude_shape(mfd; nproj=0)
    @test_throws ArgumentError magnitude_shape(MultivariateFunctionalData([fd, fd]))  # collinear MO
    coarse = GridFunctionalData(randn(rng, 12, 3), [0.0, 0.5, 1.0])
    @test_throws ArgumentError magnitude_shape(MultivariateFunctionalData([fd, coarse]))
    # Exhaustive MCD subsets: small problems enumerate all, large ones refuse.
    @test FunctionalDataAnalysis._mcd_nsamp(:exact, 10, 3) == (true, binomial(10, 3))
    @test_throws ArgumentError FunctionalDataAnalysis._mcd_nsamp(:exact, 200, 6)
end

@testset "outlier displays: show methods" begin
    grid = collect(range(0, 1; length=15))
    rng = Random.Xoshiro(86)
    fd = GridFunctionalData([a + b * sin(2pi * t) for (a, b) in zip(randn(rng, 20), randn(rng, 20)), t in grid], grid)
    @test occursin("AdjustedFactor(ntrials=", sprint(show, AdjustedFactor()))
    o = outliergram(fd)
    @test sprint(show, MIME"text/plain"(), o) == sprint(show, o)
    s = magnitude_shape(fd)
    @test sprint(show, MIME"text/plain"(), s) == sprint(show, s)
end
