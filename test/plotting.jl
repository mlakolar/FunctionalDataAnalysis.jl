using Test, LinearAlgebra, DelimitedFiles
using Random
using FunctionalDataAnalysis
import StatsAPI
import RecipesBase

function plotting_fixture()
    grid = [0.0, 0.1, 0.4, 0.7, 1.0]
    w = quadrature_weights(grid)
    phi = hcat(ones(5), (grid .- 0.5) ./ sqrt(dot(w, (grid .- 0.5).^2)))
    latent = [sqrt(5.0) 0; -sqrt(5.0) 0; 0 sqrt(1.25); 0 -sqrt(1.25); 0 0; 0 0]
    fd = GridFunctionalData(1 .+ grid' .+ latent * phi', grid)
    model = @inferred (data -> StatsAPI.fit(FPCA, data;
        method=:dense, ncomponents=FVE(0.7), max_k=2))(fd)
    times = [[0.0, 0.1, 0.7], [0.1, 0.4, 1.0], [0.0, 0.4, 0.7, 1.0], [0.4]]
    values = [1 .+ t .+ i/4 for (i, t) in enumerate(times)]
    irregular = IrregularFunctionalData(times, values; domain=(0.0, 1.0))
    surface = phi * Diagonal([2.0, 0.5]) * phi'
    pace = @inferred StatsAPI.fit(FPCA, irregular; grid, ncomponents=1, max_k=2,
        mean=(; grid, values=1 .+ grid), covariance=(; grid, surface), noise_variance=0.1)
    (; grid, phi, latent, fd, model, times, values, irregular, pace, surface)
end

# Resolve only our user recipes. This inspects all emitted series without Plots,
# a backend, or test-only definitions of RecipesBase methods.
function plotting_series(args...; attrs=Dict{Symbol,Any}())
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

@testset "Plotting data and recipes (without Plots)" begin
    f = plotting_fixture()
    @testset "Observed and basis curves" begin
        d = @inferred curve_data(f.fd; curves=[3, 1])
        @test d.curves == [3, 1]
        @test d.grid == f.grid && d.grid !== f.fd.grid
        @test d.times == [f.grid, f.grid]
        @test d.values == [f.fd.values[3, :], f.fd.values[1, :]]
        d.values[1][1] = 200
        d.times[1][1] = 200
        @test f.fd.values[3, 1] != 200 && f.fd.grid[1] == 0
        @test d.times[2][1] == 0
        @test curve_data(f.fd; curves=2).curves == [2]
        @test curve_data(f.fd; curves=[true, false, true, false, false, false]).curves == [1, 3]
        masked = GridFunctionalData([1.0 missing 3.0; missing missing missing], [0.0, 0.5, 1.0])
        md = @inferred curve_data(masked)
        @test md.times == [[0.0, 1.0], Float64[]]
        @test md.values == [[1.0, 3.0], Float64[]]
        @test md.design == :sparse
        id = @inferred curve_data(f.irregular)
        @test id.grid === nothing
        @test id.times == f.times && id.values == f.values
        @test id.times[1] !== f.irregular.times
        basis = BasisFunctionalData([1.0 2; -1 3], MonomialBasis((0.0, 1.0), 1))
        bd = @inferred curve_data(basis)
        @test length(bd.grid) == 201 && extrema(bd.grid) == (0, 1)
        @test bd.values[1] ≈ 1 .+ 2bd.grid
        @test bd.values[2] ≈ -1 .+ 3bd.grid
        custom = [0.0, 0.3, 1.0]
        @test only(curve_data(basis; grid=custom, curves=2).values) ≈ -1 .+ 3custom
        @test curve_data(basis; grid=custom).grid !== custom
        mfd = MultivariateFunctionalData([f.fd, to_irregular(f.fd),
            BasisFunctionalData(hcat(collect(1.0:6.0), ones(6)), basis.basis)])
        multi = curve_data(mfd; curves=2:3)
        @test multi.curves == [2, 3] && length(multi.nodes) == 3
        @test multi.nodes[1].values == multi.nodes[2].values
        @test length(multi.nodes[3].grid) == 201
        small = GridFunctionalData(Float32[1 2; 3 4], Float32[0, 1])
        @test eltype((@inferred curve_data(small)).values[1]) == Float32
        for bad in (:none, Int[], [1, 1], [0], [7], true, [1.0])
            @test_throws ArgumentError curve_data(f.fd; curves=bad)
        end
        @test_throws DimensionMismatch curve_data(f.fd; curves=[true])
        @test_throws ArgumentError curve_data(f.fd; grid=f.grid)
        @test_throws ArgumentError curve_data(f.irregular; grid=f.grid)
        for bad in (Float64[], [0.0], [0.5, 0.0], [0.0, NaN], [-0.1, 1.0], :auto)
            @test_throws ArgumentError curve_data(basis; grid=bad)
        end
    end

    @testset "Means, modes, and full-spectrum scree" begin
        mu = @inferred mean_data(f.model)
        @test mu.values ≈ 1 .+ f.grid
        @test mu.values !== f.model.mean.values
        sm = StatsAPI.fit(SmoothedMean, f.fd; grid=f.grid, bandwidth=2.0)
        @test (@inferred mean_data(sm)).values ≈ 1 .+ f.grid
        sd = @inferred scree_data(f.model)
        @test sd.eigenvalues ≈ [2.0, 0.5]
        @test sd.individual ≈ [0.8, 0.2] && sd.cumulative ≈ [0.8, 1.0]
        @test sd.chosen == 1 && sd.threshold == 0.7 && sd.rule == :FVE
        @test sd.eigenvalues !== f.model.all_eigenvalues
        @test scree_data(f.pace).threshold === nothing
        modes = @inferred modes_of_variation(f.model, 1)
        @test modes.multipliers == [-2, -1, 1, 2]
        @test modes.eigenvalue ≈ 2 && modes.fve ≈ 0.8
        @test modes.mean ≈ 1 .+ f.grid
        for (j, c) in enumerate(modes.multipliers)
            @test modes.values[:, j] ≈ 1 .+ f.grid .+ c*sqrt(2)
            @test modes.values[:, j] ≈ mu.values .+
                c*sqrt(eigvals(f.model)[1])*eigenfunctions(f.model).values[:, 1]
        end
        c = [0.0, 0.5]
        custom = modes_of_variation(f.model, 1; multipliers=c)
        c[1] = 10
        @test custom.multipliers == [0, 0.5]
        @test custom.values[:, 1] == mu.values
        for bad in (0, 2, true, 1.5)
            @test_throws ArgumentError modes_of_variation(f.model, bad)
        end
        for bad in ((), [NaN], [Inf], ("x",), :all)
            @test_throws ArgumentError modes_of_variation(f.model, 1; multipliers=bad)
        end
        multi = MultivariateFunctionalData([f.fd, GridFunctionalData(2 .* f.fd.values, f.grid .+ 2)])
        joint = @inferred StatsAPI.fit(MFPCA, multi; ncomponents=2, univariate=(; ncomponents=2, max_k=2))
        jm = @inferred modes_of_variation(joint, 1)
        @test length(jm.nodes) == 2
        @test (@inferred mean_data(joint)).nodes[2].grid == f.grid .+ 2
        for j in 1:2, (l, c) in enumerate(jm.nodes[j].multipliers)
            expected = vec(component(joint, j).mean.values) .+
                c*sqrt(eigvals(joint)[1])*eigenfunctions(joint)[j].values[:, 1]
            @test jm.nodes[j].values[:, l] ≈ expected
        end
        @test (@inferred scree_data(joint)).threshold === nothing
        @test scree_data(joint).eigenvalues == joint.settings.all_eigenvalues
        @test length(scree_data(joint.univariate).nodes) == 2
        @test scree_data(joint.univariate).nodes[1].eigenvalues ==
            component(joint, 1).all_eigenvalues
        @test modes_of_variation(joint.univariate, 1).nodes[2].eigenvalue ≈ 8
        samegrid = MultivariateFunctionalData([f.fd, GridFunctionalData(2 .* f.fd.values, f.grid)])
        pooled = StatsAPI.fit(FPCA, samegrid; basis=:pooled, ncomponents=1, max_k=2)
        @test scree_data(pooled).eigenvalues ≈ [5.0, 1.25]
        @test modes_of_variation(pooled, 1).nodes[2].eigenvalue ≈ 5
        @test length(mean_data(pooled).nodes) == 2
    end

    @testset "Covariance surfaces and matrix orientation" begin
        for model in (f.model, f.pace)
            stored = @inferred covariance_data(model)
            projected = @inferred covariance_data(model; surface=:fitted)
            @test stored.values ≈ f.surface
            @test projected.values ≈ f.surface
            @test norm(projected.values - eigvals(model)[1] .* (model.basis.values * model.basis.values')) > 0.1
            @test stored.grid1 == stored.grid2 == f.grid
            @test stored.grid1 !== stored.grid2
            @test stored.values !== model.covariance
            @test projected.values !== model.settings.full_projected_covariance
            @test_throws ArgumentError covariance_data(model; surface=:other)
            @test_throws ArgumentError covariance_data(model; surface=missing)
        end
        sm = StatsAPI.fit(SmoothedMean, f.fd; grid=f.grid, bandwidth=2.0)
        sc = StatsAPI.fit(SmoothedCovariance, f.fd, sm; grid=f.grid, bandwidth=2.0)
        @test (@inferred covariance_data(sc)).values == sc.surface
        @test covariance_data(sc).values !== sc.surface
        other = GridFunctionalData(f.fd.values[:, [1, 3, 5]] .* 2, [2.0, 2.4, 3.0])
        exact = StatsAPI.fit(CrossCovariance, f.fd, other)
        sm2 = StatsAPI.fit(SmoothedMean, other; grid=other.grid, bandwidth=2.0)
        cross = StatsAPI.fit(SmoothedCrossCovariance, f.fd, other, sm, sm2;
            grid1=f.grid, grid2=other.grid, bandwidth=2.0)
        for object in (exact, cross)
            cd = @inferred covariance_data(object)
            @test size(cd.values) == (5, 3)
            @test cd.grid1 == f.grid && cd.grid2 == other.grid
            @test cd.values == object.surface && cd.values !== object.surface
            recipe = only(plotting_series(FunctionalDataAnalysis._FDAPlot{:covariance}(cd)))
            @test recipe.args[1] == f.grid && recipe.args[2] == other.grid
            @test recipe.args[3] == object.surface'
            @test recipe.plotattributes[:seriestype] == :heatmap
        end
        # The exact rectangle is independently known from the KL fixture.
        @test exact.surface ≈ f.surface[:, [1, 3, 5]] .* 2
    end

    @testset "All ordered pairs, diagonal policy, and exact counts" begin
        repeated = IrregularFunctionalData([[0.0, 0.5, 1.0] for _ in 1:5], [ones(3) for _ in 1:5])
        d = @inferred design_data(repeated)
        @test length(d.x) == 30 && all(d.x .!= d.y)
        @test d.curves == repeat(1:5; inner=6)
        @test d.counts.x == [0, 0, 0.5, 0.5, 1, 1]
        @test d.counts.y == [0.5, 1, 0, 1, 0, 0.5]
        @test d.counts.values == fill(5, 6) && d.counts.display == fill(4, 6)
        withdiag = design_data(repeated; diagonal=true)
        @test length(withdiag.x) == 45 && length(withdiag.counts.values) == 9
        @test all(==(5), withdiag.counts.values)
        singleton = IrregularFunctionalData([[0.5]], [[2.0]])
        @test isempty(design_data(singleton).x)
        @test design_data(singleton; diagonal=true).counts.values == [1]
        zeros = IrregularFunctionalData([[-0.0, 1.0], [0.0, 1.0]], [ones(2), ones(2)])
        @test design_data(zeros).grid == [0.0, 1.0]
        @test design_data(zeros).counts.values == [2, 2]
        mask = GridFunctionalData([1.0 missing 2.0; missing 1.0 2.0], [0.0, 0.5, 1.0])
        @test design_data(mask) == design_data(to_irregular(mask))
        multi = MultivariateFunctionalData([mask, to_irregular(mask)])
        @test design_data(multi).nodes[1] == design_data(multi).nodes[2]
        basis = BasisFunctionalData(ones(1, 2), MonomialBasis((0.0, 1.0), 1))
        @test_throws ArgumentError design_data(basis)
        @test_throws ArgumentError design_data(repeated; diagonal=:yes)
        # R 4.5.1 / fdapace 0.6.0 counts, generated by reference/plotting_m5.R.
        golden = readdlm(joinpath(@__DIR__, "data", "m5_design.csv"), ','; comments=true)
        for diagonal in (false, true)
            ref = golden[golden[:, 1] .== Int(diagonal), :]
            actual = design_data(f.irregular; diagonal).counts
            @test actual.x == ref[:, 2] && actual.y == ref[:, 3]
            @test actual.values == ref[:, 4]
        end
    end

    @testset "Training observations and analytic reconstructions" begin
        grid = [0.0, 0.25, 1.0]
        dense = @inferred fitted_curve_data(f.model, 1; grid)
        @test dense.observed.times == f.grid && dense.observed.values == f.fd.values[1, :]
        @test dense.values ≈ 1 .+ grid .+ sqrt(5)
        @test dense.grid !== grid
        pace = @inferred fitted_curve_data(f.pace, 2; grid)
        t = f.times[2]
        # Analytic BLUP, with both retained covariance components in the solve.
        slope_norm = sqrt(dot(quadrature_weights(f.grid), (f.grid .- 0.5).^2))
        phi = hcat(ones(length(t)), (t .- 0.5) ./ slope_norm)
        covariance = phi * Diagonal([2.0, 0.5]) * phi' + 0.1I
        score = 2 * dot(ones(length(t)), covariance \ fill(0.5, length(t)))
        @test pace.observed.times == t && pace.observed.values == f.values[2]
        @test pace.values ≈ 1 .+ grid .+ score
        @test pace.method == :pace && pace.ncomponents == 1
        @test f.pace.training_data !== f.irregular
        # Copies survive mutations to both caller inputs and returned coordinates.
        f.irregular.values[1] = 900
        @test fitted_curve_data(f.pace, 1).observed.values[1] != 900
        pace.observed.values[1] = 700
        @test fitted_curve_data(f.pace, 2).observed.values[1] != 700
        f.fd.values[1, 1] = 800
        @test fitted_curve_data(f.model, 1).observed.values[1] != 800
        original = plotting_fixture().fd
        smooth = StatsAPI.fit(FPCA, original; ncomponents=1, max_k=2,
            presmooth=LocalPolynomial(; bandwidth=2.0, degree=0))
        @test fitted_curve_data(smooth, 1).observed.values == original.values[1, :]
        @test norm(fitted_curve_data(smooth, 1).values - original.values[1, :]) > 0.01
        masked_values = Matrix{Union{Missing,Float64}}(original.values)
        masked_values[1, 2] = missing
        masked = GridFunctionalData(masked_values, f.grid)
        mf = StatsAPI.fit(FPCA, masked; grid=f.grid, ncomponents=1, max_k=2,
            mean=(; grid=f.grid, values=1 .+ f.grid),
            covariance=(; grid=f.grid, surface=f.surface), noise_variance=0.1)
        @test fitted_curve_data(mf, 1).observed.times == f.grid[[1, 3, 4, 5]]
        multi = MultivariateFunctionalData([original, GridFunctionalData(2 .* original.values, f.grid .+ 2)])
        joint = StatsAPI.fit(MFPCA, multi; ncomponents=2, univariate=(; ncomponents=2, max_k=2))
        jd = @inferred fitted_curve_data(joint, 2; grid=[grid, grid .+ 2])
        @test length(jd.nodes) == 2 && jd.curve == 2
        @test jd.nodes[2].observed.times == f.grid .+ 2
        # Curve two is affine, so the off-grid value is known analytically.
        @test jd.nodes[1].values ≈ 1 .+ grid .- sqrt(5)
        for bad in (0, 7, true, 1.5)
            @test_throws ArgumentError fitted_curve_data(f.model, bad)
        end
        for bad in ([0.0], [0.0, NaN], [-0.1, 1.0], [0.5, 0.0], :auto)
            @test_throws ArgumentError fitted_curve_data(f.model, 1; grid=bad)
        end
    end

    @testset "Recipe series and attribute overrides" begin
        # Recreate inputs after the explicit mutation checks above.
        f = plotting_fixture()
        raw = plotting_series(f.fd; attrs=Dict{Symbol,Any}(:curves => [2, 1], :linewidth => 3))
        @test length(raw) == 2
        @test raw[1].args == (f.grid, f.fd.values[2, :])
        @test all(s -> s.plotattributes[:linewidth] == 3, raw)
        @test all(s -> !haskey(s.plotattributes, :curves), raw)
        @test all(s -> s.plotattributes[:markershape] == :none, raw)
        sparse = plotting_series(f.irregular)
        @test length(sparse) == 4 && sparse[1].args[1] == f.times[1]
        @test all(s -> s.plotattributes[:markershape] == :circle, sparse)
        basis = BasisFunctionalData([1.0 2.0], MonomialBasis((0.0, 1.0), 1))
        bs = only(plotting_series(basis; attrs=Dict{Symbol,Any}(:evaluation_grid => [0.0, 1.0])))
        @test bs.args == ([0.0, 1.0], [1.0, 3.0])
        @test !haskey(bs.plotattributes, :evaluation_grid)
        sm = StatsAPI.fit(SmoothedMean, f.fd; grid=f.grid, bandwidth=2.0)
        @test only(plotting_series(sm)).args == (sm.grid, sm.values)
        mode = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:modes), modes_of_variation(f.model, 1)))
        @test length(mode) == 5
        @test mode[1].args[2] ≈ 1 .+ f.grid
        @test occursin("λ=", mode[1].plotattributes[:title])
        @test occursin("FVE=", mode[1].plotattributes[:title])
        scree = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:scree), scree_data(f.model)))
        @test [s.plotattributes[:seriestype] for s in scree] == [:bar, :path, :vline, :hline]
        @test scree[1].args[2] ≈ [0.8, 0.2]
        @test scree[3].args == ([1],) && scree[4].args == ([0.7],)
        surface = only(plotting_series(FunctionalDataAnalysis._FDAPlot{:covariance}(covariance_data(f.model));
            attrs=Dict{Symbol,Any}(:seriestype => :surface, :title => "Custom")))
        @test surface.plotattributes[:seriestype] == :surface && surface.plotattributes[:title] == "Custom"
        design = only(plotting_series(FunctionalDataAnalysis._plot_payload(Val(:design), design_data(f.irregular))))
        @test design.plotattributes[:seriestype] == :scatter
        @test design.plotattributes[:marker_z] == design_data(f.irregular).counts.display
        fitted = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:fit), fitted_curve_data(f.pace, 1)))
        @test length(fitted) == 2
        @test fitted[1].args == (f.times[1], f.values[1])
        @test fitted[2].args[2] ≈ vec(reconstruct(f.pace).values[1, :])
        multi = MultivariateFunctionalData([f.fd, f.fd])
        panels = plotting_series(multi; attrs=Dict{Symbol,Any}(:curves => 1))
        @test length(panels) == 2
        @test [s.plotattributes[:subplot] for s in panels] == [1, 2]
        @test all(s -> s.plotattributes[:layout] == (2, 1), panels)
        @test occursin("Node 2", panels[2].plotattributes[:title])
        joint = StatsAPI.fit(MFPCA, multi; ncomponents=2, univariate=(; ncomponents=2, max_k=2))
        modes = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:modes), modes_of_variation(joint, 1)))
        @test length(modes) == 10
        @test occursin("Node 2: Mode", modes[6].plotattributes[:title])
        @test occursin("FVE=", modes[6].plotattributes[:title])
        @test !isdefined(Main, :Plots)
        @test all(s -> s.args != (nothing,), modes)
    end

    @testset "Band coordinates and ribbon recipes" begin
        fixture = plotting_fixture()
        b = fitted_bands(fixture.pace; curves=[2, 1], coverage=:band)
        d = @inferred band_data(b)
        @test d.grid == b.grid && d.grid !== b.grid
        @test d.values == b.fitted && d.values !== b.fitted
        @test d.lower == b.lower && d.lower !== b.lower
        @test d.upper == b.upper && d.upper !== b.upper
        @test d.curves == [2, 1] && d.types == [:band, :band]
        series = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:band), d);
            attrs=Dict{Symbol,Any}(:fillalpha => 0.4, :linewidth => 3))
        @test length(series) == 2
        @test series[1].args == (b.grid, b.fitted[1, :])
        @test series[1].plotattributes[:ribbon][1] ≈ b.fitted[1, :] - b.lower[1, :]
        @test series[1].plotattributes[:ribbon][2] ≈ b.upper[1, :] - b.fitted[1, :]
        @test all(s -> s.plotattributes[:fillalpha] == 0.4 && s.plotattributes[:linewidth] == 3, series)
        mean = StatsAPI.fit(SmoothedMean, fixture.fd; grid=fixture.grid, bandwidth=2.0)
        mb = mean_bands(fixture.fd, mean; B=19, rng=MersenneTwister(1))
        md = @inferred band_data(mb)
        @test md.types == [:interval, :band] && size(md.values) == (2, 5)
        @test md.lower[1, :] == mb.interval.lower && md.upper[2, :] == mb.band.upper
        @test (@inferred band_data(mb; coverage=:band)).types == [:band]
        @test length(plotting_series(FunctionalDataAnalysis._plot_payload(Val(:band), md))) == 2
        onlypoint = mean_bands(fixture.fd, mean; B=9, rng=MersenneTwister(1), coverage=:interval)
        @test_throws ArgumentError band_data(onlypoint; coverage=:both)
        @test_throws ArgumentError band_data(onlypoint; coverage=:band)
        @test_throws ArgumentError band_data(mb; coverage=:unknown)
        sandwich = mean_bands(fixture.fd, mean; method=:sandwich)
        @test band_data(sandwich).types == [:interval]
        # Never clip a percentile interval to force it to include the estimate.
        shifted = band_data(mb; coverage=:interval)
        shifted.lower .= shifted.values .+ 2
        shifted.upper .= shifted.values .+ 3
        ribbon = only(plotting_series(FunctionalDataAnalysis._FDAPlot{:band}(shifted))).plotattributes[:ribbon]
        @test ribbon[1] ≈ fill(-2.0, 5)
        @test ribbon[2] ≈ fill(3.0, 5)
        nodes = @inferred band_data((; nodes=[b, b]))
        panels = plotting_series(FunctionalDataAnalysis._plot_payload(Val(:band), nodes))
        @test length(panels) == 4
        @test panels[3].plotattributes[:subplot] == 2
        @test occursin("Node 2", panels[3].plotattributes[:title])
        d.values[1, 1] = 999
        @test b.fitted[1, 1] != 999
        @test !isdefined(Main, :Plots)
    end
end

# Convenience functions must prepare the correct recipe payload and forward
# plot attributes even when no plotting backend is loaded. Inspect the actual
# missing-backend call instead of installing test-only RecipesBase methods.
function plotting_boundary_payload(call)
    failure = try
        call()
        nothing
    catch error
        error
    end
    @test failure isa MethodError
    failure isa MethodError || return nothing
    if failure.f === Core.kwcall
        @test failure.args[1] == (; color=:blue)
        @test failure.args[2] === RecipesBase.plot
    else
        @test failure.f === RecipesBase.plot
    end
    last(failure.args)
end

@testset "plot convenience functions reach the backend boundary" begin
    f = plotting_fixture()
    cross = StatsAPI.fit(CrossCovariance, f.fd, f.fd)
    bands = fitted_bands(f.pace; curves=[1])
    mean = StatsAPI.fit(SmoothedMean, f.fd; grid=f.grid, bandwidth=2.0)
    mean_uncertainty = mean_bands(f.fd, mean; method=:sandwich)
    custom_grid = [0.0, 0.5, 1.0]
    cases = (
        (() -> modes_plot(f.model, 1; multipliers=[-1, 1], color=:blue),
            :modes, modes_of_variation(f.model, 1; multipliers=[-1, 1])),
        (() -> scree_plot(f.model; color=:blue), :scree, scree_data(f.model)),
        (() -> cov_plot(cross; color=:blue), :covariance, covariance_data(cross)),
        (() -> cov_plot(f.model; surface=:fitted, color=:blue),
            :covariance, covariance_data(f.model; surface=:fitted)),
        (() -> design_plot(f.irregular; diagonal=true, color=:blue),
            :design, design_data(f.irregular; diagonal=true)),
        (() -> fit_plot(f.model, 1; color=:blue), :fit, fitted_curve_data(f.model, 1)),
        (() -> fit_plot(f.model, 1; grid=custom_grid, color=:blue),
            :fit, fitted_curve_data(f.model, 1; grid=custom_grid)),
        (() -> band_plot(bands; color=:blue), :band, band_data(bands)),
        (() -> band_plot(mean_uncertainty; coverage=:interval, color=:blue),
            :band, band_data(mean_uncertainty; coverage=:interval)),
    )
    for (call, kind, expected) in cases
        payload = plotting_boundary_payload(call)
        @test payload isa FunctionalDataAnalysis._FDAPlot{kind}
        @test isequal(payload.data, expected)
    end
end
