import KernelSmoothing
import StatsAPI
using DelimitedFiles

@testset "Pooled mean" begin
    t = collect(0.0:0.1:1.0)
    for y in (fill(3.0, length(t)), 1 .+ 2t)
        fd = GridFunctionalData(permutedims(hcat(y, y)), t)
        m = @inferred StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=0.4)
        @test m.values ≈ y
        @test (@inferred evaluate(m, [0.05, 0.95])) ≈ [y[1] + (y[2]-y[1])/2, (y[end-1]+y[end])/2]
        @test (@inferred evaluate(m, 0.5)) ≈ y[6]
        @test evaluate(m, [1.0, 0.0]) ≈ y[[end, 1]]
        @test isempty(evaluate(m, Float64[]))
        @test m.design == sampling_design(fd)
        @test m.selection === nothing
        @test m.folds == 0 && m.binning == :none && m.minobs == 0
        @test m.grid !== t
        for q in ([-0.1], [1.1], [NaN], [Inf])
            @test_throws ArgumentError evaluate(m, q)
        end
    end
    # More observations of the high-valued curve shift the observation-weighted
    # fit upward; the endpoint estimates also have closed-form weighted means.
    unequal = IrregularFunctionalData([t, [0.0, 1.0]], [fill(4.0, 11), [0.0, 0.0]])
    obs = StatsAPI.fit(SmoothedMean, unequal; grid=[0.0, 1.0], bandwidth=2.0, degree=0)
    subj = StatsAPI.fit(SmoothedMean, unequal; grid=[0.0, 1.0], bandwidth=2.0, degree=0, weighting=:subj)
    @test all(obs.values .> subj.values)
    # Independent analytic weighted constant regression at the left endpoint.
    k = KernelSmoothing.EpanechnikovKernel()
    a, b = sum(k.(t ./ 2)), sum(k.([0.0, 0.5]))
    @test obs.values[1] ≈ 4a/(a+b)
    @test subj.values[1] ≈ (4a/11)/(a/11+b/2)

    times = [t, [0.0, .2, .5, .8, 1.0], [0.0, .1, .4, .6, .9, 1.0], t]
    ys = [sin.(3x) .+ i/10 for (i, x) in enumerate(times)]
    fd = IrregularFunctionalData(times, ys)
    candidates = [0.8, 0.45, 1.2]
    for weighting in (:obs, :subj)
        cv = @inferred CurveCV(candidates; folds=4)
        m = @inferred StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv, weighting)
        brute = map(candidates) do h
            score, total = 0.0, 0.0
            for i in 1:4
                train = setdiff(1:4, [i])
                x, y = vcat(times[train]...), vcat(ys[train]...)
                w = vcat([fill(weighting == :obs ? 1.0 : 1/length(times[j]), length(times[j])) for j in train]...)
                direct = KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, x, y;
                    bandwidth=h, degree=1, kernel=k, points=times[i], weights=w, minobs=0)
                wi = weighting == :obs ? 1.0 : 1/length(times[i])
                score += wi * sum(abs2, ys[i] - KernelSmoothing.coef(direct))
                total += wi * length(times[i])
            end
            score/total
        end
        @test m.selection.errors ≈ brute
        @test m.selection.assignments == 1:4
        @test m.bandwidth == candidates[argmin(brute)]
        @test m.selection.bandwidths !== cv.bandwidths
        @test occursin("selection", sprint(show, m))
    end
    rng = MersenneTwister(48)
    baseline_rng = copy(rng)
    cv1 = CurveCV(candidates; folds=2, rng)
    @test cv1.rng !== rng
    @test rand(rng) == rand(baseline_rng)
    cv2 = CurveCV(candidates; folds=2, rng=MersenneTwister(48))
    m1 = StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv1)
    m2 = StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv2)
    @test m1.selection == m2.selection
    @test sort(m1.selection.assignments) == [1, 1, 2, 2]
    other_split = StatsAPI.fit(SmoothedMean, fd; grid=t,
        bandwidth=CurveCV(candidates; folds=2, rng=MersenneTwister(12)))
    # Different seeds can produce the same partition; check balance instead.
    @test sort(other_split.selection.assignments) == [1, 1, 2, 2]
    @test StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv1).selection == m1.selection
    supplied = copy(candidates)
    selector = CurveCV(supplied; folds=4)
    supplied[1] = 10
    @test selector.bandwidths == candidates
    for hs in (Float64[], [0.0], [-1.0], [NaN], [Inf], [0.5, 0.5])
        @test_throws ArgumentError CurveCV(hs)
    end
    @test_throws ArgumentError CurveCV(candidates; folds=1)
    @test_throws ArgumentError CurveCV(candidates; folds=2.5)
    @test_throws ArgumentError CurveCV(candidates; rng=1)
    for cv in (CurveCV(candidates; folds=5), CurveCV(candidates; folds=2))
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=cv)
    end
    for kwargs in ((; weighting=:bad), (; degree=-1), (; kernel=:bad))
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=.5, kwargs...)
    end
    for h in (0, -1, Inf, NaN)
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid=t, bandwidth=h)
    end
    for grid in ([0.0], [1.0, 0.0], [0.0, Inf])
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=.5)
    end
    emptycurve = GridFunctionalData(fill(missing, 1, length(t)), t)
    @test_throws ArgumentError StatsAPI.fit(SmoothedMean, emptycurve; grid=t, bandwidth=.5)
    err = try
        StatsAPI.fit(SmoothedMean, fd; grid=[0.05, 0.06], bandwidth=.001)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("point 0.05", sprint(showerror, err))
    @test occursin("larger bandwidth", sprint(showerror, err))
    f32 = GridFunctionalData(Float32[1 2 3; 1 2 3], Float32[0, .5, 1])
    fitted32 = @inferred StatsAPI.fit(SmoothedMean, f32; grid=Float32[0, .5, 1], bandwidth=2f0)
    @test eltype(fitted32.values) == Float32
    @test fitted32.values ≈ Float32[1, 2, 3]
    @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid=t,
        bandwidth=CurveCV([.001]; folds=4))
    masked = GridFunctionalData(Union{Missing,Float64}[1 missing 3; 1 2 3], [0.0, .5, 1.0])
    @test StatsAPI.fit(SmoothedMean, masked; grid=t, bandwidth=2).values ≈ 1 .+ 2t

    input = readdlm(joinpath(@__DIR__, "data", "m2_input.csv"), ','; comments=true)
    golden = readdlm(joinpath(@__DIR__, "data", "m2_fdapace_mean.csv"), ',', Any; comments=true)
    ref_fd = IrregularFunctionalData([Float64.(input[input[:, 1] .== i, 2]) for i in 1:4],
        [Float64.(input[input[:, 1] .== i, 3]) for i in 1:4])
    for weighting in (:obs, :subj)
        rows = golden[:, 1] .== string(weighting)
        workgrid = Float64.(golden[rows, 3])
        expected = Float64.(golden[rows, 4])
        actual = StatsAPI.fit(SmoothedMean, ref_fd; grid=workgrid,
            bandwidth=Float64(golden[findfirst(rows), 2]), weighting)
        @test norm(actual.values - expected) / norm(expected) < 1e-8
    end
end

@testset "Pooled mean bandwidth rules and pilots" begin
    times = [collect(range(0., 1.; length=n)) for n in (21, 25, 29, 33)]
    ys = [sin.(4t) .+ .15i .* cos.(17t) for (i, t) in enumerate(times)]
    fd = IrregularFunctionalData(times, ys)
    grid = collect(0.:.1:1.)
    kernel = KernelSmoothing.GaussianKernel()
    x, y = vcat(times...), vcat(ys...)
    for weighting in (:obs, :subj)
        w = vcat([fill(weighting == :obs ? 1. : 1/length(t), length(t)) for t in times]...)
        for rule in (KernelSmoothing.RuleOfThumb(), KernelSmoothing.IMSEOptimal())
            direct = KernelSmoothing.select_bandwidth(rule,
                KernelSmoothing.SmoothingData(x, y; weights=w), kernel; degree=1)
            # KernelSmoothing selectors' `details` types depend on the data (and on
            # the KernelSmoothing version), so selector fits are not inferred.
            m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=rule, kernel, weighting)
            @test m.bandwidth == m.selection.bandwidth == direct.h
            @test m.selection.rule == rule
            @test m.selection.details == direct.details
            @test m.folds == 0
            @test occursin(string(nameof(typeof(rule))), sprint(show, m))
        end
        multipliers = [1., 2., 4.]
        pilot = KernelSmoothing.RuleOfThumb()
        cv = @inferred CurveCV(; pilot, multipliers, folds=2, rng=MersenneTwister(52))
        @test_throws ArgumentError("CurveCV multipliers must be nonempty") CurveCV(; pilot, multipliers=Float64[])
        @test_throws ArgumentError("CurveCV multipliers must be finite and positive") CurveCV(; pilot, multipliers=[1., -2.])
        multipliers[1] = 10.
        m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=cv, kernel, weighting)
        h = KernelSmoothing.select_bandwidth(pilot,
            KernelSmoothing.SmoothingData(x, y; weights=w), kernel; degree=1).h
        candidates = [1., 2., 4.] .* h
        explicit = StatsAPI.fit(SmoothedMean, fd; grid, kernel, weighting,
            bandwidth=CurveCV(candidates; folds=2, rng=MersenneTwister(52)))
        @test m.selection.h_pilot == h
        @test m.selection.pilot == pilot
        @test m.selection.multipliers == [1., 2., 4.]
        @test m.selection.bandwidths == candidates
        @test m.selection.errors == explicit.selection.errors
        @test m.selection.assignments == explicit.selection.assignments
        @test m.bandwidth == explicit.bandwidth
        @test m.values == explicit.values
        @test occursin("pilot=", sprint(show, cv))
    end
    for rule in (KernelSmoothing.NormalReference(), KernelSmoothing.SilvermanRule(),
                 KernelSmoothing.SheatherJones(), KernelSmoothing.LSCV(), KernelSmoothing.MSEOptimal())
        err = try
            StatsAPI.fit(SmoothedMean, fd; grid, kernel, bandwidth=rule)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(rule isa KernelSmoothing.MSEOptimal ? "pointwise" : "density-only", sprint(showerror, err))
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean, fd; grid, kernel,
            bandwidth=CurveCV(; pilot=rule, folds=4))
    end
    @test CurveCV(; pilot=KernelSmoothing.RuleOfThumb()).multipliers == 2.0 .^ range(-1, 2; length=10)
    @test_throws ArgumentError CurveCV(; pilot=1.)
    for multipliers in (Float64[], [0.], [-1.], [NaN], [Inf], [1., 1.])
        @test_throws ArgumentError CurveCV(; pilot=KernelSmoothing.RuleOfThumb(), multipliers)
    end
end
