import KernelSmoothing
import StatsAPI

@testset "Sample cross-covariance" begin
    g1, g2 = [0., .3, 1.], [-1., 2.]
    x = GridFunctionalData([1. 2 4; 2 0 1; 3 -2 -2; 4 -4 -5], g1; ids=1:4)
    y = GridFunctionalData([3. 4; 5 1; 7 -2; 9 -5], g2; ids=1:4)
    c = @inferred StatsAPI.fit(CrossCovariance, x, y)
    # Both nodes are affine in the subject index, whose centered sum of
    # squares is five. This is an analytic outer-product covariance.
    @test c.surface == 5 .* ([1., -2., -3.] * [2., -3.]') ./ 3
    @test c.surface == StatsAPI.fit(CrossCovariance, y, x).surface'
    @test c.denominator == 3
    @test c.mean1 == [2.5, -1., -.5]
    @test c.mean2 == [6., -.5]
    @test c.weights1 == x.weights
    @test c.weights2 == y.weights
    @test c.ids == x.ids
    @test c.ids !== x.ids
    @test c.grid1 !== x.grid && c.grid2 !== y.grid
    @test c.weights1 !== x.weights && c.weights2 !== y.weights
    @test (@inferred evaluate(c, .15, .5)) ≈ sum(c.surface[1:2, :])/4
    @test evaluate(c, 1., 2.) == c.surface[end,end]
    @test occursin("denominator=3", sprint(show, c))
    @test_throws DimensionMismatch StatsAPI.fit(CrossCovariance, x[1:3], y)
    wrong = GridFunctionalData(y.values, g2; ids=[2,1,3,4])
    @test_throws ArgumentError StatsAPI.fit(CrossCovariance, x, wrong)
    @test StatsAPI.fit(CrossCovariance, x, GridFunctionalData(y.values, g2)).surface == c.surface
    @test_throws ArgumentError StatsAPI.fit(CrossCovariance, x[1:1], y[1:1])
    masked = GridFunctionalData([1. missing; 2 3; 4 5; 6 7], g2)
    @test_throws ArgumentError StatsAPI.fit(CrossCovariance, x, masked)
    for (s,t) in ((-.1,0.), (0.,3.), (NaN,0.), (0.,Inf))
        @test_throws ArgumentError evaluate(c, s, t)
    end
    small = GridFunctionalData(Float32.(y.values), Float32.(g2))
    @test eltype((@inferred StatsAPI.fit(CrossCovariance, small, small)).surface) == Float32
    @test eltype((@inferred StatsAPI.fit(CrossCovariance, x, small)).surface) == Float64
    huge = GridFunctionalData([1e308 1e308; -1e308 -1e308],[0.,1.])
    @test_throws ArgumentError StatsAPI.fit(CrossCovariance,huge,huge)
end

@testset "Smoothed cross-covariance" begin
    input = golden_read("m4_cross_input.csv")
    nodes = [golden_sparse4(input,node;subjects=1:12,ids=1:12) for node in 1:2]
    x, y = nodes
    g1, g2 = collect(range(0., 1.; length=51)), collect(range(0., 1.; length=51))
    m1 = GridFunctionalData(reshape(1 .+ g1, 1, :), g1)
    m2 = GridFunctionalData(reshape(-.2 .+ .3g2, 1, :), g2)
    golden = golden_read("m4_fdapace_crosscov.csv")
    kernel = KernelSmoothing.EpanechnikovKernel()
    for exclude_same_time in (false, true)
        c = @inferred StatsAPI.fit(SmoothedCrossCovariance, x, y, m1, m2;
            grid1=g1, grid2=g2, bandwidth=(.55,.8), kernel, exclude_same_time)
        expected = reshape(golden[golden[:,1] .== Int(exclude_same_time), 6], 51, 51)
        relative_error = golden_relative_error(c.surface,expected)
        @test relative_error < 1e-8
        if get(ENV, "FDA_M4_CROSS_REPORT", "false") == "true"
            @info "GetCrCovYX relative error" exclude_same_time relative_error
        end
        @test norm(c.surface-c.surface') > .1
        @test c.bandwidth == (.55,.8)
        @test c.exclude_same_time == exclude_same_time
        @test c.grid1 !== g1 && c.grid2 !== g2
        @test c.mean1 !== m1 && c.mean1.values !== m1.values
        @test c.mean2 !== m2 && c.mean2.grid !== m2.grid
        @test c.weights1 == quadrature_weights(g1)
        @test c.weights2 == quadrature_weights(g2)
        @test c.selection === nothing && c.folds == 0
        @test (@inferred evaluate(c, .01, .03)) ≈ sum(c.surface[1:2,2:3])/4
        @test evaluate(c, 1., 1.) == c.surface[end,end]
        reversefit = StatsAPI.fit(SmoothedCrossCovariance, y, x, m2, m1;
            grid1=g2, grid2=g1, bandwidth=(.8,.55), kernel, exclude_same_time)
        @test reversefit.surface ≈ transpose(c.surface) rtol=1e-12
        transposed = FunctionalDataAnalysis._cross_transpose(c)
        @test transposed.bandwidth == reversefit.bandwidth
        @test transposed.surface == transpose(c.surface)
        for field in ("grid1", "grid2", "bandwidth", "kernel", "weighting", "exclude_same_time",
                "binning", "degree", "design", "folds", "selection", "mean1", "mean2")
            @test occursin(field * "=", sprint(show, c))
        end
        for q in ((-1.,0.), (0.,2.), (NaN,0.), (0.,Inf))
            @test_throws ArgumentError evaluate(c, q...)
        end
    end
    # Independent QR regression on explicitly constructed raw products checks
    # both weightings, subject folds, and snapped-pair CV with aggregation.
    grid = [0.,.2,.7,1.]
    for weighting in (:obs, :subj), exclude_same_time in (false, true), binning in (nothing, 3)
        s, t, z, w, subjects = Float64[], Float64[], Float64[], Float64[], Int[]
        for i in 1:12
            tx, vx = x[i]; ty, vy = y[i]
            for j in eachindex(tx), k in eachindex(ty)
                exclude_same_time && tx[j] == ty[k] && continue
                push!(s, tx[j]); push!(t, ty[k])
                push!(z, (vx[j]-1-tx[j])*(vy[k]+.2-.3ty[k]))
                push!(w, weighting == :obs ? 1. : 1/(length(tx)*length(ty)))
                push!(subjects, i)
            end
        end
        if binning !== nothing
            s = round.(s .* 2, RoundNearestTiesUp) ./ 2
            t = round.(t .* 2, RoundNearestTiesUp) ./ 2
        end
        selector = CurveCV([1.2, 1.6]; folds=3, rng=MersenneTwister(13))
        c = @inferred StatsAPI.fit(SmoothedCrossCovariance, x, y, m1, m2;
            grid1=grid, bandwidth=selector, kernel, weighting, exclude_same_time, binning)
        assignments = c.selection.assignments
        function qr_at(s0, t0, h, train)
            u, v = (s[train] .- s0)/h, (t[train] .- t0)/h
            rootweights = sqrt.(w[train] .* kernel.(u) .* kernel.(v))
            ((hcat(ones(length(u)), u, v) .* rootweights) \ (z[train] .* rootweights))[1]
        end
        errors = map(selector.bandwidths) do h
            sum(w[q] * abs2(z[q]-qr_at(s[q], t[q], h,
                assignments[subjects] .!= assignments[subjects[q]])) for q in eachindex(z))/sum(w)
        end
        @test c.selection.errors ≈ errors rtol=1e-12
        @test c.bandwidth == selector.bandwidths[argmin(errors)]
        @test c.selection.bandwidths !== selector.bandwidths
        expected = [qr_at(a, b, c.bandwidth, trues(length(z))) for a in grid, b in grid]
        @test c.surface ≈ expected rtol=1e-12
        pool = FunctionalDataAnalysis._cross_pool(x,y,m1,m2,weighting,exclude_same_time,binning)
        @test sum(pool.weights) ≈ sum(w)
        if binning !== nothing
            @test length(pool.z) < length(z)
            @test any(>(0), pool.scatter)
        end
        again = StatsAPI.fit(SmoothedCrossCovariance, x, y, m1, m2;
            grid1=grid, bandwidth=selector, kernel, weighting, exclude_same_time, binning)
        @test c.selection == again.selection
    end
    gaussian = @inferred StatsAPI.fit(SmoothedCrossCovariance, x, y, m1, m2;
        grid1=grid, bandwidth=1.)
    @test gaussian.kernel isa KernelSmoothing.GaussianKernel
    @test gaussian.weighting == :obs && !gaussian.exclude_same_time && gaussian.binning === nothing
    @test gaussian.surface == StatsAPI.fit(SmoothedCrossCovariance, x,y,m1,m2;
        grid1=grid, bandwidth=(1.,1.)).surface
    @test_throws DimensionMismatch StatsAPI.fit(SmoothedCrossCovariance, x[1:11],y,m1,m2;
        grid1=grid, bandwidth=1.)
    wrong = IrregularFunctionalData([y[i][1] for i in 1:12], [y[i][2] for i in 1:12]; ids=12:-1:1)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,wrong,m1,m2; grid1=grid, bandwidth=1.)
    for kwargs in ((; bandwidth=0.), (; bandwidth=Inf), (; bandwidth=(1.,NaN)),
            (; bandwidth=(1.,)), (; bandwidth=:bad), (; bandwidth=.001, kernel),
            (; bandwidth=1., kernel=:bad), (; bandwidth=1., weighting=:bad),
            (; bandwidth=1., exclude_same_time=1), (; bandwidth=1., binning=1),
            (; bandwidth=1., binning=2.5), (; bandwidth=KernelSmoothing.RuleOfThumb()),
            (; bandwidth=CurveCV(; pilot=KernelSmoothing.RuleOfThumb())),
            (; bandwidth=CurveCV([1.]; folds=13)), (; bandwidth=CurveCV([1.]; folds=2)))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,y,m1,m2; grid1=grid, kwargs...)
    end
    for g in ([0.], [0.,0.], [1.,0.], [0.,Inf], [.1,1.])
        @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,y,m1,m2; grid1=g, bandwidth=1.)
        @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,y,m1,m2; grid1=grid, grid2=g, bandwidth=1.)
    end
    narrow = GridFunctionalData(zeros(1,2), [.1,.9])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,y,narrow,m2; grid1=grid, bandwidth=1.)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,x,y,GridFunctionalData(zeros(2,2),[0.,1.]),m2;
        grid1=grid, bandwidth=1.)
    singleton = IrregularFunctionalData([[0.], [1.]], [[1.],[-1.]])
    zero_mean = GridFunctionalData(zeros(1,2), [0.,1.])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,singleton,singleton,zero_mean,zero_mean;
        grid1=[0.,1.], bandwidth=2., exclude_same_time=true)
    singlex = IrregularFunctionalData([[0.],[0.],[1.],[1.]], [[1.],[1.],[1.],[1.]])
    singley = IrregularFunctionalData([[0.],[1.],[0.],[1.]], [[1.],[3.],[2.],[4.]])
    singlefit = @inferred StatsAPI.fit(SmoothedCrossCovariance,singlex,singley,zero_mean,zero_mean;
        grid1=[0.,.5,1.], bandwidth=2., weighting=:subj)
    @test singlefit.surface ≈ [1+s+2t for s in [0.,.5,1.], t in [0.,.5,1.]]
    huge = GridFunctionalData(fill(1e200,2,2),[0.,1.])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,huge,huge,zero_mean,zero_mean;
        grid1=[0.,1.],bandwidth=2.)
    emptycurve = GridFunctionalData([missing missing; 1. -1.], [0.,1.])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCrossCovariance,emptycurve,emptycurve,zero_mean,zero_mean;
        grid1=[0.,1.], bandwidth=2.)
    f32 = GridFunctionalData(Float32[1 2 3; -1 -2 -3], Float32[0,.5,1])
    m32 = StatsAPI.fit(SmoothedMean,f32; grid=f32.grid, bandwidth=2f0)
    c32 = @inferred StatsAPI.fit(SmoothedCrossCovariance,f32,f32,m32,m32;
        grid1=f32.grid, bandwidth=(2f0,3f0))
    @test eltype(c32.surface) == Float32
    # Mixed representations, means, numeric types, and rectangular domains.
    shifted = GridFunctionalData(Float64.(f32.values), [2.,2.5,3.])
    shifted_mean = GridFunctionalData(zeros(1,2), [2.,3.])
    mixed = @inferred StatsAPI.fit(SmoothedCrossCovariance,to_irregular(f32),shifted,m32,shifted_mean;
        grid1=Float32[0,1], grid2=[2.,2.25,3.], bandwidth=2.)
    @test size(mixed.surface) == (2,3)
    @test eltype(mixed.surface) == Float64
end

@testset "Cross-covariance blocks" begin
    grid = [0.,.2,.6,1.]
    values = [i*(1+t) + .1i^2*t^2 for i in 1:6, t in grid]
    x = GridFunctionalData(values,grid; ids=1:6)
    y = GridFunctionalData(reverse(values; dims=1),grid; ids=1:6)
    mfd = MultivariateFunctionalData([x,y])
    blocks = @inferred cross_covariances(mfd; bandwidth=2.)
    @test size(blocks) == (2,2)
    @test blocks[1,1] isa SmoothedCovariance
    @test blocks[1,1].noise_details.noise_method == :rotated
    @test blocks[1,1].noise_details.noise_interval == :full
    @test blocks[1,2] isa SmoothedCrossCovariance
    @test blocks[1,2].surface == transpose(blocks[2,1].surface)
    @test blocks[1,2].mean1.values == sum(values; dims=1)/6
    @test blocks[1,1].surface == StatsAPI.fit(SmoothedCovariance,x,blocks[1,2].mean1;
        grid, bandwidth=2., kernel=KernelSmoothing.GaussianKernel()).surface
    @test blocks[1,2].mean1 !== blocks[2,1].mean2
    blocks[2,1].surface[1,1] += 1
    @test blocks[1,2].surface != transpose(blocks[2,1].surface)
    for basis in (:marginal,:pooled)
        fpca = StatsAPI.fit(FPCA,mfd; basis, ncomponents=1, max_k=2)
        b = cross_covariances(mfd; fpca, bandwidth=2., grid=[grid, [0.,.5,1.]])
        @test b[1,2].mean1.values == fpca.means[1].values
        @test b[1,2].mean2.values == fpca.means[2].values
        @test b[1,2].mean1 !== fpca.means[1]
        @test b[1,2].surface == transpose(b[2,1].surface)
        @test size(b[1,2].surface) == (4,3)
        @test_throws ArgumentError cross_covariances(mfd; fpca, means=fpca.means, bandwidth=2.)
        @test_throws ArgumentError cross_covariances(mfd; fpca, mean_bandwidth=1., bandwidth=2.)
    end
    means = [StatsAPI.fit(SmoothedMean,fd; grid, bandwidth=1.) for fd in (x,y)]
    b = cross_covariances(mfd; means, bandwidth=CurveCV([1.5,2.]; folds=6), exclude_same_time=true)
    @test b[1,2].selection == b[2,1].selection
    @test b[1,2].exclude_same_time
    @test b[1,2].surface == transpose(b[2,1].surface)
    smoothed = cross_covariances(mfd; bandwidth=2., mean_bandwidth=[1.,2.])
    @test smoothed[1,2].mean1.bandwidth == 1.
    @test smoothed[1,2].mean2.bandwidth == 2.
    # Sufficiently varied pooled observation times for the 5%-range mean default.
    ts = [sort!(unique(vcat(0.,1.,collect(.02:.02:.98) .+ i/10000))) for i in 1:6]
    ir = IrregularFunctionalData(ts, [1 .+ t for t in ts]; ids=1:6)
    defaults = cross_covariances(MultivariateFunctionalData([ir,ir]); bandwidth=2.)
    @test defaults[1,2].mean1.bandwidth == .05
    @test length(defaults[1,2].grid1) == 51
    @test defaults[1,2].mean1.kernel isa KernelSmoothing.GaussianKernel
    @test defaults[1,2].mean1.values ≈ 1 .+ defaults[1,2].grid1
    for kwargs in ((; means=means[1:1]), (; grid=[grid]), (; mean_bandwidth=[1.]),
            (; bandwidth=(1.,2.)), (; fpca=:bad), (; weighting=:bad),
            (; means=[:bad,:bad]), (; exclude_same_time=1), (; grid=[.1,1.]))
        options = merge((; bandwidth=2.),kwargs)
        @test_throws Union{ArgumentError,DimensionMismatch} cross_covariances(mfd; options...)
    end
    wrongids = MultivariateFunctionalData([GridFunctionalData(values,grid; ids=6:-1:1),
        GridFunctionalData(values,grid; ids=6:-1:1)])
    fpca = StatsAPI.fit(FPCA,wrongids; ncomponents=1,max_k=2)
    @test_throws ArgumentError cross_covariances(mfd; fpca,bandwidth=2.)
    fewer = StatsAPI.fit(FPCA,mfd[1:5]; ncomponents=1,max_k=2)
    @test_throws DimensionMismatch cross_covariances(mfd; fpca=fewer,bandwidth=2.)
    @test size(cross_covariances(MultivariateFunctionalData([x]); bandwidth=2.)) == (1,1)
end

@testset "Sparse shared-score and measurement-error recovery" begin
    rng = MersenneTwister(20261024)
    n = 1000
    times = [sort(randperm(rng,11)[1:5] .- 1) ./ 10 for _ in 1:n]
    score = randn(rng,n)
    f1(t) = 1 + .5t
    f2(t) = .7 - .2t
    noise1, noise2 = [randn(rng,5) for _ in 1:n], [randn(rng,5) for _ in 1:n]
    grid = collect(0.:.1:1.)
    truth = [f1(s)*f2(t) for s in grid, t in grid]
    mu = GridFunctionalData(zeros(1,2),[0.,1.])
    independent = map((false,true)) do exclude_same_time
        x = IrregularFunctionalData(times,[score[i].*f1.(times[i]) + .4noise1[i] for i in 1:n])
        y = IrregularFunctionalData(times,[score[i].*f2.(times[i]) + .4noise2[i] for i in 1:n])
        StatsAPI.fit(SmoothedCrossCovariance,x,y,mu,mu; grid1=grid,bandwidth=.6,
            kernel=KernelSmoothing.EpanechnikovKernel(),exclude_same_time)
    end
    for c in independent
        @test maximum(abs,c.surface-truth) < .2
    end
    @test maximum(abs,independent[1].surface-independent[2].surface) < .08
    # Isolate systematic correlated-noise bias using balanced orthogonal
    # signs within each sparse design. Six nonconstant Walsh columns have
    # exactly zero means and identity second moments: one score and five
    # time-specific errors. Thus off-time noise products cancel exactly.
    signs = [isodd(count_ones(i & j)) ? -1. : 1. for i in 0:7, j in 1:6]
    grouped_times = [sort(randperm(rng,11)[1:5] .- 1) ./ 10 for _ in 1:125]
    balanced_times = [t for t in grouped_times for _ in 1:8]
    x = IrregularFunctionalData(balanced_times,
        [signs[a,1].*f1.(t) + 2signs[a,2:6] for t in grouped_times for a in 1:8])
    y = IrregularFunctionalData(balanced_times,
        [signs[a,1].*f2.(t) + 2signs[a,2:6] for t in grouped_times for a in 1:8])
    kept, excluded = map((false,true)) do exclude_same_time
        StatsAPI.fit(SmoothedCrossCovariance,x,y,mu,mu; grid1=grid,bandwidth=.6,
            kernel=KernelSmoothing.EpanechnikovKernel(),exclude_same_time)
    end
    @test sum(diag(kept.surface-truth))/length(grid) > .5
    @test maximum(abs,excluded.surface-truth) < .03
    @test norm(excluded.surface-truth) < norm(kept.surface-truth)/3
    if get(ENV,"FDA_M4_CROSS_REPORT","false") == "true"
        @info "Cross-covariance recovery" independent_error_kept=maximum(abs,independent[1].surface-truth) independent_error_excluded=maximum(abs,independent[2].surface-truth) independent_change=maximum(abs,independent[1].surface-independent[2].surface) correlated_diagonal_bias=sum(diag(kept.surface-truth))/length(grid) excluded_error=maximum(abs,excluded.surface-truth)
    end
end
