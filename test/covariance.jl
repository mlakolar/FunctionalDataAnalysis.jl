import KernelSmoothing
import StatsAPI
using DelimitedFiles

@testset "Pooled covariance" begin
    input = readdlm(joinpath(@__DIR__, "data", "m2_input.csv"), ','; comments=true)
    times = [Float64.(input[input[:,1] .== i,2]) for i in 1:4]
    ys = [Float64.(input[input[:,1] .== i,3]) for i in 1:4]
    fd = IrregularFunctionalData(times, ys)
    golden = readdlm(joinpath(@__DIR__, "data", "m2_fdapace_cov.csv"), ',', Any; comments=true)
    grid = collect(0.0:.125:1.0)
    for weighting in (:obs, :subj)
        # The fixture centers at direct mean estimates: store the mean at every
        # observation time so interpolation is exact there.
        m = StatsAPI.fit(SmoothedMean, fd; grid=sort(unique(vcat(times...))), bandwidth=.65, weighting)
        c = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=.65, weighting)
        rows = golden[:,1] .== string(weighting)
        expected = reshape(Float64.(golden[rows,5]), length(grid), length(grid))
        @test norm(c.surface - expected)/norm(expected) < 1e-8
        @test c.surface == transpose(c.surface)
        @test c.weights == quadrature_weights(grid)
        @test c.grid !== grid
        @test (@inferred evaluate(c, .125, .25)) == c.surface[2,3]
        @test evaluate(c, .0625, .1875) ≈ sum(c.surface[1:2,2:3])/4
        @test evaluate(c, 1.0, 1.0) == c.surface[end,end]
        @test c.diagonal_bandwidth == c.noise_details.bandwidth == .65
        @test c.noise_details.selection === nothing
        @test c.noise_details.folds == 0
        @test occursin("diagonal_bandwidth=0.65", sprint(show, c))
        @test occursin("noise_details", sprint(show, c))
        for q in ((-1., 0.), (0., 2.), (NaN, 0.), (0., Inf))
            @test_throws ArgumentError evaluate(c, q...)
        end
        pool = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, nothing)
        for i in 1:4
            ix = pool.subjects .== i
            n = length(times[i])
            @test all(pool.weights[ix] .≈ (weighting == :obs ? 1.0 : 1/(n*(n-1))))
            @test sum(pool.weights[ix]) ≈ (weighting == :obs ? n*(n-1) : 1)
        end
        cv = CurveCV([.8, 1.2]; folds=4)
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=cv, weighting)
        # Independent weighted-design QR prediction checks both pair weights and
        # leave-curve-out scoring (rather than reusing the FDA smoother).
        errors = map(cv.bandwidths) do h
            score = 0.0
            for i in eachindex(pool.z)
                train = pool.subjects .!= pool.subjects[i]
                u, v = (pool.x[train] .- pool.x[i])/h, (pool.y[train] .- pool.y[i])/h
                w = sqrt.(pool.weights[train] .* KernelSmoothing.EpanechnikovKernel().(u) .* KernelSmoothing.EpanechnikovKernel().(v))
                X = hcat(ones(length(u)), u, v)
                beta = (X .* w) \ (pool.z[train] .* w)
                score += pool.weights[i] * abs2(pool.z[i] - beta[1])
            end
            score/sum(pool.weights)
        end
        @test selected.selection.errors ≈ errors
        @test selected.bandwidth == cv.bandwidths[argmin(errors)]
        @test selected.selection.assignments == 1:4
    end
    m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=.65)
    @test_throws UndefKeywordError StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65)
    for h in (0., -1., Inf, NaN, :bad)
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=h)
    end
    for selector in (CurveCV([1.]; folds=5), CurveCV([1.]; folds=2), CurveCV([.001]; folds=4))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=selector)
    end
    # Independent weighted QR fits to raw squares. Test original held-out times
    # even when training times are snapped, and both observation weight rules.
    for weighting in (:obs, :subj), binning in (nothing, 5),
            selector in (CurveCV([.8, 1.2]; folds=4),
                         CurveCV([.8, 1.2]; folds=2, rng=MersenneTwister(17)))
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=selector, weighting, binning)
        reference = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=selector, weighting)
        assignments = reference.selection.assignments
        original = vcat(times...)
        squares = (vcat(ys...) - evaluate(m, original)).^2
        subjects = vcat([fill(i, length(times[i])) for i in 1:4]...)
        weights = weighting == :obs ? ones(length(original)) : 1 ./ length.(times)[subjects]
        training_times = binning === nothing ? original : round.(original .* 4, RoundNearestTiesUp) ./ 4
        errors = map(selector.bandwidths) do h
            score = 0.0
            for i in eachindex(original)
                train = assignments[subjects] .!= assignments[subjects[i]]
                u = (training_times[train] .- original[i])/h
                w = sqrt.(weights[train] .* KernelSmoothing.EpanechnikovKernel().(u))
                beta = (hcat(ones(length(u)), u) .* w) \ (squares[train] .* w)
                score += weights[i]*abs2(squares[i]-beta[1])
            end
            score/sum(weights)
        end
        @test selected.noise_details.selection.bandwidths == selector.bandwidths
        @test selected.noise_details.selection.bandwidths !== selector.bandwidths
        @test selected.noise_details.selection.assignments == assignments
        @test selected.noise_details.selection.errors ≈ errors
        @test selected.diagonal_bandwidth == selector.bandwidths[argmin(errors)]
        @test selected.noise_details.bandwidth == selected.diagonal_bandwidth
        @test selected.noise_details.folds == selector.folds
        @test selected.surface == StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=.65, weighting, binning).surface
    end
    binned = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=1.2, binning=5)
    @test binned.binning == 5
    pool = FunctionalDataAnalysis._covariance_pool(fd, m, :subj, 5)
    @test all(t -> t in 0:.25:1, pool.times)
    @test length(pool.z) < sum(n*(n-1) for n in length.(times))
    # Expand snapped raw pairs independently. Aggregation must preserve both
    # fixed-bandwidth estimates and the original raw-product CV objective.
    for weighting in (:obs, :subj)
        raw = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, nothing)
        snap(t) = round(t*4, RoundNearestTiesUp)/4
        x, y = snap.(raw.x), snap.(raw.y)
        grouped = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, 5)
        @test sum(grouped.weights) ≈ sum(raw.weights)
        @test any(>(0), grouped.scatter)
        for i in eachindex(grouped.z)
            cell = (raw.subjects .== grouped.subjects[i]) .&
                (x .== grouped.x[i]) .& (y .== grouped.y[i])
            w, z = raw.weights[cell], raw.z[cell]
            @test grouped.weights[i] ≈ sum(w)
            @test grouped.z[i] ≈ dot(w, z)/sum(w)
            @test grouped.scatter[i] ≈ sum(w .* (z .- grouped.z[i]).^2) atol=1e-14
        end
        fit = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=1.2, weighting, binning=5)
        expected = [FunctionalDataAnalysis._local_linear2d(x, y, raw.z, raw.weights,
            a, b, 1.2, fit.kernel) for a in grid, b in grid]
        @test fit.surface ≈ (expected + expected')/2
        selector = CurveCV([.8, 1.2]; folds=2, rng=MersenneTwister(3))
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65,
            grid, bandwidth=selector, weighting, binning=5)
        assignments = selected.selection.assignments
        errors = map(selector.bandwidths) do h
            score = 0.0
            for i in eachindex(raw.z)
                train = assignments[raw.subjects] .!= assignments[raw.subjects[i]]
                prediction = FunctionalDataAnalysis._local_linear2d(x[train], y[train],
                    raw.z[train], raw.weights[train], x[i], y[i], h, fit.kernel)
                score += raw.weights[i]*abs2(raw.z[i]-prediction)
            end
            score/sum(raw.weights)
        end
        @test selected.selection.errors ≈ errors
        @test selected.bandwidth == selector.bandwidths[argmin(errors)]
    end
    for kwargs in ((; bandwidth=.001), (; bandwidth=0), (; bandwidth=Inf),
            (; bandwidth=1., weighting=:bad), (; bandwidth=1., binning=1),
            (; bandwidth=1., binning=2.5), (; bandwidth=1., kernel=:bad),
            (; bandwidth=CurveCV([1.]; folds=5)), (; bandwidth=CurveCV([1.]; folds=2)))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, kwargs...)
    end
    for g in ([0.], [1.,0.], [0.,Inf])
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid=g, bandwidth=1.)
    end
    single = IrregularFunctionalData([[0.]], [[1.]])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, single, m; diagonal_bandwidth=.65, grid, bandwidth=1.)
    narrow_mean = StatsAPI.fit(SmoothedMean, fd; grid=[.25,.75], bandwidth=1.)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, narrow_mean; diagonal_bandwidth=.65, grid, bandwidth=1.)
    a = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=CurveCV([1.]; folds=2, rng=MersenneTwister(3)))
    b = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=CurveCV([1.]; folds=2, rng=MersenneTwister(3)))
    @test a.selection == b.selection

    # Two normalized Legendre eigenfunctions give a curved covariance with
    # distinct eigenvalues. Orthogonal sign columns give exactly centered,
    # uncorrelated scores and measurement errors, independent of RNG versions.
    workgrid = collect(0.0:.1:1.0)
    eigenfunctions = [t -> sqrt(3)*(2t-1), t -> sqrt(5)*(6t^2-6t+1)]
    eigenvalues = [.3, .1]
    truth = [sum(eigenvalues[k]*eigenfunctions[k](s)*eigenfunctions[k](t)
        for k in 1:2) for s in workgrid, t in workgrid]
    # Nonconstant columns of a 16 x 16 Walsh matrix: two score columns and
    # eleven noise columns. Their empirical second moments are the identity.
    signs = [isodd(count_ones(i & j)) ? -1.0 : 1.0 for i in 0:15, j in 1:13]
    scores = signs[:,1:2] .* sqrt.(eigenvalues)'
    clean = scores * [f(t) for t in workgrid, f in eigenfunctions]'
    noise = .6 .* signs[:,3:13]
    @test vec(sum(clean; dims=1)) ≈ zeros(11) atol=1e-14
    @test clean' * clean / 16 ≈ truth
    @test clean' * noise / 16 ≈ zeros(11, 11) atol=1e-14
    @test noise' * noise / 16 ≈ .36 .* Matrix{Float64}(I, 11, 11)
    fits = map((clean, clean + noise)) do values
        data = GridFunctionalData(values, workgrid)
        meanfit = StatsAPI.fit(SmoothedMean, data; grid=workgrid, bandwidth=.205)
        StatsAPI.fit(SmoothedCovariance, data, meanfit; diagonal_bandwidth=.205, grid=workgrid, bandwidth=.205)
    end
    noisy = GridFunctionalData(clean + noise, workgrid)
    noisy_mean = StatsAPI.fit(SmoothedMean, noisy; grid=workgrid, bandwidth=.205)
    different_diagonal = StatsAPI.fit(SmoothedCovariance, noisy, noisy_mean;
        grid=workgrid, bandwidth=.205, diagonal_bandwidth=.405)
    @test different_diagonal.surface == fits[2].surface
    @test different_diagonal.diagonal_bandwidth == .405
    @test !isapprox(different_diagonal.noise_variance, fits[2].noise_variance; atol=1e-8)
    for (covfit, noise) in zip(fits, (0., .6))
        @test maximum(abs, covfit.surface - truth) < .18
        @test abs(covfit.noise_variance - noise^2) < .07
    end
    # Only deterministic smoothing bias remains in the analytic recovery bound;
    # noise changes diagonal squares but leaves pooled off-diagonal products equal.
    @test fits[2].surface ≈ fits[1].surface atol=1e-13
    # Unequal observation counts can make the untruncated difference negative:
    # the two nonzero curves receive relatively more pair than diagonal weight.
    ts = [workgrid, workgrid, [0.,1.], [0.,1.]]
    unequal = IrregularFunctionalData(ts, [ones(11), -ones(11), zeros(2), zeros(2)])
    um = StatsAPI.fit(SmoothedMean, unequal; grid=workgrid, bandwidth=2.)
    uc = StatsAPI.fit(SmoothedCovariance, unequal, um; diagonal_bandwidth=2., grid=[0.,.2,.6,1.], bandwidth=2.)
    @test uc.noise_details.truncated
    @test uc.noise_details.raw_variance < 0
    @test uc.noise_variance == 0
    @test uc.noise_details.integration_grid == [.25,.6,.75]
    # Independent trapezoids at clipped boundaries of a nonuniform work grid.
    d = uc.noise_details.diagonal - diag(uc.surface)
    left = d[2] + (.25-.2)/(.6-.2)*(d[3]-d[2])
    right = d[3] + (.75-.6)/(1-.6)*(d[4]-d[3])
    expected = 2*((.6-.25)*(left+d[3])/2 + (.75-.6)*(d[3]+right)/2)
    @test uc.noise_details.raw_variance ≈ expected
    f32 = GridFunctionalData(Float32[1 2 3; -1 -2 -3], Float32[0,.5,1])
    m32 = StatsAPI.fit(SmoothedMean, f32; grid=Float32[0,.5,1], bandwidth=2f0)
    c32 = @inferred StatsAPI.fit(SmoothedCovariance, f32, m32; diagonal_bandwidth=2f0, grid=Float32[0,.5,1], bandwidth=2f0)
    @test eltype(c32.surface) == Float32
    # The diagonal smoother uses covariance-specific context for undefined
    # windows, while ordinary mean fits retain their default label.
    method = LocalPolynomial(; bandwidth=.01)
    for label in (nothing, "noise variance diagonal smoother")
        kwargs = label === nothing ? NamedTuple() : (; label)
        error = try
            FunctionalDataAnalysis._mean_predict([0.,1.], [1.,1.], [1.,1.], [.5],
                method; kwargs...)
        catch err
            err
        end
        expected_label = label === nothing ? "pooled mean" : label
        @test error isa ArgumentError
        @test occursin("$expected_label has an undefined estimate at point 0.5; use a larger bandwidth",
            sprint(showerror, error))
    end
    # Only two pair locations cannot identify three local-linear coefficients.
    endpoints = GridFunctionalData([1. -1.; -1. 1.], [0.,1.])
    em = StatsAPI.fit(SmoothedMean, endpoints; grid=[0.,1.], bandwidth=2.)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, endpoints, em; diagonal_bandwidth=.65, grid=[0.,1.], bandwidth=2.)
end
