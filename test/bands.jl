using Test, LinearAlgebra, Random, DelimitedFiles
using FunctionalDataAnalysis
import StatsAPI
import KernelSmoothing

function bands_fixture(; K=3, scalar=Float64)
    readfixture(name) = scalar.(readdlm(joinpath(@__DIR__, "data", name), ','; comments=true))
    input = readfixture("m3_sparse_input.csv")
    mean_input = readfixture("m3_sparse_mean.csv")
    surface = readfixture("m3_sparse_surface.csv")
    sigma2 = only(readfixture("m3_sparse_noise.csv"))
    grid = mean_input[:, 1]
    n = Int(maximum(input[:, 1]))
    times = [input[input[:, 1] .== i, 2] for i in 1:n]
    values = [input[input[:, 1] .== i, 3] for i in 1:n]
    fd = IrregularFunctionalData(times, values)
    model = StatsAPI.fit(FPCA, fd; grid, mean=(; grid, values=mean_input[:, 2]),
        covariance=(; grid, surface), noise_variance=sigma2, ncomponents=K, maxK=5,
        eigen_tol=scalar === Float32 ? 1e-5 : 1e-10)
    (; fd, model, grid)
end

@testset "Confidence bands" begin
    @testset "R critical values and fdapace 0.6.0 golden bands" begin
        golden_q = readdlm(joinpath(@__DIR__, "data", "m5_chisq.csv"), ','; comments=true)
        for (K, alpha, expected) in eachrow(golden_q)
            actual = FunctionalDataAnalysis._chisq_upper_quantile(Int(K), alpha)
            @test abs(actual / expected - 1) < 1e-10
        end
        golden = readdlm(joinpath(@__DIR__, "data", "m5_fdapace_bands.csv"), ',', Any; comments=true)
        for K in (1, 3, 5)
            f = bands_fixture(; K)
            for coverage in (:interval, :band)
                b = @inferred fitted_bands(f.model; coverage)
                ref = golden[(golden[:, 1] .== String(coverage)) .& (golden[:, 2] .== K), :]
                for (field, col) in ((:fitted, 5), (:lower, 6), (:upper, 7))
                    expected = permutedims(reshape(Float64.(ref[:, col]), length(f.grid), nobs(f.fd)))
                    @test norm(getproperty(b, field) - expected) / norm(expected) < 1e-8
                end
                @test b.ncomponents == K && b.alpha == 0.05 && b.coverage == coverage
                @test b.method == :pace && b.curves == collect(1:nobs(f.fd))
                @test b.grid !== f.model.grid
                @test b.fitted ≈ reconstruct(f.model).values
            end
        end
    end

    @testset "Analytic conditional variance, interpolation, and nodes" begin
        grid = [0.0, 0.2, 0.7, 1.0]
        phi = ones(length(grid))
        fd = IrregularFunctionalData([[0.0, 0.7], [0.2, 1.0]], [[2.0, 3.0], [0.0, 1.0]])
        mu = (; grid, values=1 .+ grid)
        covariance = (; grid, surface=fill(2.0, length(grid), length(grid)))
        f = StatsAPI.fit(FPCA, fd; grid, mean=mu, covariance, noise_variance=0.5,
            ncomponents=1, maxK=1)
        query = [0.0, 0.1, 0.8, 1.0]
        b = @inferred fitted_bands(f; grid=query, curves=[2, 1])
        # A random intercept with prior variance 2 and two noisy measurements:
        # posterior variance = (1/2 + 2/(1/2))^-1 = 2/9.
        @test b.variance ≈ fill(2/9, 2, 4)
        @test b.fitted[1, :] ≈ 1 .+ query .- (2/9) * 2 * (1.2 + 1)
        @test b.fitted[2, :] ≈ 1 .+ query .+ (2/9) * 2 * (1 + 1.3)
        @test b.upper - b.fitted ≈ fill(1.959963984540054 * sqrt(2/9), 2, 4) rtol=1e-12
        @test fitted_bands(f; coverage=:band, grid=query).upper ≈ fitted_bands(f; grid=query).upper
        @test fitted_bands(f; curves=[true, false]).curves == [1]
        query[2] = 0.3
        @test b.grid[2] == 0.1
        @test b.settings.psd_relative_tolerance == 100eps(Float64)
        @test b.settings.variance_source == :stored_selected_score_covariance
        @test b.settings.measurement_error == false
        @test occursin("critical_value=", sprint(show, b))
        indefinite = deepcopy(f)
        indefinite.score_covariances[1, 1, 1] = -1.0
        @test_throws ArgumentError fitted_bands(indefinite)
        mfd = MultivariateFunctionalData([fd, fd])
        mf = StatsAPI.fit(FPCA, mfd; grid, mean=mu, covariance, noise_variance=0.5,
            ncomponents=1, maxK=1)
        mb = @inferred fitted_bands(mf; curves=2)
        @test length(mb.nodes) == 2
        @test mb.nodes[1].variance ≈ fill(2/9, 1, 4)
        off = @inferred fitted_bands(mf; grid=[[0.0, 0.5, 1.0], grid])
        @test size(off.nodes[1].fitted) == (2, 3)
        @test off.nodes[2].grid == grid
        @test_throws DimensionMismatch fitted_bands(mf; grid=[grid])
        dense = StatsAPI.fit(FPCA, GridFunctionalData([0.0 1 2 3; 1.0 2 3 4; 2.0 3 4 5], grid);
            ncomponents=1, maxK=1)
        @test_throws ArgumentError fitted_bands(dense)
        @test_throws ArgumentError fitted_bands(StatsAPI.fit(FPCA,
            MultivariateFunctionalData([dense.training_data, dense.training_data]); ncomponents=1, maxK=1))
        for bad in (0, 1, -0.1, NaN, Inf, true, :auto)
            @test_throws ArgumentError fitted_bands(f; alpha=bad)
        end
        for bad in (:both, :simultaneous, nothing)
            @test_throws ArgumentError fitted_bands(f; coverage=bad)
        end
        for bad in (0, 3, [1, 1], Int[], :first)
            @test_throws ArgumentError fitted_bands(f; curves=bad)
        end
        @test_throws DimensionMismatch fitted_bands(f; curves=[true])
        for bad in ([0.0], [0.0, NaN], [-0.1, 1.0], [0.5, 0.0], :auto)
            @test_throws ArgumentError fitted_bands(f; grid=bad)
        end
        single = bands_fixture(; scalar=Float32)
        @test eltype(fitted_bands(single.model).fitted) == Float32
        @test eltype(fitted_bands(single.model; grid=Float64.(single.grid)).fitted) == Float64
    end

    @testset "Curve bootstrap, stored choices, and RNG snapshots" begin
        grid = [0.0, 0.15, 0.6, 1.0]
        y = [1.0, 2.0, 4.0, 8.0]
        fd = GridFunctionalData(y .+ transpose(2grid), grid)
        mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=2.0,
            weighting=:subj, kernel=KernelSmoothing.GaussianKernel())
        rng = MersenneTwister(1234)
        snapshot = copy(rng)
        b = @inferred mean_bands(fd, mean; B=39, rng)
        @test rand(rng, 5) == rand(copy(snapshot), 5)
        generator = copy(snapshot)
        shifts = [sum(y[rand(generator, 1:4, 4)]) / 4 for _ in 1:39]
        # Each bootstrap mean is affine. Intercepts follow sampled whole curves.
        @test b.bootstrap ≈ shifts .+ transpose(2grid)
        @test b.standard_error ≈ fill(sqrt(sum(abs2, shifts .- sum(shifts)/39)/38), 4)
        sorted = sort(shifts)
        # R type-7 at .025 has h=1.95: .05*y[1] + .95*y[2].
        expected_lo = 0.05sorted[1] + 0.95sorted[2]
        expected_hi = 0.95sorted[38] + 0.05sorted[39]
        @test b.interval.lower ≈ expected_lo .+ 2grid
        @test b.interval.upper ≈ expected_hi .+ 2grid
        q = sort(abs.(shifts .- sum(y)/4) ./ b.standard_error[1])
        expected_c = 0.9q[37] + 0.1q[38] # h=1+38*.95=37.1
        @test b.band.upper ≈ mean.values .+ expected_c .* b.standard_error
        again = mean_bands(fd, mean; B=39, rng=b.rng_state)
        @test again.bootstrap == b.bootstrap && again.band == b.band
        @test b.rng_state == snapshot && b.mean !== mean
        @test fieldtype(typeof(b), :rng_state) === typeof(snapshot)
        @test b.mean.bandwidth == 2.0 && b.mean.weighting == :subj
        @test b.settings.bandwidth_selection == :fixed && b.settings.quantile == :type7
        @test occursin("B=39", sprint(show, b)) && occursin("rng_state=", sprint(show, b))
        # Default TaskLocalRNG must also be snapshotted without advancing it.
        @inferred mean_bands(fd, mean; B=5)
        default_state = copy(Random.default_rng())
        default = mean_bands(fd, mean; B=5)
        @test copy(Random.default_rng()) == default_state
        @test default.rng_state == default_state
        for type in (:interval, :band)
            limited = @inferred mean_bands(fd, mean; B=39, rng=snapshot, coverage=type)
            @test (limited.interval === nothing) == (type === :band)
            @test (limited.band === nothing) == (type === :interval)
        end
        # Fixed post-selection bandwidth: bootstrap records the original selector,
        # but every replicate uses its chosen scalar bandwidth.
        selected = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=CurveCV([1.5, 2.0]; folds=4))
        sb = mean_bands(fd, selected; B=5, rng=snapshot)
        @test sb.mean.selection == selected.selection
        @test sb.mean.bandwidth == selected.bandwidth
        # Observation/subject weights and masked observations are preserved.
        raw = GridFunctionalData([1.0 missing 2.0 4.0; 3.0 4.0 missing 5.0; 0.0 1.0 2.0 3.0], grid)
        for weighting in (:obs, :subj)
            m = StatsAPI.fit(SmoothedMean, raw; grid, bandwidth=2.0, weighting)
            bb = mean_bands(raw, m; B=7, rng=snapshot)
            ir = mean_bands(to_irregular(raw), m; B=7, rng=snapshot)
            @test bb.bootstrap == ir.bootstrap
            gen = copy(snapshot)
            for i in 1:7
                sample = raw[rand(gen, 1:3, 3)]
                expected = StatsAPI.fit(SmoothedMean, sample; grid, bandwidth=m.bandwidth,
                    kernel=m.kernel, degree=m.degree, weighting)
                @test bb.bootstrap[i, :] ≈ expected.values
            end
        end
        f32 = GridFunctionalData(Float32.(fd.values), Float32.(grid))
        m32 = StatsAPI.fit(SmoothedMean, f32; grid=Float32.(grid), bandwidth=2f0)
        @test eltype(mean_bands(f32, m32; B=5, rng=snapshot).band.lower) == Float32
        # Every point remains in the result, even if a resample's range shrinks.
        irregular = IrregularFunctionalData([[0.0, 0.3], [0.7, 1.0]], [[1.0, 1.6], [2.4, 3.0]])
        m = StatsAPI.fit(SmoothedMean, irregular; grid=[0.0, 0.5, 1.0], bandwidth=2.0)
        @test mean_bands(irregular, m; B=10, rng=snapshot).grid == m.grid
        zero = GridFunctionalData(zeros(3, 4), grid)
        mz = StatsAPI.fit(SmoothedMean, zero; grid, bandwidth=2.0)
        bz = mean_bands(zero, mz; B=5, rng=snapshot)
        @test all(iszero, bz.standard_error) && all(iszero, bz.band.lower)
        @test bz.settings.critical_value == 0
        @test mean_bands(zero, mz).B == 1000
        # A mean differing only within the compatibility tolerance still must
        # not turn zero bootstrap variation into a fabricated sup-t width.
        onefd = GridFunctionalData(ones(3, 4), grid)
        onemean = StatsAPI.fit(SmoothedMean, onefd; grid, bandwidth=2.0, degree=0,
            kernel=KernelSmoothing.UniformKernel())
        onemean.values .+= 8eps(Float64)
        @test_throws ArgumentError mean_bands(onefd, onemean; B=5, rng=snapshot)
        for bad in (0, 1, 1.5, true, :auto)
            @test_throws ArgumentError mean_bands(fd, mean; B=bad)
        end
        for bad in (0, 1, NaN, true)
            @test_throws ArgumentError mean_bands(fd, mean; alpha=bad)
        end
        @test_throws ArgumentError mean_bands(fd, mean; coverage=:simultaneous)
        @test_throws ArgumentError mean_bands(fd, mean; rng=123)
        @test_throws ArgumentError mean_bands(fd, mean; rng=RandomDevice())
        @test_throws ArgumentError mean_bands(fd[1:1], mean; B=3)
        @test_throws ArgumentError mean_bands(GridFunctionalData(fd.values .+ 1, grid), mean; B=3)
        # Compact windows can leave a bootstrap resample undefined. No retries.
        narrow = StatsAPI.fit(SmoothedMean, irregular; grid=[0.0, 0.3, 0.7, 1.0], bandwidth=0.4)
        err = try
            mean_bands(irregular, narrow; B=10, rng=snapshot)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("replicate", err.msg)
    end

    @testset "Analytic curve-cluster sandwich intervals" begin
        grid = [0.0, 0.15, 0.6, 1.0]
        intercepts = [1.0, 2.0, 4.0, 8.0]
        fd = GridFunctionalData(intercepts .+ transpose(2grid), grid)
        m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=2.0, degree=0,
            kernel=KernelSmoothing.UniformKernel())
        # The literal method determines the concrete RNG-state type. Keep it
        # in the inferred call rather than widening the keyword to Symbol.
        sandwich(data, mean) = mean_bands(data, mean; method=:sandwich)
        result = @inferred sandwich(fd, m)
        # Equal-size clusters, a constant local fit, uniform kernel, full windows:
        # CR1 is the sample variance of cluster means divided by n.
        expected = sqrt(sum(abs2, intercepts .- sum(intercepts)/4) / (4*3))
        @test result.standard_error ≈ fill(expected, 4)
        @test result.interval.lower ≈ m.values .- 1.959963984540054 * expected
        @test result.interval.upper ≈ m.values .+ 1.959963984540054 * expected
        @test result.band === nothing && result.coverage == :interval
        @test result.B == 0 && result.rng_state === nothing
        @test fieldtype(typeof(result), :rng_state) === Nothing
        @test size(result.bootstrap) == (0, 4)
        @test result.settings.variance_estimator == :CR1
        @test result.settings.inference == :conventional
        @test result.settings.minobs == 0 && result.settings.clusters == :curves
        @test occursin("method = :sandwich", sprint(show, result))
        @test mean_bands(to_irregular(fd), m; method=:sandwich).interval == result.interval
        @test_throws ArgumentError mean_bands(fd, m; method=:unknown)
        @test_throws ArgumentError mean_bands(fd, m; method=:sandwich, coverage=:both)
        @test_throws ArgumentError mean_bands(fd, m; method=:sandwich, B=1000)
        @test_throws ArgumentError mean_bands(fd, m; method=:sandwich, rng=MersenneTwister(1))
        @test_throws ArgumentError mean_bands(fd, m; method=:sandwich, alpha=0)
        # A compact window containing just one curve has no cluster variance.
        sparse = IrregularFunctionalData([[0.0, 0.3], [0.7, 1.0]], [[1.0, 1.6], [2.4, 3.0]])
        narrow = StatsAPI.fit(SmoothedMean, sparse; grid=[0.0, 0.3, 0.7, 1.0], bandwidth=0.4)
        @test_throws ArgumentError mean_bands(sparse, narrow; method=:sandwich)
    end
end
