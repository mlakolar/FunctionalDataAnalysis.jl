import KernelSmoothing
import StatsAPI
import RecipesBase

@testset "Automatic FPCA defaults" begin
    functions = fourier_eigenfunctions((-1.0, 2.0), 2)
    mu(t) = 1 + 0.3t
    small = simulate_kl(MersenneTwister(2026102401), 64; mean=mu,
        eigenfunctions=functions, eigenvalues=[1.5, 0.6], noise_sd=0.5,
        sampling=SparseSampling((-1.0, 2.0), 3:6)).data

    @testset "No-keyword routes and recorded choices" begin
        original_times, original_values = copy(small.times), copy(small.values)
        model = @inferred StatsAPI.fit(FPCA, small)
        observed_range = maximum(small.times) - minimum(small.times)
        @test model.method == :pace
        @test model.grid == collect(range(extrema(small.times)...; length=51))
        @test domain(small) == (-1.0, 2.0)
        @test extrema(model.grid) != domain(small)
        @test model.settings.maxK == 20
        @test model.selection.rule == :FVE
        @test model.selection.threshold == 0.99
        @test model.settings.kernel isa KernelSmoothing.GaussianKernel
        @test model.settings.weighting == :obs
        @test model.settings.mean_input.bandwidth ≈ 0.05observed_range
        @test model.settings.covariance_input.bandwidth ≈ 0.10observed_range
        @test model.settings.covariance_input.diagonal_bandwidth ==
            model.settings.covariance_input.bandwidth
        @test model.settings.requested_binning == :auto
        @test model.settings.binning === nothing
        @test model.settings.mean_source == :estimated
        @test model.settings.covariance_source == :estimated
        @test model.settings.noise_variance_source == :estimated
        @test model.settings.covariance_input.noise_details.noise_method == :rotated
        @test model.settings.covariance_input.noise_details.noise_interval == :full
        @test isfinite(noise_variance(model)) && noise_variance(model) >= 0
        @test all(isfinite, scores(model))
        @test small.times == original_times
        @test small.values == original_values
        # Downstream uncertainty, derivative and plot consumers must accept a
        # PACE fit whose noise was estimated with the rotated/full default.
        bands = fitted_bands(model; curves=[1, 2])
        @test bands.fitted ≈ reconstruct(model).values[1:2, :]
        @test all(bands.lower .<= bands.fitted .<= bands.upper)
        differentiated = derivative(model; bandwidth=0.6)
        @test all(isfinite, reconstruct(differentiated).values)
        @test size(score_covariances(differentiated)) == size(score_covariances(model))
        @test fitted_curve_data(model, 1).values ≈ vec(bands.fitted[1, :])
        payload = FunctionalDataAnalysis._plot_payload(Val(:fit), fitted_curve_data(model, 1))
        @test length(RecipesBase.apply_recipe(Dict{Symbol,Any}(), payload)) == 2
        for setting in ("grid_source", "GaussianKernel", "weighting", "obs",
                "mean_bandwidth", "covariance_bandwidth", "diagonal_bandwidth",
                "requested_binning", "binning", "maxK", "0.99",
                "noise_variance", "mean_source", "covariance_source",
                "eigen_tol", "sign_tol", "regularization")
            @test occursin(setting, sprint(show, model))
        end

        # NoInterpolation preserves exactly the same observations and the
        # automatic work grid uses their range rather than missing end cells.
        target = sort!(unique(vcat(-1.0, small.times, 2.0)))
        masked = to_grid(small, target, NoInterpolation())
        @test masked.observed !== nothing
        masked_model = StatsAPI.fit(FPCA, masked)
        @test masked_model.method == :pace
        @test masked_model.grid == model.grid
        @test eigvals(masked_model) ≈ eigvals(model)
        @test scores(masked_model) ≈ scores(model)

        grid = collect(range(-1.0, 2.0; length=51))
        regular = simulate_kl(MersenneTwister(2026102402), 32; mean=mu,
            eigenfunctions=functions, eigenvalues=[1.5, 0.6],
            sampling=RegularSampling(grid)).data
        dense = StatsAPI.fit(FPCA, regular)
        explicit_dense = @inferred (fd -> StatsAPI.fit(FPCA, fd; method=:dense))(regular)
        @test eigvals(explicit_dense) == eigvals(dense)
        @test dense.method == :dense
        @test dense.grid == grid
        @test dense.grid !== regular.grid
        @test dense.selection.maxK == 30 # The dense default has no cap of 20.
        @test dense.selection.threshold == 0.99
        @test noise_variance(dense) === nothing
    end

    @testset "Multivariate routes inherit rotated/full noise" begin
        data = MultivariateFunctionalData([small, small])
        marginal = StatsAPI.fit(FPCA, data; ncomponents=2, maxK=3)
        joint = StatsAPI.fit(MFPCA, data; ncomponents=2,
            univariate=(; ncomponents=2, maxK=3))
        for fitted in (marginal, joint), j in 1:2
            node = component(fitted, j)
            details = node.settings.covariance_input.noise_details
            @test details.noise_method == :rotated
            @test details.noise_interval == :full
            @test isfinite(noise_variance(node))
        end
        @test all(isfinite, scores(joint))
    end

    @testset "Overrides, selectors, and errors" begin
        grid = collect(range(extrema(small.times)...; length=11))
        explicit = StatsAPI.fit(FPCA, small; grid, ncomponents=1, maxK=3,
            mean_bandwidth=0.6, covariance_bandwidth=0.8,
            kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:subj,
            binning=7, noise_variance=0.25, eigen_tol=1e-9, sign_tol=1e-11)
        @test explicit.grid == grid
        @test explicit.settings.maxK == 3
        @test ncomponents(explicit) == 1
        @test explicit.settings.mean_input.bandwidth == 0.6
        @test explicit.settings.covariance_input.bandwidth == 0.8
        @test explicit.settings.covariance_input.diagonal_bandwidth == 0.8
        @test explicit.settings.kernel isa KernelSmoothing.EpanechnikovKernel
        @test explicit.settings.weighting == :subj
        @test explicit.settings.requested_binning == 7
        @test explicit.settings.binning == 7
        @test noise_variance(explicit) == 0.25
        @test explicit.settings.eigen_tol == 1e-9
        @test explicit.settings.sign_tol == 1e-11
        separate_diagonal = StatsAPI.fit(FPCA, small; grid,
            mean_bandwidth=0.6, covariance_bandwidth=0.8,
            diagonal_bandwidth=0.5, noise_variance=0.25)
        @test separate_diagonal.settings.covariance_input.diagonal_bandwidth == 0.5
        @test separate_diagonal.settings.covariance_input.noise_details.diagonal_source == :user

        mean_cv = CurveCV([0.5, 0.8]; folds=2, rng=MersenneTwister(101))
        covariance_cv = CurveCV([0.6, 0.9]; folds=2, rng=MersenneTwister(102))
        selected = StatsAPI.fit(FPCA, small; grid, mean_bandwidth=mean_cv,
            covariance_bandwidth=covariance_cv, noise_variance=0.25)
        @test selected.settings.mean_input.selection.bandwidths == mean_cv.bandwidths
        @test selected.settings.covariance_input.selection.bandwidths == covariance_cv.bandwidths
        @test selected.settings.mean_input.folds == 2
        @test selected.settings.covariance_input.folds == 2
        rule = KernelSmoothing.RuleOfThumb()
        selected_rule = StatsAPI.fit(FPCA, small; grid, mean_bandwidth=rule,
            covariance_bandwidth=0.8, noise_variance=0.25)
        @test selected_rule.settings.mean_input.selection.rule == rule
        @test selected_rule.settings.mean_input.bandwidth > 0
        @test selected_rule.settings.mean_input.folds == 0

        for bandwidth in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small; mean_bandwidth=bandwidth)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small; covariance_bandwidth=bandwidth)
        end
        for binning in (:invalid, 1, 2.5)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small; binning)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, small; grid=[0.0, 0.0])
        @test_throws ArgumentError StatsAPI.fit(FPCA, small; grid=[-0.5, 1.5])
        @test_throws ArgumentError StatsAPI.fit(FPCA, small; weighting=:invalid)
        for bandwidth in (0.6, -1.0, NaN)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small;
                mean=explicit.settings.mean_input, mean_bandwidth=bandwidth)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small;
                covariance=explicit.settings.covariance_input,
                covariance_bandwidth=bandwidth)
            @test_throws ArgumentError StatsAPI.fit(FPCA, small;
                covariance=explicit.settings.covariance_input,
                diagonal_bandwidth=bandwidth)
        end

        @testset "Supplied inputs reject unused smoothing options" begin
            fitted_mean = explicit.settings.mean_input
            fitted_covariance = explicit.settings.covariance_input
            means = (fitted_mean, (; grid=fitted_mean.grid, values=fitted_mean.values))
            covariances = (fitted_covariance,
                (; grid=fitted_covariance.grid, surface=fitted_covariance.surface))
            for mean in means, covariance in covariances
                options = (; grid, mean, covariance, noise_variance=0.25,
                    ncomponents=1, maxK=3)
                reused = @inferred StatsAPI.fit(FPCA, small; options...)
                @test eigvals(reused) ≈ eigvals(explicit)
                @test scores(reused) ≈ scores(explicit)
                @test reused.settings.kernel === nothing
                @test reused.settings.weighting === nothing
                @test reused.settings.binning === nothing
                @test reused.settings.binning_rule === :not_applicable
                for kernel in (KernelSmoothing.EpanechnikovKernel(),
                        KernelSmoothing.GaussianKernel())
                    @test_throws ArgumentError StatsAPI.fit(FPCA, small; options..., kernel)
                end
                @test_throws ArgumentError StatsAPI.fit(FPCA, small;
                    options..., weighting=:subj)
                for binning in (7, nothing)
                    @test_throws ArgumentError StatsAPI.fit(FPCA, small; options..., binning)
                end
            end

            # A supplied covariance disables binning, but kernel and weighting
            # still control the mean when it must be estimated.
            for covariance in covariances
                options = (; grid, covariance, noise_variance=0.25,
                    ncomponents=1, maxK=3, mean_bandwidth=0.6,
                    kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:subj)
                estimated_mean = @inferred StatsAPI.fit(FPCA, small; options...)
                @test estimated_mean.settings.mean_source === :estimated
                @test estimated_mean.settings.mean_input.kernel isa KernelSmoothing.EpanechnikovKernel
                @test estimated_mean.settings.mean_input.weighting === :subj
                @test estimated_mean.settings.binning === nothing
                @test estimated_mean.settings.binning_rule === :not_applicable
                for binning in (7, nothing)
                    @test_throws ArgumentError StatsAPI.fit(FPCA, small; options..., binning)
                end
            end
            # Original fitted settings remain available when an input is reused.
            reused = StatsAPI.fit(FPCA, small; grid, mean=fitted_mean,
                covariance=fitted_covariance, noise_variance=0.25, ncomponents=1, maxK=3)
            @test reused.settings.mean_input.kernel isa KernelSmoothing.EpanechnikovKernel
            @test reused.settings.mean_input.weighting === :subj
            @test reused.settings.covariance_input.binning == 7
        end
    end

    @testset "Supplied covariance skips automatic binning" begin
        # More than 400 observations per curve would select 400 bins if the
        # covariance were estimated. Reusing a surface must not record that.
        times = collect(range(0.0, 1.0; length=401))
        fd = IrregularFunctionalData([times, times], [1 .+ times, 2 .+ times])
        grid = [0.0, 0.5, 1.0]
        covariance = (; grid, surface=ones(3, 3))
        for mean in (nothing, (; grid, values=1.5 .+ grid))
            model = StatsAPI.fit(FPCA, fd; grid, mean, covariance,
                noise_variance=0.25, ncomponents=1, maxK=1)
            @test model.settings.requested_binning === :auto
            @test model.settings.binning === nothing
            @test model.settings.binning_rule === :not_applicable
            displayed = sprint(show, model)
            @test occursin("binning_rule", displayed)
            @test occursin(":not_applicable", displayed)
        end
    end

    @testset "fdapace geometric bandwidth candidates" begin
        # The shared four-time design has a 4-by-4 Cartesian pair cloud.
        # Every third-neighbor distance is exactly 1/3; Minb at lag 3 is 1.
        times = collect(range(0.0, 1.0; length=4))
        tiny = IrregularFunctionalData([times, times],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        candidates = @inferred pace_bandwidth_candidates(tiny)
        @test length(candidates.mean) == length(candidates.covariance) == 10
        # The sparse mean start is min(2.5*0.75*1, 1), then sorted.
        # Gaussian covariance starts at 0.2*BwNN.
        expected_mean = sort((0.25) .^ ((0:9) ./ 9))
        endpoint = sqrt(2.0) / 4
        initial = 0.2 / 3
        expected_covariance = initial .* (endpoint / initial) .^ ((0:9) ./ 9)
        @test candidates.mean ≈ expected_mean rtol=1e-14
        @test candidates.covariance ≈ expected_covariance rtol=1e-14
        @test all(isfinite, candidates.mean)
        @test all(>(0), candidates.mean)
        @test all(>(0), diff(candidates.mean))
        @test all(>(0), diff(candidates.covariance))
        @test pace_bandwidth_candidates(tiny; grid=[0.0, 0.5, 1.0]) == candidates
        @test_throws ArgumentError pace_bandwidth_candidates(tiny; grid=[0.0, 0.0])
        @test_throws ArgumentError pace_bandwidth_candidates(tiny; grid=[0.1, 1.0])

        # fdapace's design classification uses coverage of pooled times,
        # independently of whether identical observations are stored on a grid.
        grid = collect(range(0.0, 1.0; length=9))
        complete = GridFunctionalData(ones(2, 9), grid)
        @test FunctionalDataAnalysis._pace_reference_design(complete) == :Dense
        @test FunctionalDataAnalysis._pace_reference_design(to_irregular(complete)) == :Dense
        @test pace_bandwidth_candidates(complete) ==
            pace_bandwidth_candidates(to_irregular(complete))
        mostly_observed = Matrix{Union{Missing,Float64}}(ones(2, 9))
        mostly_observed[1, 3] = missing
        dense_missing = GridFunctionalData(mostly_observed, grid)
        @test FunctionalDataAnalysis._pace_reference_design(dense_missing) == :DenseWithMV
        @test FunctionalDataAnalysis._pace_reference_design(to_irregular(dense_missing)) == :DenseWithMV
        @test pace_bandwidth_candidates(dense_missing) ==
            pace_bandwidth_candidates(to_irregular(dense_missing))
        sparse_values = fill!(Matrix{Union{Missing,Float64}}(undef, 4, 9), missing)
        for i in 1:4
            sparse_values[i, 2i-1] = 1.0
            sparse_values[i, 2i] = 1.0
        end
        sparse_grid = GridFunctionalData(sparse_values, grid)
        @test FunctionalDataAnalysis._pace_reference_design(sparse_grid) == :Sparse
        @test FunctionalDataAnalysis._pace_reference_design(to_irregular(sparse_grid)) == :Sparse
        @test pace_bandwidth_candidates(sparse_grid) ==
            pace_bandwidth_candidates(to_irregular(sparse_grid))
    end

    @testset "fdapace automatic bin counts" begin
        # Exercise the count rule without smoothing the large designs.
        dense_grid = collect(range(0.0, 1.0; length=401))
        dense = GridFunctionalData(ones(2, 401), dense_grid)
        count, rule = FunctionalDataAnalysis._pace_binning(dense, :auto)
        @test count == 400
        @test rule.design == :Dense
        @test rule.count_statistic == :maximum
        @test rule.summary_count == 401
        irregular = IrregularFunctionalData([dense_grid, [0.0, 1.0]],
            [ones(401), ones(2)])
        count, rule = FunctionalDataAnalysis._pace_binning(irregular, :auto)
        @test count === nothing
        @test rule.design == :Sparse
        @test rule.count_statistic == :median
        @test rule.summary_count == 201.5
        @test first(FunctionalDataAnalysis._pace_binning(dense, nothing)) === nothing
        @test first(FunctionalDataAnalysis._pace_binning(dense, 17)) == 17

        count = let data = GridFunctionalData(ones(5001, 21),
                collect(range(0.0, 1.0; length=21)))
            first(FunctionalDataAnalysis._pace_binning(data, :auto))
        end
        @test count === nothing
        count = let data = GridFunctionalData(ones(50000, 21),
                collect(range(0.0, 1.0; length=21)))
            first(FunctionalDataAnalysis._pace_binning(data, :auto))
        end
        @test count == 20
        count = let data = GridFunctionalData(ones(50000, 20),
                collect(range(0.0, 1.0; length=20)))
            first(FunctionalDataAnalysis._pace_binning(data, :auto))
        end
        @test count === nothing
        # fdapace compares the unrounded mstar to m. Here mstar<400,
        # but ceil(mstar)==400, so binning remains explicitly enabled.
        count = let data = GridFunctionalData(ones(5001, 400),
                collect(range(0.0, 1.0; length=400)))
            first(FunctionalDataAnalysis._pace_binning(data, :auto))
        end
        @test count == 400
    end

    @testset "Sparse Fourier recovery with measurement error" begin
        # Statistical recovery tolerances include sampling variation and the
        # fixed Gaussian smoother's bias. These are not fdapace golden tests.
        functions = fourier_eigenfunctions((0.0, 1.0), 2)
        recovery_mu = t -> 1 + 0.5t
        sigma2 = 1.0
        simulation = simulate_kl(MersenneTwister(2026102403), 400; mean=recovery_mu,
            eigenfunctions=functions, eigenvalues=[2.0, 0.7], noise_sd=sqrt(sigma2),
            sampling=SparseSampling((0.0, 1.0), 2:6))
        model = StatsAPI.fit(FPCA, simulation.data)
        weights = model.weights
        mean_error = sqrt(dot(weights, (vec(model.mean.values) - recovery_mu.(model.grid)).^2))
        @test mean_error < 0.3
        @test ncomponents(model) >= 2
        eigenfunction_errors = Float64[]
        for k in 1:2
            truth = functions[k].(model.grid)
            truth ./= sqrt(dot(weights, truth.^2))
            recovered = model.basis.values[:, k]
            error = min(sqrt(dot(weights, (recovered-truth).^2)),
                sqrt(dot(weights, (recovered+truth).^2)))
            push!(eigenfunction_errors, error)
            @test error < 0.35
        end
        noise_relative_error = abs(noise_variance(model) / sigma2 - 1)
        @test noise_relative_error < 0.4
        @info "Default PACE sparse Fourier recovery" mean_L2_error=mean_error eigenfunction_L2_errors=eigenfunction_errors noise_relative_error
    end

    @testset "Dense and PACE agree for densely observed smooth curves" begin
        grid = collect(range(0.0, 1.0; length=25))
        functions = legendre_eigenfunctions((0.0, 1.0), 2)
        simulation = simulate_kl(MersenneTwister(2026102404), 80; mean=t -> 1 + t,
            eigenfunctions=functions, eigenvalues=[2.0, 0.7], noise_sd=0.15,
            sampling=RegularSampling(grid))
        dense = StatsAPI.fit(FPCA, simulation.data; ncomponents=2)
        pace = StatsAPI.fit(FPCA, simulation.data; method=:pace, ncomponents=2)
        # Covariance smoothing, diagonal exclusion, and measurement noise
        # make this a cross-method agreement check rather than an identity.
        eigenvalue_relative_error = maximum(abs.((eigvals(pace) - eigvals(dense)) ./ eigvals(dense)))
        @test eigenvalue_relative_error < 0.15
        agreements = Float64[]
        for k in 1:2
            dense_values = basismatrix(dense.basis, pace.grid)[:, k]
            dense_values ./= sqrt(dot(pace.weights, dense_values.^2))
            agreement = abs(dot(pace.weights, dense_values .* pace.basis.values[:, k]))
            push!(agreements, agreement)
            @test agreement > 0.98
        end
        @info "Dense versus PACE agreement" eigenvalue_relative_error eigenfunction_absolute_correlations=agreements
    end

    @testset "Default sparse fit runtime" begin
        # Earlier fits compile the default route. The 120 second bound is
        # deliberately generous for CI and guards against accidental CV or
        # repeated all-pairs searches in the no-keyword path.
        simulation = simulate_kl(MersenneTwister(2026102405), 500;
            mean=t -> 1 + t, eigenfunctions=fourier_eigenfunctions((0.0, 1.0), 2),
            eigenvalues=[2.0, 0.7], noise_sd=0.5,
            sampling=SparseSampling((0.0, 1.0), 2:6))
        elapsed = @elapsed model = StatsAPI.fit(FPCA, simulation.data)
        @info "Default PACE fit: n=500 sparse curves" elapsed_seconds=elapsed
        @test elapsed < 120
        @test nobs(model) == 500
        @test model.settings.mean_input.folds == 0
        @test model.settings.covariance_input.folds == 0
        @test model.settings.requested_binning == :auto
        @test model.settings.binning === nothing
    end
end
