import KernelSmoothing
import StatsAPI

@testset "PACE FPCA" begin
    @testset "Analytic conditional expectations and full covariance" begin
        grid = [0.0, 0.2, 0.5, 0.8, 1.0]
        weights = quadrature_weights(grid)
        slope_norm = sqrt(dot(weights, (grid .- 0.5).^2))
        phi(t) = [1.0, (t - 0.5) / slope_norm]
        lambda = [2.0, 0.7]
        mu(t) = 1 + t
        source_grid = [0.0, 0.5, 1.0]
        source_phi = reduce(vcat, [transpose(phi(t)) for t in source_grid])
        surface = Matrix(Symmetric(source_phi * Diagonal(lambda) * source_phi'))
        supplied_mean = (; grid=[-0.1, 0.5, 1.1], values=mu.([-0.1, 0.5, 1.1]))
        supplied_covariance = (; grid=source_grid, surface)
        times = [[0.0, 0.2, 0.65, 1.0], [0.1, 0.4, 0.9],
            [0.05, 0.5, 0.7, 0.95], [0.25, 0.75]]
        values = [mu.(t) .+ [dot(phi(x), [i / 3, (-1)^i / 2]) for x in t]
            for (i, t) in enumerate(times)]
        fd = IrregularFunctionalData(times, values; domain=(0.0, 1.0),
            ids=["curve-$i" for i in eachindex(times)])
        sigma2 = 0.13
        model = @inferred StatsAPI.fit(FPCA, fd; grid, ncomponents=1, max_k=2,
            mean=supplied_mean, covariance=supplied_covariance, noise_variance=sigma2)
        @test model.method == :pace
        _test_prediction_contract(model, fd)
        # Entirely unobserved grid columns need not lie in the PACE work grid.
        masked_values = Matrix{Union{Missing,Float64}}(missing, 4, 5)
        masked_values[:,2:4] = [mu(t) + i/3 for i in 1:4, t in source_grid]
        masked = GridFunctionalData(masked_values, [-0.1, 0.0, 0.5, 1.0, 1.1]; ids=ids(fd))
        masked_model = StatsAPI.fit(FPCA, masked; method=:pace, grid,
            ncomponents=1, max_k=2, mean=supplied_mean,
            covariance=supplied_covariance, noise_variance=sigma2)
        _test_prediction_contract(masked_model, masked)
        @test model.design == :sparse
        @test nobs(model) == 4
        @test ncomponents(model) == 1
        @test model.all_eigenvalues ≈ lambda
        @test eigvals(model) ≈ lambda[1:1]
        @test model.basis.values[:, 1] ≈ ones(length(grid))
        @test model.weights == weights
        @test model.grid !== grid
        @test (@inferred noise_variance(model)) == sigma2
        @test ids(model) == ids(fd)
        @test ids(model) !== ids(fd)
        @test model.settings.mean_source == :supplied
        @test model.settings.covariance_source == :supplied
        @test model.settings.noise_variance_source == :supplied
        @test model.settings.noise_floor === nothing
        @test model.settings.noise_floor_relative === nothing
        @test model.settings.noise_floor_scale === nothing
        @test model.settings.noise_floor_scale_source === nothing
        @test !model.settings.noise_floor_applied

        expected_scores = zeros(4, 1)
        expected_covariances = zeros(1, 1, 4)
        selected_only = zeros(4)
        for i in eachindex(times)
            functions = reduce(vcat, [transpose(phi(t)) for t in times[i]])
            residual = values[i] - mu.(times[i])
            observation_covariance = functions * Diagonal(lambda) * functions' + sigma2 * I
            numerator = lambda[1] * functions[:, 1]'
            expected_scores[i, 1] = only(numerator * (observation_covariance \ residual))
            expected_covariances[1, 1, i] = lambda[1] -
                only(numerator * (observation_covariance \ numerator'))
            reduced_covariance = lambda[1] * functions[:, 1] * functions[:, 1]' + sigma2 * I
            selected_only[i] = only(numerator * (reduced_covariance \ residual))
        end
        @test (@inferred scores(model)) ≈ expected_scores
        @test (@inferred score_covariances(model)) ≈ expected_covariances
        @test norm(vec(expected_scores) - selected_only) > 0.01
        predicted_scores, predicted_covariances = @inferred StatsAPI.predict(model, fd)
        @test predicted_scores == scores(model)
        @test predicted_covariances == score_covariances(model)
        @test predicted_scores !== scores(model)
        @test predicted_covariances !== score_covariances(model)

        target = [0.15, 0.45, 0.85]
        new_values = reduce(vcat, [transpose(mu.(target) .+ i) for i in 1:2])
        grid_data = GridFunctionalData(new_values, target)
        grid_scores, grid_covariances = @inferred StatsAPI.predict(model, grid_data)
        irregular_scores, irregular_covariances = StatsAPI.predict(model, to_irregular(grid_data))
        @test grid_scores ≈ irregular_scores
        @test grid_covariances ≈ irregular_covariances
        restored = @inferred reconstruct(model)
        @test restored.values ≈ expected_scores * ones(1, length(grid)) .+ transpose(mu.(grid))
        @test ids(restored) == ids(fd)
        @test ids(reconstruct(model, copy(scores(model)))) === nothing
        @test reconstruct(model; grid=target).values ≈
            expected_scores * ones(1, length(target)) .+ transpose(mu.(target))

        displayed = sprint(show, model)
        for choice in ("pace", "sparse", "selection", "max_k", "mean_source",
                "covariance_source", "noise_variance_source", "kernel", "weighting",
                "binning", "eigen_tol", "sign_tol", "dropped", "noise_variance",
                "noise_floor_relative", "noise_floor_scale", "noise_floor_scale_source")
            @test occursin(choice, displayed)
        end
        scale = 1e-6
        scaled_data = IrregularFunctionalData(times, [scale .* y for y in values];
            domain=(0.0, 1.0))
        scaled_model = StatsAPI.fit(FPCA, scaled_data; grid, ncomponents=1, max_k=2,
            mean=(; grid=supplied_mean.grid, values=scale .* supplied_mean.values),
            covariance=(; grid=source_grid, surface=scale^2 .* surface),
            noise_variance=scale^2 * sigma2)
        @test scores(scaled_model) ≈ scale .* scores(model)
        @test score_covariances(scaled_model) ≈ scale^2 .* score_covariances(model)
        # The fit owns every user-supplied array, including its settings snapshots.
        supplied_mean.values .= 100
        supplied_covariance.surface .= 100
        grid .= -100
        @test vec(model.mean.values) ≈ mu.(model.grid)
        @test model.settings.mean_input.values != supplied_mean.values
        @test model.settings.covariance_input.surface != supplied_covariance.surface
        @test StatsAPI.predict(model, fd)[1] == scores(model)
    end

    @testset "Estimated noise floor and explicit overrides" begin
        grid = collect(0.0:0.25:1.0)
        mean = (; grid, values=zeros(5))
        times = [[0.0, 0.4, 1.0], [0.1, 0.9]]
        observations = [[1.0, 2.0, 3.0], [-1.0, -2.0]]
        data = IrregularFunctionalData(times, observations)
        # A fitted covariance with an analytic rank-one surface isolates the
        # noise policy from smoothing. Its raw estimates cover the boundary.
        function covariance(raw; value_scale=1.0, time_scale=1.0)
            local_grid = time_scale .* grid
            SmoothedCovariance(local_grid, fill(value_scale^2, 5, 5),
                quadrature_weights(local_grid), 1.0, 1.0,
                KernelSmoothing.GaussianKernel(), :obs, nothing, 1,
                :sparse, 0, nothing, max(raw, 0.0), (; raw_variance=raw))
        end
        reference = StatsAPI.fit(FPCA, data; grid, mean,
            covariance=covariance(0.0), ncomponents=1, maxK=1)
        # Obtain the exact floating-point threshold to exercise equality,
        # while checking its independently known analytic value below.
        threshold = reference.settings.noise_floor
        @test threshold ≈ 0.025 rtol=1e-12
        for raw in (-0.2, 0.0, 1e-12, threshold/2, threshold, 0.2)
            fitted_covariance = covariance(raw)
            model = @inferred StatsAPI.fit(FPCA, data; grid, mean,
                covariance=fitted_covariance, ncomponents=1, maxK=1)
            used = max(raw, threshold)
            @test noise_variance(model) == used
            @test model.settings.raw_noise_variance == raw
            @test model.settings.noise_floor == threshold
            @test model.settings.noise_floor_relative == 0.025
            @test model.settings.noise_floor_scale ≈ 1.0 rtol=1e-12
            @test model.settings.noise_floor_scale_source == :maximum_fitted_pointwise_variance
            @test model.settings.noise_floor_applied == (raw < threshold)
            @test model.settings.noise_variance_source == :estimated
            @test fitted_covariance.noise_variance == max(raw, 0.0)
            for i in 1:nobs(data)
                _, y = data[i]
                # For C=11', the posterior mean is sum(y)/(m+sigma2)
                # and the posterior score variance is sigma2/(m+sigma2).
                @test scores(model)[i, 1] ≈ sum(y)/(length(y)+used) rtol=1e-8
                @test score_covariances(model)[1, 1, i] ≈ used/(length(y)+used) rtol=1e-8
            end
            prediction = StatsAPI.predict(model, data)
            @test prediction.scores == scores(model)
            @test prediction.covariances == score_covariances(model)
        end
        # Changing observation units scales variances quadratically and the
        # posterior reconstruction linearly, including when the floor binds.
        for value_scale in (1e-6, 1e6)
            scaled = IrregularFunctionalData(times,
                [value_scale .* y for y in observations])
            model = StatsAPI.fit(FPCA, scaled; grid, mean,
                covariance=covariance(0.0; value_scale), ncomponents=1, maxK=1)
            @test model.settings.noise_floor_applied
            @test noise_variance(model) ≈ value_scale^2 * noise_variance(reference) rtol=1e-12
            @test reconstruct(model).values ≈ value_scale .* reconstruct(reference).values rtol=1e-12
            @test score_covariances(model) ≈ value_scale^2 .* score_covariances(reference) rtol=1e-10
        end
        # Changing time units changes integrated eigenvalues but leaves the
        # pointwise variance scale and posterior reconstruction unchanged.
        time_scale = 10.0
        time_grid = time_scale .* grid
        time_data = IrregularFunctionalData([time_scale .* t for t in times], observations)
        time_model = StatsAPI.fit(FPCA, time_data; grid=time_grid,
            mean=(; grid=time_grid, values=zeros(5)),
            covariance=covariance(0.0; time_scale), ncomponents=1, maxK=1)
        @test noise_variance(time_model) ≈ noise_variance(reference) rtol=1e-12
        @test time_model.settings.noise_floor_scale ≈ 1.0 rtol=1e-12
        @test only(eigvals(time_model)) ≈ time_scale * only(eigvals(reference)) rtol=1e-12
        @test reconstruct(time_model).values ≈ reconstruct(reference).values rtol=1e-12
        # The floor also precedes the candidate likelihood solves, whose
        # one-component observation covariances are rank deficient at zero.
        for rule in (AIC(), BIC())
            model = StatsAPI.fit(FPCA, data; grid, mean, covariance=covariance(0.0),
                ncomponents=rule, maxK=2)
            @test model.settings.noise_floor_applied
            @test all(isfinite, model.selection.scores)
        end
        single = IrregularFunctionalData([[0.3], [0.7]], [[2.0], [-1.0]])
        for supplied in (0.0, 1e-12)
            model = @inferred StatsAPI.fit(FPCA, single; grid, mean,
                covariance=covariance(-0.2), noise_variance=supplied,
                ncomponents=1, maxK=1)
            @test noise_variance(model) == supplied
            @test model.settings.noise_floor === nothing
            @test model.settings.noise_floor_relative === nothing
            @test model.settings.noise_floor_scale === nothing
            @test model.settings.noise_floor_scale_source === nothing
            @test !model.settings.noise_floor_applied
            @test model.settings.noise_variance_source == :supplied
            @test vec(scores(model)) ≈ [2.0, -1.0] ./ (1+supplied) rtol=1e-12
        end
    end

    @testset "AIC and BIC: complete likelihood curve and first increase" begin
        grid = [0.0, 0.5, 1.0]
        functions = [1.0 sqrt(2.0) 1.0; 1.0 0.0 -1.0; 1.0 -sqrt(2.0) 1.0]
        lambda = [3.0, 2.0, 1.0]
        surface = Matrix(Symmetric(functions * Diagonal(lambda) * functions'))
        values = repeat(transpose(5 .* functions[:, 3]), 8, 1)
        data = GridFunctionalData(values, grid)
        sigma2 = 0.25
        for (rule, penalty) in ((AIC(), 2.0), (BIC(), log(8)))
            model = @inferred StatsAPI.fit(FPCA, data; method=:pace, grid,
                mean=(; grid, values=zeros(3)), covariance=(; grid, surface),
                noise_variance=sigma2, max_k=3, ncomponents=rule)
            criteria = map(1:3) do k
                covariance = functions[:, 1:k] * Diagonal(lambda[1:k]) * functions[:, 1:k]' + sigma2 * I
                factor = cholesky(Symmetric(covariance))
                likelihood = sum(3log(2pi) + logdet(factor) + dot(y, factor \ y)
                    for y in eachrow(values))
                likelihood + penalty * k
            end
            @test model.selection.scores ≈ criteria
            @test model.selection.candidates == 1:3
            @test criteria[2] > criteria[1]
            @test argmin(criteria) == 3
            @test ncomponents(model) == model.selection.chosen == 1
            @test model.selection.global_minimizer == 3
            @test model.selection.stopping == :first_increase
        end
        rank_one_surface = ones(3, 3)
        for (rule, penalty) in ((AIC(), 2.0), (BIC(), log(8)))
            model = StatsAPI.fit(FPCA, data; method=:pace, grid,
                mean=(; grid, values=zeros(3)), covariance=(; grid, surface=rank_one_surface),
                noise_variance=sigma2, max_k=3, ncomponents=rule)
            @test length(model.all_eigenvalues) == 1
            @test model.selection.candidates == 1:3
            @test diff(model.selection.scores) ≈ fill(penalty, 2)
            @test ncomponents(model) == model.selection.global_minimizer == 1
        end
    end

    @testset "Routes, validation, and singular conditional systems" begin
        grid = collect(0.0:0.25:1.0)
        mu = (; grid, values=1 .+ grid)
        surface = ones(5, 5)
        covariance = (; grid, surface)
        fd = IrregularFunctionalData([[0.0, 0.3, 1.0], [0.1, 0.8]],
            [[1.0, 1.4, 2.0], [1.2, 1.6]])
        options = (; grid, mean=mu, covariance, noise_variance=0.2, max_k=1)
        @test StatsAPI.fit(FPCA, fd; options...).method == :pace
        dense_times = [collect(range(0.0, 1.0; length=m)) for m in (22, 23)]
        dense_irregular = IrregularFunctionalData(dense_times, [1 .+ t for t in dense_times])
        @test sampling_design(dense_irregular) == :dense_irregular
        @test StatsAPI.fit(FPCA, dense_irregular; options...).method == :pace
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:dense, options...)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:invalid, options...)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd[1:1]; options...)
        @test StatsAPI.fit(FPCA, fd; mean=mu, covariance,
            noise_variance=0.2, max_k=1).settings.grid_source == :observed_range_51
        @test StatsAPI.fit(FPCA, fd; grid, covariance,
            noise_variance=0.2, max_k=1).settings.mean_bandwidth_source == :default_fraction
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
            noise_variance=0.2, max_k=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance_bandwidth=1.0, noise_variance=0.2, max_k=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance, max_k=1)
        for noise in (-1.0, Inf, NaN, :invalid)
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
                covariance, noise_variance=noise, max_k=1)
        end
        short_mean = (; grid=[0.2, 0.8], values=[1.2, 1.8])
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=short_mean,
            covariance, noise_variance=0.2, max_k=1)
        asymmetric = copy(surface)
        asymmetric[1, 2] += 0.1
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance=(; grid, surface=asymmetric), noise_variance=0.2, max_k=1)
        rounded = copy(surface)
        rounded[1, 2] *= 1 + 4eps()
        near = StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance=(; grid, surface=rounded), noise_variance=0.2, max_k=1)
        exact = StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance=(; grid, surface), noise_variance=0.2, max_k=1)
        @test eigvals(near) ≈ eigvals(exact) rtol=1e-12
        @test_throws DimensionMismatch StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance=(; grid, surface=ones(4, 4)), noise_variance=0.2, max_k=1)
        @test_throws DimensionMismatch StatsAPI.fit(FPCA, fd; grid,
            mean=(; grid, values=zeros(4)), covariance, noise_variance=0.2, max_k=1)
        nonfinite = copy(surface)
        nonfinite[2, 2] = NaN
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; grid, mean=mu,
            covariance=(; grid, surface=nonfinite), noise_variance=0.2, max_k=1)
        unordered = UInt[0, 2, 1, 3]
        for supplied in ((; mean=(; grid=unordered, values=[1.0, 3.0, 2.0, 4.0]), covariance),
                (; mean=mu, covariance=(; grid=unordered, surface=ones(4, 4))))
            error = try
                StatsAPI.fit(FPCA, fd; grid, supplied..., noise_variance=0.2, max_k=1)
            catch exception
                exception
            end
            @test error isa ArgumentError
            @test occursin("increasing", sprint(showerror, error))
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; options..., presmooth=LocalPolynomial(; bandwidth=0.5))
        singular = try
            StatsAPI.fit(FPCA, fd; grid, mean=mu, covariance, noise_variance=0.0, max_k=1)
        catch error
            error
        end
        @test singular isa ArgumentError
        @test occursin("curve 1", sprint(showerror, singular))
        @test occursin("noise_variance", sprint(showerror, singular))
        model = StatsAPI.fit(FPCA, fd; options...)
        outside = IrregularFunctionalData([[-0.01, 0.5]], [[0.0, 1.0]])
        @test_throws ArgumentError StatsAPI.predict(model, outside)
        @test_throws ArgumentError StatsAPI.predict(model,
            GridFunctionalData(ones(1, 2), [0.5, 1.01]))

        masked = GridFunctionalData(Union{Missing,Float64}[
            1 missing 2 2 2; missing 1 1 missing 2], grid; ids=[:a, :b])
        masked_fit = StatsAPI.fit(FPCA, masked; options...)
        irregular_fit = StatsAPI.fit(FPCA, to_irregular(masked); options...)
        @test masked_fit.method == :pace
        @test ids(masked_fit) == [:a, :b]
        @test scores(masked_fit) == scores(irregular_fit)
        @test score_covariances(masked_fit) == score_covariances(irregular_fit)
        @test_throws ArgumentError StatsAPI.fit(FPCA, masked; method=:dense)
        grid32 = Float32.(grid)
        data32 = GridFunctionalData(Float32[1 2 3 4 5; 2 3 4 5 6], grid32)
        fit32 = @inferred StatsAPI.fit(FPCA, data32; method=:pace, grid=grid32,
            mean=(; grid=grid32, values=1f0 .+ grid32),
            covariance=(; grid=grid32, surface=ones(Float32, 5, 5)),
            noise_variance=0.2f0, max_k=1)
        @test eltype(scores(fit32)) == Float32
        @test eltype(score_covariances(fit32)) == Float32
        @test eltype(eigvals(fit32)) == Float32
    end

    @testset "Reused pooled smoothers and source settings" begin
        grid = collect(0.0:0.125:1.0)
        times = [collect(range(0.0, 1.0; length=m)) for m in (7, 8, 9, 10)]
        values = [1 .+ t .+ (-1)^i .* cos.(2t) for (i, t) in enumerate(times)]
        fd = IrregularFunctionalData(times, values)
        mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=1.2, weighting=:subj)
        covariance = StatsAPI.fit(SmoothedCovariance, fd, mean;
            grid, bandwidth=1.2, weighting=:subj, binning=5)
        @test covariance.noise_details.noise_method == :rotated
        @test covariance.noise_details.noise_interval == :full
        # A positive explicit noise variance separates integration of the
        # smoothers from invertibility of their estimated latent covariance.
        estimated = @inferred StatsAPI.fit(FPCA, fd; grid, ncomponents=1, max_k=2,
            mean_bandwidth=1.2, covariance_bandwidth=1.2,
            kernel=KernelSmoothing.EpanechnikovKernel(),
            noise_variance=0.2, weighting=:subj, binning=5)
        supplied = @inferred StatsAPI.fit(FPCA, fd; grid, ncomponents=1, max_k=2,
            mean, covariance, noise_variance=0.2)
        @test vec(estimated.mean.values) ≈ mean.values
        @test estimated.covariance ≈ covariance.surface
        @test scores(estimated) ≈ scores(supplied)
        @test estimated.settings.mean_source == :estimated
        @test estimated.settings.covariance_source == :estimated
        @test estimated.settings.noise_variance_source == :supplied
        @test supplied.settings.mean_source == :supplied
        @test supplied.settings.covariance_source == :supplied
        @test supplied.settings.mean_input.bandwidth == 1.2
        @test supplied.settings.covariance_input.binning == 5
        @test covariance.noise_variance > 0
        inherited = StatsAPI.fit(FPCA, fd; grid, ncomponents=1, max_k=2, mean, covariance)
        @test noise_variance(inherited) == covariance.noise_variance
        @test inherited.settings.noise_variance_source == :estimated
        estimated_noise = StatsAPI.fit(FPCA, fd; grid, ncomponents=1, max_k=2,
            mean_bandwidth=1.2, covariance_bandwidth=1.2,
            kernel=KernelSmoothing.EpanechnikovKernel(), weighting=:subj, binning=5)
        @test noise_variance(estimated_noise) == covariance.noise_variance
        @test estimated_noise.settings.noise_variance_source == :estimated
    end

    @testset "Sparse Gaussian conditional covariance calibration" begin
        n = 3000
        grid = collect(range(0.0, 1.0; length=41))
        functions = legendre_eigenfunctions((0.0, 1.0), 2)
        lambda = [1.5, 0.6]
        sigma2 = 0.2
        mu(t) = 0.5 + t
        simulation = simulate_kl(MersenneTwister(20261024), n; mean=mu,
            eigenfunctions=functions, eigenvalues=lambda, noise_sd=sqrt(sigma2),
            sampling=SparseSampling((0.0, 1.0), 3:8))
        truth_functions = [f(t) for t in grid, f in functions]
        surface = Matrix(Symmetric(truth_functions * Diagonal(lambda) * truth_functions'))
        model = StatsAPI.fit(FPCA, simulation.data; grid, ncomponents=2, max_k=2,
            mean=(; grid, values=mu.(grid)), covariance=(; grid, surface),
            noise_variance=sigma2)
        # The true affine functions are exactly interpolated. Convert their
        # continuous-normalized scores to this fit's trapezoidal coordinates.
        rotation = truth_functions' * (model.weights .* model.basis.values)
        true_scores = simulation.scores * rotation
        errors = scores(model) - true_scores
        conditional_covariances = score_covariances(model)
        mse = sum(abs2, errors) / n
        predicted_mse = sum(tr(conditional_covariances[:, :, i]) for i in 1:n) / n
        # For 3000 independent Gaussian curves, 10% permits several Monte
        # Carlo standard errors without relying on exact RNG stream values.
        @test abs(mse / predicted_mse - 1) < 0.10
        for k in 1:2
            standardized = errors[:, k] ./ sqrt.(conditional_covariances[k, k, :])
            @test abs(sum(standardized) / n) < 0.08
            centered = standardized .- sum(standardized) / n
            @test abs(sum(abs2, centered) / (n - 1) - 1) < 0.10
        end
    end
end
