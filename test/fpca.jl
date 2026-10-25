import StatsAPI
using DelimitedFiles

module StatsAPIReconstructionClient
import StatsAPI: reconstruct
end

@testset "FPCA" begin
    @testset "fdapace 0.6.0 dense golden values" begin
        input = readdlm(joinpath(@__DIR__, "data", "m3_dense_input.csv"), ','; comments=true)
        golden = readdlm(joinpath(@__DIR__, "data", "m3_fdapace_dense.csv"), ',', Any; comments=true)
        grid = Float64.(input[input[:, 1] .== 1, 2])
        n = Int(maximum(input[:, 1]))
        values = reduce(vcat, [transpose(Float64.(input[input[:, 1] .== i, 3])) for i in 1:n])
        fd = GridFunctionalData(values, grid)
        model = StatsAPI.fit(FPCA, fd; ncomponents=3, max_k=5)
        reference(quantity) = Float64.(golden[golden[:, 1] .== quantity, 4])
        relative_error(actual, expected) = norm(actual - expected) / norm(expected)
        @test relative_error(vec(model.mean.values), reference("mean")) < 1e-8
        @test relative_error(model.covariance,
            reshape(reference("covariance"), length(grid), length(grid))) < 1e-8

        # fdapace eigendecomposes G*h and then normalizes each eigenfunction
        # with trapezoidal integration. Public fits use trapezoids for both.
        # Reproduce the audited R convention explicitly for golden comparison.
        rectangle = fill(grid[2] - grid[1], length(grid))
        compatible = FunctionalDataAnalysis._fpca_eigen(model.covariance,
            rectangle, vec(model.mean.values); max_k=5,
            normalization_weights=fd.weights)
        @test relative_error(compatible.eigenvalues[1:3], reference("eigenvalue")) < 1e-8
        @test relative_error(compatible.eigenvalues, reference("all_eigenvalue")) < 1e-8
        @test relative_error(compatible.eigenfunctions[:, 1:3],
            reshape(reference("eigenfunction"), length(grid), 3)) < 1e-8
        @test relative_error(compatible.fve, reference("fve")) < 1e-8
        compatible_scores = (values .- model.mean.values) *
            (fd.weights .* compatible.eigenfunctions[:, 1:3])
        @test relative_error(compatible_scores,
            reshape(reference("score"), n, 3)) < 1e-8
        @test ncomponents(model) == Int(reference("selected_k")[1])
        @test findfirst(>=(0.99), compatible.fve) == Int(reference("selected_k")[2])

        # For these deterministic cosine curves, trapezoidal orthonormality
        # gives the exact finite-sample spectrum with denominator n-1.
        expected_lambda = n / (n - 1) .* [4.0, 1.0, 0.3, 0.08, 0.02]
        @test model.all_eigenvalues ≈ expected_lambda rtol=1e-12
        @test relative_error(model.all_eigenvalues, reference("all_eigenvalue")) > 1e-3
    end

    @testset "Exact nonuniform-grid covariance and scores" begin
        grid = [0.0, 0.1, 0.35, 0.7, 1.0]
        weights = quadrature_weights(grid)
        firstphi = ones(length(grid))
        secondphi = (grid .- 0.5) / sqrt(sum(weights .* (grid .- 0.5).^2))
        truebasis = hcat(firstphi, secondphi)
        a = [-2.0, -1, 0, 1, 2, 0]
        b = [1.0, -1, 1, -1, 1, -1]
        true_scores = hcat(a, b)
        meanvalues = 2firstphi + 3secondphi
        values = transpose(meanvalues) .+ true_scores * transpose(truebasis)
        labels = ["subject-$i" for i in 1:6]
        fd = GridFunctionalData(values, grid; ids=labels)
        model = @inferred (data -> StatsAPI.fit(FPCA, data;
            method=:dense, ncomponents=2, max_k=3))(fd)
        @test FunctionalDataAnalysis.reconstruct === StatsAPI.reconstruct
        @test StatsAPIReconstructionClient.reconstruct === FunctionalDataAnalysis.reconstruct
        @test StatsAPIReconstructionClient.reconstruct(model).values ≈ values
        @test model.method == :dense
        _test_prediction_contract(model, fd)
        @test model.design == :regular
        @test model.grid == grid
        @test model.weights == weights
        @test model.grid !== fd.grid
        @test model.weights !== fd.weights
        @test model.mean.values ≈ reshape(meanvalues, 1, :)
        @test model.covariance ≈ truebasis * Diagonal([2.0, 1.2]) * transpose(truebasis)
        @test (@inferred eigenfunctions(model)) === model.basis
        @test (@inferred eigvals(model)) ≈ [2.0, 1.2]
        @test model.all_eigenvalues ≈ [2.0, 1.2]
        @test (@inferred fve(model)) ≈ [0.625, 1.0]
        @test (@inferred scores(model)) ≈ true_scores atol=2e-14
        @test (@inferred score_covariances(model)) === nothing
        @test (@inferred noise_variance(model)) === nothing
        @test (@inferred ncomponents(model)) == 2
        @test (@inferred nobs(model)) == 6
        @test ids(model) == labels
        phi = basis_matrix(eigenfunctions(model), grid)
        @test phi ≈ truebasis atol=2e-14
        @test transpose(phi) * (weights .* phi) ≈ Matrix{Float64}(I, 2, 2)
        @test transpose(scores(model)) * scores(model) / (nobs(model) - 1) ≈ Diagonal(eigvals(model))
        @test model.selection.rule == :fixed
        @test model.selection.chosen == 2
        @test model.selection.max_k == 3
        @test model.selection.threshold === nothing
        @test isempty(model.selection.scores)
        @test model.presmooth === nothing

        @test (@inferred StatsAPI.predict(model, fd)) ≈ scores(model)
        restored = @inferred reconstruct(model)
        @test restored.values ≈ values atol=2e-14
        @test ids(restored) == labels
        @test ids(reconstruct(model, copy(scores(model)))) === nothing
        finegrid = collect(range(0, 1; length=21))
        finer = reconstruct(model; grid=finegrid)
        # The generating functions and mean are linear, so interpolation is exact.
        finesecond = (finegrid .- 0.5) / sqrt(sum(weights .* (grid .- 0.5).^2))
        expected = transpose(2ones(21) + 3finesecond) .+
            true_scores * transpose(hcat(ones(21), finesecond))
        @test finer.values ≈ expected atol=3e-14
        @test ids(finer) == labels
        @test_throws ArgumentError reconstruct(model; grid=[-0.1, 0.5])
        @test_throws DimensionMismatch reconstruct(model, zeros(6, 3))
        @test_throws ArgumentError reconstruct(model, fill(NaN, 1, 2))
        @test_throws ArgumentError StatsAPI.predict(model, GridFunctionalData(values, grid .+ 0.001))
        @test_throws DimensionMismatch StatsAPI.predict(model, GridFunctionalData(values[:, 1:4], grid[1:4]))

        @test StatsAPI.fit(FPCA, fd; ncomponents=FVE(0.6), max_k=3).selection.chosen == 1
        boundary = fve(model)[1]
        atboundary = StatsAPI.fit(FPCA, fd; ncomponents=FVE(boundary), max_k=3)
        aboveboundary = StatsAPI.fit(FPCA, fd; ncomponents=FVE(nextfloat(boundary)), max_k=3)
        @test ncomponents(atboundary) == 1
        @test ncomponents(aboveboundary) == 2
        @test atboundary.selection.rule == :FVE
        @test atboundary.selection.threshold == boundary
        @test atboundary.selection.scores == fve(atboundary)
        @test ncomponents(StatsAPI.fit(FPCA, fd; ncomponents=FVE(1), max_k=3)) == 2
        capped = StatsAPI.fit(FPCA, fd; ncomponents=FVE(0.99), max_k=1)
        @test ncomponents(capped) == 1
        @test capped.all_eigenvalues ≈ [2.0]
        @test fve(capped) == [1.0]
        default = StatsAPI.fit(FPCA, fd)
        @test default.selection.threshold == 0.99
        @test default.selection.max_k == 3
        for field in ("method", "design", "selection", "dropped", "sign_rule", "presmooth")
            @test occursin(field, sprint(show, model))
        end

        # A fit owns its data-derived arrays and copied IDs.
        fd.values[1, 1] = 1000
        fd.grid[2] = 0.2
        fd.weights[1] = 1000
        fd.ids[1] = "changed"
        @test model.grid == grid
        @test model.weights == weights
        @test model.mean.values ≈ reshape(meanvalues, 1, :)
        @test ids(model) == labels
    end

    @testset "Dense default retains more than twenty components" begin
        grid = collect(range(0, 1; length=26))
        weights = quadrature_weights(grid)
        # Nonconstant Walsh columns have zero means and orthogonal scores.
        # Coordinate functions scaled by quadrature weights make the full
        # sample covariance spectrum known, with 24 substantial components.
        signs = [isodd(count_ones(i & j)) ? -1.0 : 1.0 for i in 0:31, j in 1:24]
        eigenvalues = 1 .- collect(0:23) ./ 48
        truebasis = Matrix{Float64}(I, 26, 24) ./ sqrt.(weights)
        true_scores = signs .* transpose(sqrt.(eigenvalues .* (31 / 32)))
        fd = GridFunctionalData(2 .+ true_scores * transpose(truebasis), grid)
        model = StatsAPI.fit(FPCA, fd)
        @test model.selection.max_k == 24 # min(m-2, n-2), without a cap of 20.
        @test model.all_eigenvalues ≈ eigenvalues rtol=1e-12
        @test fve(model) ≈ cumsum(eigenvalues) ./ sum(eigenvalues) rtol=1e-12
        @test ncomponents(model) == 24
        @test ncomponents(StatsAPI.fit(FPCA, fd; max_k=20)) == 20
    end

    @testset "Eigenvalue cutoff, sign rules, and quadrature convergence" begin
        step = FunctionalDataAnalysis._fpca_eigen
        aligned = step(Matrix(Diagonal([3.0, 1.0])), [1.0, 1.0], [-1.0, 1.0]; max_k=2)
        @test aligned.eigenvalues == [3.0, 1.0]
        @test aligned.eigenfunctions ≈ [-1.0 0.0; 0.0 1.0]
        fallback = step(Matrix(Diagonal([3.0, 1.0])), [1.0, 1.0], zeros(2); max_k=2)
        @test fallback.eigenfunctions ≈ Matrix{Float64}(I, 2, 2)
        # Mean inner products below the stated tolerance also use the fallback.
        tinymean = step(Matrix(Diagonal([3.0, 1.0])), [1.0, 1.0], [-1e-15, 1.0];
            max_k=2, sign_tol=1e-12)
        @test tinymean.eigenfunctions ≈ Matrix{Float64}(I, 2, 2)
        projected = step(Matrix(Diagonal([3.0, -2.0, 1e-12])), ones(3), ones(3);
            max_k=3, eigen_tol=1e-10)
        @test projected.eigenvalues == [3.0]
        @test projected.fve == [1.0]
        @test projected.dropped.number == 1
        @test projected.dropped.total == 2.0
        @test projected.dropped.share ≈ 2 / (5 + 1e-12)
        @test_throws ArgumentError step(-Matrix{Float64}(I, 2, 2), ones(2), ones(2); max_k=2)

        # Finite inputs can have a spectrum whose direct sum overflows.
        large = step(Matrix(Diagonal([1e308, 9e307])), ones(2), ones(2); max_k=2)
        @test large.fve[1] ≈ 1 / 1.9
        @test large.fve[end] == 1.0
        indefinite = step(Matrix(Diagonal([1e308, -1e308])), ones(2), ones(2); max_k=2)
        @test indefinite.dropped.total == 1e308
        @test indefinite.dropped.share ≈ 0.5

        # Tiny positive weights produce large but representable eigenfunctions.
        # Their squares overflow, but their weighted norms remain exactly one.
        tinyweights = fill(1e-310, 2)
        small_weighted = step(Matrix(Diagonal([1e308, 5e307])), tinyweights,
            ones(2); max_k=2)
        @test all(isfinite, small_weighted.eigenfunctions)
        @test all(k -> maximum(abs, small_weighted.eigenfunctions[:, k]) > 0, 1:2)
        rootweights = sqrt.(tinyweights)
        for k in 1:2
            @test norm(rootweights .* small_weighted.eigenfunctions[:, k]) ≈ 1.0
        end
        @test dot(rootweights .* small_weighted.eigenfunctions[:, 1],
            rootweights .* small_weighted.eigenfunctions[:, 2]) ≈ 0.0
        # Relative mean alignment is unchanged by finite rescaling, even when
        # the unscaled alignment and product of norms would overflow.
        scaled_sign = step(Matrix(Diagonal([1e308, 5e307])), tinyweights,
            [-1e308, 1e308]; max_k=2)
        @test scaled_sign.eigenfunctions[1, 1] < 0
        @test scaled_sign.eigenfunctions[2, 2] > 0
        scaled_fallback = step(Matrix(Diagonal([1e308, 5e307])), tinyweights,
            [-1e294, 1e308]; max_k=2, sign_tol=1e-12)
        @test scaled_fallback.eigenfunctions[1, 1] > 0

        errors = Float64[]
        for m in (11, 21, 41)
            grid = collect(range(0, 1; length=m))
            trap = quadrature_weights(grid)
            surface = ones(m, m) # Kernel G(s,t)=1 has eigenvalue exactly 1.
            ours = step(surface, trap, ones(m); max_k=1)
            rectangle = fill(1 / (m - 1), m)
            compatible = step(surface, rectangle, ones(m); max_k=1, normalization_weights=trap)
            @test only(ours.eigenvalues) ≈ 1.0
            @test only(compatible.eigenvalues) ≈ 1 + 1 / (m - 1)
            @test compatible.eigenfunctions[:, 1] ≈ ones(m)
            push!(errors, abs(only(compatible.eigenvalues) - only(ours.eigenvalues)))
        end
        @test errors[2] ≈ errors[1] / 2
        @test errors[3] ≈ errors[2] / 2
    end

    @testset "Noiseless Fourier simulation (Monte Carlo tolerances)" begin
        grid = collect(range(0, 1; length=101))
        weights = quadrature_weights(grid)
        functions = fourier_eigenfunctions((0, 1), 3)
        eigenvalues = [3.0, 1.0, 0.25]
        simulation = simulate_kl(Xoshiro(20261024), 5000; mean=t -> 2 + t,
            eigenfunctions=functions, eigenvalues, noise_sd=0,
            sampling=RegularSampling(grid))
        model = StatsAPI.fit(FPCA, simulation.data; ncomponents=3)
        # Six score-variance standard errors; exact RNG values are not assumed.
        @test all(abs.(eigvals(model) - eigenvalues) .<
            6sqrt(2 / (5000 - 1)) .* eigenvalues)
        phi = basis_matrix(eigenfunctions(model), grid)
        truephi = [f(t) for t in grid, f in functions]
        # The constant has positive population mean alignment; the sine has
        # negative alignment with 2+t, so the mean-alignment sign rule fixes both.
        overlaps = transpose(truephi) * (weights .* phi)
        @test overlaps[1, 1] > 0.98
        @test overlaps[2, 2] < -0.98
        # The cosine has zero population mean alignment. Its sign is sensitive
        # to finite-grid and sample perturbations, so only this overlap uses abs.
        @test abs(overlaps[3, 3]) > 0.98
        # Six times the n^(-1/2) rotation scale for these fixed eigenvalue gaps.
        @test norm(overlaps - Diagonal(diag(overlaps))) < 6 / sqrt(5000)
        # Entrywise rounding bound avoids accumulating the error over 505000 values.
        @test maximum(abs, reconstruct(model).values - simulation.data.values) <
            64eps(Float64) * maximum(abs, simulation.data.values)
        @test StatsAPI.predict(model, simulation.data) ≈ scores(model)
    end

    @testset "Presmoothing and scalar types" begin
        grid = collect(range(0, 1; length=9))
        fd = GridFunctionalData([i + (i - 2)t for i in 1:4, t in grid], grid)
        localmethod = LocalPolynomial(; bandwidth=0.75)
        localfit = StatsAPI.fit(FPCA, fd; presmooth=localmethod, ncomponents=1)
        _test_prediction_contract(localfit, fd)
        direct = StatsAPI.fit(FPCA, smooth_curves(fd, localmethod; grid); ncomponents=1)
        @test eigvals(localfit) ≈ eigvals(direct)
        @test scores(localfit) ≈ scores(direct)
        @test localfit.presmooth.bandwidth == localmethod.bandwidth
        @test localfit.presmooth.degree == localmethod.degree
        @test StatsAPI.predict(localfit, fd) ≈ scores(localfit)
        spline = PenalizedSpline(MonomialBasis((0, 1), 1); lambda=0.0)
        splinefit = StatsAPI.fit(FPCA, fd; presmooth=spline, ncomponents=1)
        @test eigvals(splinefit) ≈ eigvals(StatsAPI.fit(FPCA, fd; ncomponents=1))
        @test splinefit.presmooth.lambda == 0.0
        @test StatsAPI.predict(splinefit, fd) ≈ scores(splinefit)

        # Select a positive penalty on noisy training curves. Exact quadratic
        # new curves would select zero, so reselection changes their projection.
        traininggrid = collect(range(-1, 1; length=9))
        training = GridFunctionalData([i + (i - 2)t + (2 + 0.2i)t^2 +
            0.8 * (-1)^j for i in 1:4, (j, t) in enumerate(traininggrid)], traininggrid)
        quadratic = MonomialBasis((-1, 1), 2)
        tuned = PenalizedSpline(quadratic; lambda=GCV([0.0, 0.001, 0.01, 0.1, 1.0, 10.0]))
        tunedfit = StatsAPI.fit(FPCA, training; presmooth=tuned, ncomponents=1)
        @test StatsAPI.predict(tunedfit, training) ≈ scores(tunedfit)
        selected_lambda = tunedfit.settings.preprocessing.lambda
        @test selected_lambda == tunedfit.settings.preprocessing.selection.lambda
        newdata = GridFunctionalData(reshape(1 .+ traininggrid .+ 0.3traininggrid.^2, 1, :), traininggrid)
        independently_tuned = smooth_curves(newdata, tuned)
        @test independently_tuned.metadata.lambda != selected_lambda
        frozenmethod = PenalizedSpline(quadratic; lambda=selected_lambda)
        frozenvalues = evaluate(smooth_curves(newdata, frozenmethod), traininggrid)
        expected_projection = (frozenvalues .- tunedfit.mean.values) *
            (tunedfit.weights .* eigenfunctions(tunedfit).values)
        @test StatsAPI.predict(tunedfit, newdata) ≈ expected_projection
        independently_projected = (evaluate(independently_tuned, traininggrid) .-
            tunedfit.mean.values) * (tunedfit.weights .* eigenfunctions(tunedfit).values)
        @test norm(independently_projected - expected_projection) > 1e-5
        f32 = GridFunctionalData(Float32[1 2 3 4; 2 3 4 5; 3 4 5 6], Float32[0, 1, 2, 3])
        model32 = @inferred (data -> StatsAPI.fit(FPCA, data;
            method=:dense, ncomponents=1))(f32)
        @test eigvals(model32) isa Vector{Float32}
        @test scores(model32) isa Matrix{Float32}
        @test (@inferred StatsAPI.predict(model32, f32)) isa Matrix{Float32}
        @test (@inferred project_scores(model32, f32)) isa Matrix{Float32}
        @test (@inferred predict_scores(model32, f32)).scores isa Matrix{Float32}
        @test predict_scores(model32, f32).covariances === nothing
        @test (@inferred reconstruct(model32)) isa GridFunctionalData{Float32}
    end

    @testset "Validation and route requirements" begin
        fd = GridFunctionalData([i * t for i in 1:4, t in 0.0:0.25:1.0], collect(0.0:0.25:1.0))
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd[1:1])
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:invalid)
        for count in (0, -1, 1.5, 4, AIC(), BIC())
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; ncomponents=count)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; ncomponents=2, max_k=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; ncomponents=2, max_k=3) # Positive rank is one.
        for cap in (0, -1, 1.5, 4)
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; max_k=cap)
        end
        for threshold in (0, -0.1, 1.1, NaN, Inf)
            @test_throws ArgumentError FVE(threshold)
        end
        for tolerance in (-1.0, 1.0, NaN, Inf)
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; eigen_tol=tolerance)
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; sign_tol=tolerance)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; presmooth=:unsupported)
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(ones(3, 5), fd.grid))
        twopoint = GridFunctionalData([1.0 2.0; 2.0 3.0; 3.0 4.0], [0.0, 1.0])
        twopointfit = StatsAPI.fit(FPCA, twopoint)
        @test twopointfit.selection.max_k == 1 # Clamp the m-2 default to one.
        @test eigvals(twopointfit) ≈ [1.0]
        @test reconstruct(twopointfit).values ≈ twopoint.values
        twocurvefit = StatsAPI.fit(FPCA, fd[1:2])
        @test twocurvefit.selection.max_k == 1 # Clamp the n-2 default to one.
        @test ncomponents(twocurvefit) == 1
        @test StatsAPI.fit(FPCA, fd[1:3]).selection.max_k == 1
        # Explicit max_k can still use the sample covariance's n-1 rank cap.
        @test StatsAPI.fit(FPCA, fd[1:3]; max_k=2).selection.max_k == 2
        masked = GridFunctionalData([1.0 missing 3.0; 2.0 3.0 4.0], [0.0, 0.5, 1.0])
        irregular = to_irregular(fd)
        for data in (fd, masked, irregular)
            @test StatsAPI.fit(FPCA, data; method=:pace, mean_bandwidth=0.5,
                covariance_bandwidth=0.5, noise_variance=0.1).method == :pace
        end
        for data in (masked, irregular)
            @test StatsAPI.fit(FPCA, data; mean_bandwidth=0.5,
                covariance_bandwidth=0.5, noise_variance=0.1).method == :pace
            @test_throws ArgumentError StatsAPI.fit(FPCA, data; method=:dense)
        end
        model = StatsAPI.fit(FPCA, fd; ncomponents=1)
        masked_values = Matrix{Union{Missing,Float64}}(fd.values)
        masked_values[1, 2] = missing
        masked_prediction = GridFunctionalData(masked_values, fd.grid)
        @test_throws ArgumentError StatsAPI.predict(model, masked_prediction)
        @test_throws ArgumentError StatsAPI.predict(model, irregular)
    end
end

@testset "FPCA eigen workspace ownership and complete diagnostics" begin
    for T in (Float32, Float64)
        weights = T[.25, .5, 1]
        root = sqrt.(weights)
        surface = T[3 1 0; 1 3 0; 0 0 -1] ./ (root * root')
        storage = zeros(T, 5, 5)
        storage[1:3,1:3] = surface
        snapshot = copy(storage)
        normalization_weights = Float64[.2, .3, .5]
        full = FunctionalDataAnalysis._fpca_eigen(view(storage,1:3,1:3), weights,
            zeros(T,3); max_k=3, normalization_weights)
        capped = FunctionalDataAnalysis._fpca_eigen(view(storage,1:3,1:3), weights,
            zeros(T,3); max_k=1, normalization_weights)
        tolerance = T === Float32 ? 2e-6 : 1e-12
        @test full.eigenvalues ≈ [4,2] rtol=tolerance
        @test full.fve ≈ [2/3,1] rtol=tolerance
        @test full.dropped.number == 1
        @test full.dropped.total ≈ 1 rtol=tolerance
        @test full.dropped.share ≈ 1/7 rtol=tolerance
        @test capped.dropped == full.dropped
        @test capped.eigenvalues == full.eigenvalues[1:1]
        @test capped.fve == [1]
        @test all(column -> isapprox(norm(sqrt.(normalization_weights) .* column), 1;
            rtol=tolerance), eachcol(full.eigenfunctions))
        @test storage == snapshot
        full.eigenfunctions[1,1] += 1
        @test storage == snapshot
    end
end
