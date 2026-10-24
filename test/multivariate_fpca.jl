import StatsAPI
import KernelSmoothing

function m4_dense_fixture()
    grid = [0.0, 0.1, 0.35, 0.65, 0.9, 1.0]
    weights = quadrature_weights(grid)
    slope = (grid .- 0.5) / sqrt(dot(weights, (grid .- 0.5).^2))
    functions = hcat(ones(length(grid)), slope)
    firstscores = hcat([-2.0, -1, 0, 0, 1, 2], [1.0, -1, 1, -1, -1, 1])
    secondscores = firstscores .* [2.0 0.5]
    means = [3 .+ 0.2 .* slope, 5 .+ 0.1 .* slope]
    values = [firstscores * functions' .+ means[1]',
        secondscores * functions' .+ means[2]']
    labels = ["subject-$i" for i in 1:6]
    data = MultivariateFunctionalData([
        GridFunctionalData(y, grid; ids=labels) for y in values])
    (; data, grid, weights, functions, projected=[firstscores, secondscores], means, values, labels)
end

@testset "Multivariate FPCA" begin
    @testset "Marginal dense fits preserve univariate results" begin
        fixture = m4_dense_fixture()
        data, grid = fixture.data, fixture.grid
        model = @inferred (fd -> StatsAPI.fit(FPCA, fd;
            basis=:marginal, ncomponents=2, maxK=2))(data)
        @test model isa MultivariateFPCA
        @test model.basis == :marginal
        @test (@inferred ncomponents(model)) == [2, 2]
        @test (@inferred nobs(model)) == 6
        @test ids(model) == fixture.labels
        @test (@inferred scores(model)) isa Vector{Matrix{Float64}}
        @test (@inferred eigvals(model)) isa Vector{Vector{Float64}}
        @test (@inferred eigenfunctions(model)) isa Vector{FPCBasis{Float64}}
        @test (@inferred score_covariances(model)) == [nothing, nothing]
        for j in 1:2
            univariate = StatsAPI.fit(FPCA, component(data, j); ncomponents=2, maxK=2)
            node = component(model, j)
            @test node isa FPCA
            @test node.method == univariate.method == :dense
            @test node.grid == univariate.grid
            @test node.mean.values == univariate.mean.values
            @test node.covariance == univariate.covariance
            @test scores(model, j) == scores(univariate)
            @test scores(model)[j] == scores(univariate)
            @test eigvals(model)[j] == eigvals(univariate)
            @test eigenfunctions(model)[j] == eigenfunctions(univariate)
            @test node.selection == univariate.selection
            @test node.settings == univariate.settings
            @test scores(model, j) ≈ fixture.projected[j] atol=2e-14
        end

        array = @inferred score_array(model)
        @test size(array) == (6, 2, 2)
        for j in 1:2
            @test array[:, j, :] == scores(model, j)
            centered = to_array(data)[:, j, :] .- fixture.means[j]'
            @test array[:, j, :] ≈ centered * (fixture.weights .* fixture.functions)
        end
        array[1, 1, 1] = 1000
        @test scores(model, 1)[1, 1] != 1000
        prediction = @inferred StatsAPI.predict(model, data)
        @test length(prediction) == 2
        @test prediction == scores(model)
        restored = @inferred reconstruct(model)
        @test restored isa MultivariateFunctionalData{Float64}
        @test ids(restored) == fixture.labels
        for j in 1:2
            @test component(restored, j).values ≈ fixture.values[j] atol=3e-14
        end
        copied_scores = deepcopy(scores(model))
        @test ids(reconstruct(model, copied_scores)) === nothing
        newgrid = [0.05, 0.2, 0.6, 0.95]
        finer = reconstruct(model; grid=newgrid)
        separate = reconstruct(model; grid=[newgrid, grid])
        for j in 1:2
            @test component(finer, j).values == reconstruct(component(model, j); grid=newgrid).values
        end
        @test component(separate, 1).grid == newgrid
        @test component(separate, 2).grid == grid
        @test ids(separate) == fixture.labels

        default = StatsAPI.fit(FPCA, data)
        @test default.basis == :marginal
        @test ncomponents(default) == [ncomponents(StatsAPI.fit(FPCA, component(data, j)))
            for j in 1:2]
        presmoothers = [LocalPolynomial(; bandwidth=0.4),
            LocalPolynomial(; bandwidth=0.6)]
        presmoothed = StatsAPI.fit(FPCA, data; ncomponents=2, maxK=2,
            presmooth=presmoothers)
        for j in 1:2
            reference = StatsAPI.fit(FPCA, component(data, j);
                ncomponents=2, maxK=2, presmooth=presmoothers[j])
            @test scores(presmoothed, j) == scores(reference)
            @test eigvals(presmoothed)[j] == eigvals(reference)
        end

        # Prediction centers at the training mean, even for one new subject.
        shifted = MultivariateFunctionalData([
            GridFunctionalData(fixture.values[j][1:1, :] .+ 0.7, grid) for j in 1:2])
        newprediction = StatsAPI.predict(model, shifted)
        for j in 1:2
            @test newprediction[j] ≈ StatsAPI.predict(component(model, j), component(shifted, j))
        end

        unequal = StatsAPI.fit(FPCA, data; ncomponents=[1, 2], maxK=2)
        @test ncomponents(unequal) == [1, 2]
        error = try
            score_array(unequal)
        catch exception
            exception
        end
        @test error isa ArgumentError
        @test occursin("integer", sprint(showerror, error))
        @test occursin("ncomponents", sprint(showerror, error))
        @test_throws BoundsError scores(model, 0)
        @test_throws BoundsError component(model, 3)
        @test_throws DimensionMismatch StatsAPI.predict(model,
            MultivariateFunctionalData([component(data, 1)]))
        @test_throws DimensionMismatch reconstruct(model, [zeros(6, 2)])
        @test_throws DimensionMismatch reconstruct(model, [zeros(6, 2), zeros(5, 2)])
        @test_throws DimensionMismatch reconstruct(model, [zeros(6, 1), zeros(6, 2)])
        @test_throws ArgumentError reconstruct(model, [fill(NaN, 6, 2), zeros(6, 2)])
        @test_throws DimensionMismatch reconstruct(model; grid=[grid])
        @test_throws ArgumentError reconstruct(model; grid=[-0.1, 0.5])

        # Neither training data nor a returned ID vector aliases the fit.
        returned_ids = ids(model)
        returned_ids[1] = "changed"
        component(data, 1).values .= 1000
        component(data, 1).grid[2] = 0.2
        component(data, 1).ids[1] = "changed"
        @test ids(model) == fixture.labels
        @test component(model, 1).grid == grid
        @test scores(model, 1) ≈ fixture.projected[1] atol=2e-14
        @test component(reconstruct(model), 1).values ≈ fixture.values[1] atol=3e-14
        @test occursin("marginal", sprint(show, model))
        @test occursin("settings", sprint(show, model))
    end

    @testset "Mixed dense and PACE nodes retain their routes and grids" begin
        fixture = m4_dense_fixture()
        grid = [0.0, 0.2, 0.5, 0.8, 1.0]
        weights = quadrature_weights(grid)
        slope_norm = sqrt(dot(weights, (grid .- 0.5).^2))
        phi(t) = [1.0, (t - 0.5) / slope_norm]
        lambda = [2.0, 0.7]
        mu(t) = 1 + t
        times = [[0.0, 0.2, 0.65, 1.0], [0.1, 0.4, 0.9],
            [0.05, 0.5, 0.7, 0.95], [0.25, 0.75],
            [0.0, 0.35, 0.8], [0.15, 0.55, 1.0]]
        values = [mu.(t) .+ [dot(phi(x), [i / 3, (-1)^i / 2]) for x in t]
            for (i, t) in enumerate(times)]
        sparse = IrregularFunctionalData(times, values; domain=(0.0, 1.0),
            ids=fixture.labels)
        data = MultivariateFunctionalData([component(fixture.data, 1), sparse])
        functions = reduce(vcat, [phi(t)' for t in grid])
        supplied_mean = (; grid=copy(grid), values=mu.(grid))
        supplied_covariance = (; grid=copy(grid),
            surface=Matrix(Symmetric(functions * Diagonal(lambda) * functions')))
        options = (; method=[:auto, :pace], grid=[nothing, grid],
            ncomponents=[1, 2], maxK=[2, 2], mean=[nothing, supplied_mean],
            covariance=[nothing, supplied_covariance], noise_variance=[nothing, 0.13],
            eigen_tol=[1e-10, 1e-9], sign_tol=[1e-12, 2e-12])
        model = StatsAPI.fit(FPCA, data; options...)
        @test ncomponents(model) == [1, 2]
        @test component(model, 1).method == :dense
        @test component(model, 2).method == :pace
        @test component(model, 1).grid == fixture.grid
        @test component(model, 2).grid == grid
        @test ids(model) == fixture.labels
        references = [StatsAPI.fit(FPCA, component(data, 1); ncomponents=1,
                maxK=2, eigen_tol=1e-10, sign_tol=1e-12),
            StatsAPI.fit(FPCA, component(data, 2); method=:pace, grid,
                ncomponents=2, maxK=2, mean=supplied_mean,
                covariance=supplied_covariance, noise_variance=0.13,
                eigen_tol=1e-9, sign_tol=2e-12)]
        for j in 1:2
            @test scores(model, j) == scores(references[j])
            @test eigenfunctions(model)[j] == eigenfunctions(references[j])
            @test eigvals(model)[j] == eigvals(references[j])
            @test component(model, j).settings == references[j].settings
        end
        @test score_covariances(model)[1] === nothing
        @test score_covariances(model)[2] == score_covariances(references[2])
        prediction = @inferred StatsAPI.predict(model, data)
        @test prediction[1] == scores(model, 1)
        @test prediction[2].scores == scores(model, 2)
        @test prediction[2].covariances == score_covariances(model)[2]
        restored = @inferred reconstruct(model)
        @test ids(restored) == fixture.labels
        for j in 1:2
            expected = reconstruct(references[j])
            @test component(restored, j).values == expected.values
            @test component(restored, j).grid == expected.grid
        end
        @test_throws ArgumentError score_array(model)

        # Shared values and explicit per-node values resolve identically.
        sparsepair = MultivariateFunctionalData([sparse, sparse])
        shared = StatsAPI.fit(FPCA, sparsepair; grid, ncomponents=2, maxK=2,
            mean=supplied_mean, covariance=supplied_covariance, noise_variance=0.13)
        separate = StatsAPI.fit(FPCA, sparsepair; grid=[grid, grid],
            ncomponents=[2, 2], maxK=[2, 2], mean=[supplied_mean, supplied_mean],
            covariance=[supplied_covariance, supplied_covariance], noise_variance=[0.13, 0.13])
        @test scores(shared) == scores(separate)
        @test eigvals(shared) == eigvals(separate)
        @test score_covariances(shared) == score_covariances(separate)
        mean_bandwidths = [0.45, 0.55]
        covariance_bandwidths = [0.5, 0.6]
        diagonal_bandwidths = [0.4, 0.5]
        kernels = [KernelSmoothing.GaussianKernel(), KernelSmoothing.GaussianKernel()]
        weightings = [:obs, :subj]
        binnings = [nothing, 5]
        smoothed = StatsAPI.fit(FPCA, sparsepair; grid, ncomponents=1, maxK=2,
            mean_bandwidth=mean_bandwidths, covariance_bandwidth=covariance_bandwidths,
            diagonal_bandwidth=diagonal_bandwidths, kernel=kernels,
            weighting=weightings, binning=binnings, noise_variance=0.13)
        for j in 1:2
            reference = StatsAPI.fit(FPCA, sparse;
                grid, ncomponents=1, maxK=2, mean_bandwidth=mean_bandwidths[j],
                covariance_bandwidth=covariance_bandwidths[j],
                diagonal_bandwidth=diagonal_bandwidths[j], kernel=kernels[j],
                weighting=weightings[j], binning=binnings[j], noise_variance=0.13)
            @test scores(smoothed, j) == scores(reference)
            @test eigvals(smoothed)[j] == eigvals(reference)
            @test score_covariances(smoothed)[j] == score_covariances(reference)
        end
        supplied_mean.values .= 100
        supplied_covariance.surface .= 100
        grid .= -100
        @test StatsAPI.predict(model, data)[2].scores == scores(model, 2)
    end

    @testset "Pooled covariance, shared basis, and node-specific variances" begin
        fixture = m4_dense_fixture()
        data, grid = fixture.data, fixture.grid
        model = @inferred (fd -> StatsAPI.fit(FPCA, fd;
            basis=:pooled, ncomponents=2, maxK=2))(data)
        @test model isa MultivariateFPCA
        @test model.basis == :pooled
        @test (@inferred ncomponents(model)) == [2, 2]
        @test (@inferred nobs(model)) == 6
        @test ids(model) == fixture.labels
        @test (@inferred score_covariances(model)) == [nothing, nothing]
        functions = @inferred eigenfunctions(model)
        @test functions[1] == functions[2]
        @test basismatrix(functions[1], grid) ≈ fixture.functions atol=2e-14
        @test functions[1].weights == fixture.weights
        @test functions[1].values' * (fixture.weights .* functions[1].values) ≈ I atol=2e-14
        @test eigvals(model)[1] ≈ [2.0, 1.2]
        @test eigvals(model)[2] ≈ [8.0, 0.3]
        @test (eigvals(model)[1] + eigvals(model)[2]) / 2 ≈ [5.0, 0.75]
        for j in 1:2
            @test scores(model, j) ≈ fixture.projected[j] atol=3e-14
            @test eigvals(model)[j] ≈ vec(sum(abs2, scores(model, j); dims=1)) / 5
        end
        array = @inferred score_array(model)
        @test size(array) == (6, 2, 2)
        for j in 1:2
            @test array[:, j, :] == scores(model, j)
            @test array[:, j, :] ≈
                (to_array(data)[:, j, :] .- fixture.means[j]') *
                (fixture.weights .* fixture.functions) atol=3e-14
        end
        @test (@inferred StatsAPI.predict(model, data)) == scores(model)
        restored = @inferred reconstruct(model)
        @test to_array(restored) ≈ to_array(data) atol=3e-14
        @test ids(restored) == fixture.labels
        @test ids(reconstruct(model, deepcopy(scores(model)))) === nothing
        target = [0.1, 0.3, 0.6, 0.9]
        reconstructed = reconstruct(model; grid=[target, grid])
        @test component(reconstructed, 1).grid == target
        @test component(reconstructed, 2).grid == grid
        @test component(reconstructed, 1).values ≈
            evaluate(component(data, 1), target, PiecewiseLinear()) atol=3e-14
        shifted = MultivariateFunctionalData([
            GridFunctionalData(fixture.values[j][1:1, :] .+ 0.7, grid) for j in 1:2])
        predicted = StatsAPI.predict(model, shifted)
        for j in 1:2
            @test predicted[j] ≈ fixture.projected[j][1:1, :] .+ [0.7 0.0] atol=3e-14
        end
        default = StatsAPI.fit(FPCA, data; basis=:pooled)
        @test ncomponents(default) == [2, 2]
        selected = StatsAPI.fit(FPCA, data; basis=:pooled, ncomponents=FVE(0.85), maxK=2)
        @test ncomponents(selected) == [1, 1]
        capped = StatsAPI.fit(FPCA, data; basis=:pooled, maxK=1)
        # Pooled rank is nodes*(n-1), so maxK may exceed n-1 when the grid allows.
        tgrid = collect(range(0, 1; length=12))
        few = MultivariateFunctionalData([GridFunctionalData(
            permutedims(reduce(hcat, [sin.(k * (j + 1) .* tgrid) for k in 1:3])), tgrid) for j in 1:3])
        wide = StatsAPI.fit(FPCA, few; basis=:pooled)
        @test wide.settings.maxK == min(12 - 2, 3 * 2 - 1)
        @test StatsAPI.fit(FPCA, few; basis=:pooled, maxK=6).settings.maxK == 6
        @test_throws ArgumentError StatsAPI.fit(FPCA, few; basis=:pooled, maxK=7)
        @test ncomponents(capped) == [1, 1]
        error = try
            component(model, 1)
        catch exception
            exception
        end
        @test error isa ArgumentError
        @test occursin("pooled", sprint(showerror, error))
        @test occursin("pooled", sprint(show, model))
        @test occursin("settings", sprint(show, model))

        # The covariance is averaged after centering each node independently.
        # Large differences in means must not create another pooled component.
        mean_shifted = MultivariateFunctionalData([component(data, 1),
            GridFunctionalData(fixture.values[2] .+ 1000, grid; ids=fixture.labels)])
        shifted_model = StatsAPI.fit(FPCA, mean_shifted; basis=:pooled, ncomponents=2, maxK=2)
        @test eigvals(shifted_model) ≈ eigvals(model) rtol=1e-12
        @test abs.(scores(shifted_model, 2)) ≈ abs.(scores(model, 2)) atol=3e-12

        # A node's own variance order need not be the pooled order. Pooling
        # must average the covariance surfaces before finding a shared basis.
        reversed_scores = fixture.projected[1] .* [0.25 2.0]
        reversed_data = MultivariateFunctionalData([component(data, 1),
            GridFunctionalData(reversed_scores * fixture.functions' .+
                fixture.means[2]', grid; ids=fixture.labels)])
        reversed_model = StatsAPI.fit(FPCA, reversed_data;
            basis=:pooled, ncomponents=2, maxK=2)
        @test basismatrix(eigenfunctions(reversed_model)[1], grid) ≈
            fixture.functions[:, [2, 1]] atol=3e-14
        @test eigvals(reversed_model)[1] ≈ [1.2, 2.0]
        @test eigvals(reversed_model)[2] ≈ [4.8, 0.125]
        @test scores(reversed_model, 1) ≈ fixture.projected[1][:, [2, 1]] atol=3e-14
        @test scores(reversed_model, 2) ≈ reversed_scores[:, [2, 1]] atol=3e-14

        component(data, 1).values .= 1000
        component(data, 1).grid[2] = 0.2
        component(data, 1).ids[1] = "changed"
        @test ids(model) == fixture.labels
        @test functions[1].grid == grid
        @test component(reconstruct(model), 1).values ≈ fixture.values[1] atol=3e-14
    end

    @testset "Shared-basis process (Monte Carlo tolerance)" begin
        rng = Xoshiro(20261024)
        n = 3000
        grid = collect(range(0, 1; length=21))
        weights = quadrature_weights(grid)
        functions = hcat(ones(length(grid)),
            (grid .- 0.5) / sqrt(dot(weights, (grid .- 0.5).^2)))
        variances = [[4.0, 1.0], [2.0, 0.5]]
        nodes = [GridFunctionalData(
            (randn(rng, n, 2) .* sqrt.(variances[j])') * functions' .+
            (j .+ grid)', grid) for j in 1:2]
        model = StatsAPI.fit(FPCA, MultivariateFunctionalData(nodes);
            basis=:pooled, ncomponents=2, maxK=2)
        recovered = basismatrix(eigenfunctions(model)[1], grid)
        # Six score-variance standard errors and a one-percent alignment
        # tolerance; neither check depends on exact Julia RNG output.
        expected = (variances[1] + variances[2]) / 2
        observed = (eigvals(model)[1] + eigvals(model)[2]) / 2
        @test all(abs.(observed - expected) .< 6sqrt(2 / (n - 1)) .* expected)
        @test all(abs.(diag(functions' * (weights .* recovered))) .> 0.99)
    end

    @testset "Float32 element types" begin
        fixture = m4_dense_fixture()
        data = MultivariateFunctionalData([GridFunctionalData(Float32.(values),
            Float32.(fixture.grid)) for values in fixture.values])
        for basis in (:marginal, :pooled)
            model = StatsAPI.fit(FPCA, data; basis, ncomponents=2, maxK=2)
            @test scores(model) isa Vector{Matrix{Float32}}
            @test eigvals(model) isa Vector{Vector{Float32}}
            @test eigenfunctions(model) isa Vector{FPCBasis{Float32}}
            @test score_array(model) isa Array{Float32,3}
            @test reconstruct(model) isa MultivariateFunctionalData{Float32}

            promoted_scores = [Float64.(s) for s in scores(model)]
            promoted = reconstruct(model, promoted_scores)
            @test promoted isa MultivariateFunctionalData{Float64}
            @test ids(promoted) === nothing
            perturbed_scores = deepcopy(promoted_scores)
            increment = 1e-8 # Smaller than Float32 resolution at these means.
            perturbed_scores[1][1, 1] += increment
            perturbed = reconstruct(model, perturbed_scores)
            difference = component(perturbed, 1).values[1, :] -
                component(promoted, 1).values[1, :]
            expected = increment .* Float64.(eigenfunctions(model)[1].values[:, 1])
            @test difference ≈ expected atol=3e-15
            @test component(perturbed, 1).values[2:end, :] ==
                component(promoted, 1).values[2:end, :]

            precise_grid = [0.1234567890123, 0.3456789012345, 0.7654321098765]
            on_precise_grid = reconstruct(model; grid=precise_grid)
            @test on_precise_grid isa MultivariateFunctionalData{Float64}
            for j in 1:2
                @test component(on_precise_grid, j).grid == precise_grid
            end
            @test component(on_precise_grid, 1).grid != Float64.(Float32.(precise_grid))
            on_separate_grids = reconstruct(model; grid=[Float32.(fixture.grid), precise_grid])
            @test on_separate_grids isa MultivariateFunctionalData{Float64}
            @test component(on_separate_grids, 2).grid == precise_grid
        end

        # A Float32 data object can yield a higher-precision PACE fit when
        # supplied model inputs use Float64. Keep each original marginal fit
        # and promote only the multivariate summary arrays.
        times = [Float32[0, 0.25, 1], Float32[0.1, 0.5, 0.9],
            Float32[0, 0.4, 0.8], Float32[0.2, 0.6, 1],
            Float32[0.05, 0.55, 0.95], Float32[0, 0.7, 1]]
        values = [1f0 .+ t .+ Float32(i) / 3 for (i, t) in enumerate(times)]
        sparse = IrregularFunctionalData(times, values; domain=(0f0, 1f0))
        mixed = MultivariateFunctionalData([component(data, 1), sparse])
        pace_grid = Float32[0, 0.25, 0.5, 0.75, 1]
        mean64 = (; grid=Float64.(pace_grid), values=1 .+ Float64.(pace_grid))
        covariance64 = (; grid=Float64.(pace_grid), surface=ones(5, 5))
        sigma2 = 0.123456789012345
        model = StatsAPI.fit(FPCA, mixed; ncomponents=1, maxK=[2, 1],
            grid=[nothing, pace_grid], mean=[nothing, mean64],
            covariance=[nothing, covariance64], noise_variance=[nothing, sigma2])
        references = [StatsAPI.fit(FPCA, component(mixed, 1); ncomponents=1, maxK=2),
            StatsAPI.fit(FPCA, component(mixed, 2); grid=pace_grid,
                ncomponents=1, maxK=1, mean=mean64,
                covariance=covariance64, noise_variance=sigma2)]
        @test model isa MultivariateFPCA{Float64}
        @test scores(component(model, 1)) isa Matrix{Float32}
        @test scores(component(model, 2)) isa Matrix{Float64}
        @test noise_variance(component(model, 2)) === sigma2
        for j in 1:2
            reference = references[j]
            @test scores(component(model, j)) == scores(reference)
            @test scores(model, j) == Float64.(scores(reference))
            @test eigvals(model)[j] == Float64.(eigvals(reference))
            @test eigenfunctions(model)[j].values == Float64.(eigenfunctions(reference).values)
            @test eigenfunctions(model)[j].weights == Float64.(eigenfunctions(reference).weights)
        end
        @test score_covariances(model)[1] === nothing
        @test score_covariances(model)[2] == score_covariances(references[2])
        prediction = StatsAPI.predict(model, mixed)
        @test prediction[1] == Float64.(scores(references[1]))
        @test prediction[2].scores == scores(references[2])
        @test prediction[2].covariances == score_covariances(references[2])
        @test reconstruct(model) isa MultivariateFunctionalData{Float64}
    end

    @testset "Validation of multivariate settings and pooled inputs" begin
        fixture = m4_dense_fixture()
        data, grid = fixture.data, fixture.grid
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:invalid)
        for keyword in (:method, :ncomponents, :maxK, :presmooth, :mean,
                :covariance, :noise_variance, :mean_bandwidth,
                :covariance_bandwidth, :diagonal_bandwidth, :kernel,
                :weighting, :binning, :eigen_tol, :sign_tol)
            @test_throws DimensionMismatch StatsAPI.fit(FPCA, data;
                NamedTuple{(keyword,)}(([nothing],))...)
        end
        @test_throws DimensionMismatch StatsAPI.fit(FPCA, data; grid=[grid])
        @test_throws DimensionMismatch StatsAPI.fit(FPCA, data;
            ncomponents=[1, 2, 1])
        masked = Matrix{Union{Missing,Float64}}(fixture.values[1])
        masked[1, 2] = missing
        masked_data = MultivariateFunctionalData([
            GridFunctionalData(masked, grid), component(data, 2)])
        irregular = MultivariateFunctionalData([
            to_irregular(component(data, 1)), component(data, 2)])
        different_grid = MultivariateFunctionalData([component(data, 1),
            GridFunctionalData(fixture.values[2], grid .+ 0.1)])
        @test_throws ArgumentError StatsAPI.fit(FPCA, masked_data; basis=:pooled)
        for unsupported in (irregular, different_grid)
            @test_throws ArgumentError StatsAPI.fit(FPCA, unsupported; basis=:pooled)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, data[1:1]; basis=:pooled)
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled, ncomponents=0)
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled, maxK=0)
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled, eigen_tol=NaN)
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled, sign_tol=-1)
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled,
            presmooth=LocalPolynomial(; bandwidth=0.4))
        @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled, grid)
        pace_error = try
            StatsAPI.fit(FPCA, data; basis=:pooled, method=:pace)
        catch exception
            exception
        end
        @test pace_error isa ArgumentError
        @test occursin("pooled PACE", sprint(showerror, pace_error))
        for (name, value) in pairs((; ncomponents=[1, 1], maxK=[2, 2],
                eigen_tol=[1e-10, 1e-10], sign_tol=[1e-12, 1e-12]))
            @test_throws ArgumentError StatsAPI.fit(FPCA, data; basis=:pooled,
                NamedTuple{(name,)}((value,))...)
        end
        basis_node = BasisFunctionalData(ones(6, 2), MonomialBasis((0, 1), 1);
            ids=fixture.labels)
        basis_data = MultivariateFunctionalData([component(data, 1), basis_node])
        basis_error = try
            StatsAPI.fit(FPCA, basis_data; ncomponents=1, maxK=2)
        catch exception
            exception
        end
        @test basis_error isa ArgumentError
        @test occursin("basis data", sprint(showerror, basis_error))
        @test occursin("convert", sprint(showerror, basis_error))
        model = StatsAPI.fit(FPCA, data; basis=:pooled, ncomponents=2, maxK=2)
        @test_throws DimensionMismatch StatsAPI.predict(model,
            MultivariateFunctionalData([component(data, 1)]))
        @test_throws ArgumentError StatsAPI.predict(model, different_grid)
        @test_throws ArgumentError StatsAPI.predict(model, masked_data)
        @test_throws ArgumentError StatsAPI.predict(model, irregular)
        @test_throws DimensionMismatch reconstruct(model, [zeros(6, 2)])
        @test_throws DimensionMismatch reconstruct(model, [zeros(6, 2), zeros(5, 2)])
        @test_throws DimensionMismatch reconstruct(model; grid=[grid])
    end
end
