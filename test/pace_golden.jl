import StatsAPI
using DelimitedFiles

@testset "PACE fdapace 0.6.0 golden values" begin
    input = readdlm(joinpath(@__DIR__, "data", "m3_sparse_input.csv"), ','; comments=true)
    mean_input = readdlm(joinpath(@__DIR__, "data", "m3_sparse_mean.csv"), ','; comments=true)
    surface = readdlm(joinpath(@__DIR__, "data", "m3_sparse_surface.csv"), ','; comments=true)
    sigma2 = only(readdlm(joinpath(@__DIR__, "data", "m3_sparse_noise.csv"), ','; comments=true))
    golden = readdlm(joinpath(@__DIR__, "data", "m3_fdapace_pace.csv"), ',', Any; comments=true)
    heldout = readdlm(joinpath(@__DIR__, "data", "m3_sparse_heldout.csv"), ','; comments=true)
    grid = Float64.(mean_input[:, 1])
    mu = Float64.(mean_input[:, 2])
    n = Int(maximum(input[:, 1]))
    times = [Float64.(input[input[:, 1] .== i, 2]) for i in 1:n]
    values = [Float64.(input[input[:, 1] .== i, 3]) for i in 1:n]
    labels = ["sparse-$i" for i in 1:n]
    fd = IrregularFunctionalData(times, values; ids=labels)
    supplied_mean = (; grid, values=mu)
    supplied_covariance = (; grid, surface)
    reference(quantity) = Float64.(golden[golden[:, 1] .== quantity, 4])
    relative_error(actual, expected) = norm(actual - expected) / norm(expected)

    model = StatsAPI.fit(FPCA, fd; grid, mean=supplied_mean,
        covariance=supplied_covariance, noise_variance=sigma2,
        ncomponents=3, max_k=5)
    @test model.method == :pace
    @test nobs(model) == 40
    @test ids(model) == labels
    @test noise_variance(model) == sigma2
    @test !model.settings.noise_floor_applied
    @test all(2 <= length(t) <= 8 for t in times)
    @test relative_error(vec(model.mean.values), reference("mean")) < 1e-8
    @test relative_error(model.covariance,
        reshape(reference("covariance"), length(grid), length(grid))) < 1e-8
    # Sine functions vanish at the endpoints. On this fixture the public
    # trapezoidal eigenstep agrees with fdapace's rectangular operator.
    @test relative_error(eigvals(model), reference("eigenvalue")) < 1e-8
    @test relative_error(model.all_eigenvalues, reference("all_eigenvalue")) < 1e-8
    @test relative_error(basis_matrix(eigenfunctions(model), grid),
        reshape(reference("eigenfunction"), length(grid), 3)) < 1e-8
    @test relative_error(fve(model), reference("fve")) < 1e-8
    @test relative_error(scores(model), reshape(reference("score"), n, 3)) < 1e-8
    for i in 1:n
        rows = (golden[:, 1] .== "score_covariance") .& (golden[:, 2] .== i)
        # fdapace SubsetFPCA leaves xiVar at max_k even when selecting K=3;
        # the reference contains its selected upper-left 3-by-3 block.
        expected = reshape(Float64.(golden[rows, 4]), 3, 3)
        @test relative_error(score_covariances(model)[:, :, i], expected) < 1e-8
    end
    @test relative_error(model.settings.full_projected_covariance,
        reshape(reference("fitted_covariance"), length(grid), length(grid))) < 1e-8
    # The score denominator includes components 4 and 5 even though K=3.
    selected_surface = basis_matrix(eigenfunctions(model), grid) *
        Diagonal(eigvals(model)) * transpose(basis_matrix(eigenfunctions(model), grid))
    @test norm(model.settings.full_projected_covariance - selected_surface) > 0.1
    training_prediction = StatsAPI.predict(model, fd)
    @test training_prediction.scores == scores(model)
    @test training_prediction.covariances == score_covariances(model)

    prediction_times = [Float64.(heldout[heldout[:, 1] .== i, 2]) for i in 1:3]
    prediction_values = [Float64.(heldout[heldout[:, 1] .== i, 3]) for i in 1:3]
    prediction = StatsAPI.predict(model, IrregularFunctionalData(prediction_times, prediction_values))
    @test any(t -> !(t in grid), reduce(vcat, prediction_times))
    @test relative_error(prediction.scores, reshape(reference("prediction_score"), 3, 3)) < 1e-8

    selected = Int.(reference("selected_k"))
    global_minimizers = Int.(reference("global_minimizer"))
    @test ncomponents(model) == selected[1]
    for (j, (selector, quantity)) in enumerate(((AIC(), "aic"), (BIC(), "bic")))
        chosen = StatsAPI.fit(FPCA, fd; grid, mean=supplied_mean,
            covariance=supplied_covariance, noise_variance=sigma2,
            ncomponents=selector, max_k=5)
        @test ncomponents(chosen) == selected[j + 1]
        @test chosen.selection.global_minimizer == global_minimizers[j]
        @test relative_error(chosen.selection.scores, reference(quantity)) < 1e-8
        # GetLogLik omits only this K-independent Gaussian constant.
        @test reference(quantity) ≈ reference("fdapace_" * quantity) .+
            only(reference("gaussian_constant")) rtol=1e-14
    end
    fve_model = StatsAPI.fit(FPCA, fd; grid, mean=supplied_mean,
        covariance=supplied_covariance, noise_variance=sigma2,
        ncomponents=FVE(0.99), max_k=5)
    @test ncomponents(fve_model) == selected[4]
end
