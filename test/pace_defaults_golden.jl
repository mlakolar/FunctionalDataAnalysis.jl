import StatsAPI
import KernelSmoothing
using DelimitedFiles

@testset "PACE end-to-end and automatic-default fdapace references" begin
    input = readdlm(joinpath(@__DIR__, "data", "m3_sparse_input.csv"), ','; comments=true)
    n = Int(maximum(input[:, 1]))
    fd = IrregularFunctionalData(
        [Float64.(input[input[:, 1] .== i, 2]) for i in 1:n],
        [Float64.(input[input[:, 1] .== i, 3]) for i in 1:n])
    sigma2 = only(readdlm(joinpath(@__DIR__, "data", "m3_sparse_noise.csv"), ','; comments=true))
    relative_error(actual, expected) = norm(actual - expected) / norm(expected)
    read_reference(file) = readdlm(joinpath(@__DIR__, "data", file), ',', Any; comments=true)
    reference(table, quantity) = Float64.(table[table[:, 1] .== quantity, 4])

    # fdapace uses ordinary matrix eigenvectors, rectangular eigenvalues and
    # separate trapezoidal column normalization. Reproduce that documented
    # quantity explicitly; the public fit retains its quadrature convention.
    function check_rectangular_reference(model, table, prefix)
        @test !model.settings.noise_floor_applied
        m = length(model.grid)
        k = Int(only(reference(table, prefix * "selected_k")))
        step = model.grid[2] - model.grid[1]
        system = FunctionalDataAnalysis._fpca_eigen(model.covariance, fill(step, m),
            vec(model.mean.values); maxK=model.selection.maxK,
            normalization_weights=model.weights)
        @test relative_error(system.eigenvalues[1:k],
            reference(table, prefix * "eigenvalue")) < 1e-8
        @test relative_error(system.eigenvalues,
            reference(table, prefix * "all_eigenvalue")) < 1e-8
        @test relative_error(system.fve, reference(table, prefix * "fve")) < 1e-8
        @test relative_error(system.eigenfunctions[:, 1:k],
            reshape(reference(table, prefix * "eigenfunction"), m, k)) < 1e-8
        projected = (system.eigenfunctions .* transpose(system.eigenvalues)) *
            transpose(system.eigenfunctions)
        @test relative_error(projected,
            reshape(reference(table, prefix * "fitted_covariance"), m, m)) < 1e-8
        ce = FunctionalDataAnalysis._pace_ce(fd, model.grid, vec(model.mean.values),
            system.eigenfunctions[:, 1:k], system.eigenvalues[1:k], projected, sigma2)
        @test relative_error(ce.scores,
            reshape(reference(table, prefix * "score"), n, k)) < 1e-8
        for i in 1:n
            rows = (table[:, 1] .== prefix * "score_covariance") .& (table[:, 2] .== i)
            expected = reshape(Float64.(table[rows, 4]), k, k)
            @test relative_error(ce.covariances[:, :, i], expected) < 1e-8
        end
        @test ncomponents(model) == k
    end

    @testset "Fixed-bandwidth smoothers, eigen step and CE" begin
        table = read_reference("m3_fdapace_end2end.csv")
        grid = reference(table, "grid")
        model = StatsAPI.fit(FPCA, fd; grid, mean_bandwidth=0.30,
            covariance_bandwidth=0.40, kernel=KernelSmoothing.EpanechnikovKernel(),
            weighting=:obs, binning=nothing, ncomponents=3, maxK=5,
            noise_variance=sigma2)
        @test grid ≈ sort(unique(reduce(vcat, first.(collect(fd)))))
        @test relative_error(vec(model.mean.values), reference(table, "mean")) < 1e-8
        @test relative_error(model.covariance,
            reshape(reference(table, "covariance"), length(grid), length(grid))) < 1e-8
        @test noise_variance(model) == only(reference(table, "noise_variance"))
        check_rectangular_reference(model, table, "")

        # Cross-method checks, not golden tolerances: native fdapace uses
        # rectangular eigenanalysis while public FPCA uses trapezoidal weights.
        @test relative_error(eigvals(model), reference(table, "eigenvalue")) < 0.005
        @test relative_error(eigenfunctions(model).values,
            reshape(reference(table, "eigenfunction"), length(grid), 3)) < 0.12
        @test relative_error(scores(model), reshape(reference(table, "score"), n, 3)) < 0.01
    end

    @testset "Automatic defaults and aligned interpolation" begin
        table = read_reference("m3_fdapace_defaults.csv")
        # Every tuning choice is automatic. Only sigma2 is supplied, to align
        # the distinct noise estimators documented in the fdapace audit.
        model = StatsAPI.fit(FPCA, fd; noise_variance=sigma2)
        m = length(model.grid)
        @test model.grid ≈ reference(table, "native_grid") rtol=1e-14
        @test m == 51
        @test model.selection.maxK == Int(only(reference(table, "default_maxK")))
        @test model.settings.mean_settings.bandwidth == reference(table, "default_bandwidth")[1]
        @test model.settings.covariance_settings.bandwidth == reference(table, "default_bandwidth")[2]
        @test model.settings.kernel isa KernelSmoothing.GaussianKernel
        @test relative_error(vec(model.mean.values), reference(table, "direct_mean")) < 1e-8
        @test relative_error(model.covariance,
            reshape(reference(table, "aligned_covariance"), m, m)) < 1e-8
        @test noise_variance(model) == only(reference(table, "native_noise_variance"))
        check_rectangular_reference(model, table, "aligned_")

        candidates = pace_bandwidth_candidates(fd)
        @test relative_error(candidates.mean, reference(table, "mean_candidate")) < 1e-8
        @test relative_error(candidates.covariance, reference(table, "covariance_candidate")) < 1e-8

        # Cross-method checks of the untouched native-default R FPCA call.
        # Native R smooths obsGrid then interpolates to workGrid; Julia smooths
        # workGrid and interpolates back to observation times. Eigenanalysis
        # also differs by the deliberate quadrature choice stated above.
        k = Int(only(reference(table, "native_selected_k")))
        @test ncomponents(model) == k
        @test relative_error(vec(model.mean.values), reference(table, "native_mean")) < 5e-5
        @test relative_error(model.covariance,
            reshape(reference(table, "native_covariance"), m, m)) < 3e-4
        @test relative_error(eigvals(model), reference(table, "native_eigenvalue")) < 0.001
        @test relative_error(eigenfunctions(model).values,
            reshape(reference(table, "native_eigenfunction"), m, k)) < 0.04
        @test relative_error(scores(model), reshape(reference(table, "native_score"), n, k)) < 0.01
    end
end
