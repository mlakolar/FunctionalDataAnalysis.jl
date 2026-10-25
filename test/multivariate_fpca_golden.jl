import StatsAPI
import KernelSmoothing

@testset "Multivariate FPCA FGM reference values" begin
    reference(table, basis, quantity, node=0) = golden_values6(table,basis,quantity,node)
    observed_errors = Float64[]
    relative_error(actual, expected) = golden_relative_error(observed_errors,actual,expected)
    aligned_signs(actual, expected) = sign.(vec(sum(actual .* expected; dims=1)))

    @testset "Dense marginal and pooled" begin
        golden = golden_read("m4_fgm_dense.csv")
        grid = reference(golden, "input", "grid")
        p = Int(maximum(golden[golden[:, 2] .== "value", 3]))
        n = Int(maximum(golden[golden[:, 2] .== "value", 4]))
        labels = ["fgm-dense-$i" for i in 1:n]
        nodes = [GridFunctionalData(reshape(reference(golden, "input", "value", j),
            n, length(grid)), grid; ids=labels) for j in 1:p]
        data = MultivariateFunctionalData(nodes)

        for basis in (:marginal, :pooled)
            fitted = StatsAPI.fit(FPCA, data; basis, ncomponents=2)
            @test ids(fitted) == labels
            @test ncomponents(fitted) == fill(2, p)
            array = score_array(fitted)
            @test size(array) == (n, p, 2)
            for j in 1:p
                expected_functions = reshape(reference(golden, string(basis),
                    "eigenfunction", j), length(grid), 2)
                actual_functions = basis_matrix(eigenfunctions(fitted)[j], grid)
                signs = aligned_signs(actual_functions, expected_functions)
                @test all(abs.(signs) .== 1)
                @test relative_error(actual_functions .* transpose(signs),
                    expected_functions) < 1e-8
                @test relative_error(eigvals(fitted)[j],
                    reference(golden, string(basis), "eigenvalue", j) .* n/(n-1)) < 1e-8
                expected_scores = reshape(reference(golden, string(basis), "score", j), n, 2)
                @test relative_error(scores(fitted, j) .* transpose(signs),
                    expected_scores) < 1e-8
                # The middle array index follows the same node order as to_array.
                @test array[:, j, :] == scores(fitted, j)
                @test to_array(data)[:, j, :] == nodes[j].values
            end
        end
        if get(ENV, "FDA_M4_REFERENCE_REPORT", "false") == "true"
            println("FGM dense maximum relative error: ", maximum(observed_errors))
        end
    end

    @testset "Sparse marginal conditional scores" begin
        empty!(observed_errors)
        golden = golden_read("m4_fgm_sparse.csv")
        grid = reference(golden, "input", "grid")
        input = golden[golden[:, 1] .== "input", :]
        n = Int(maximum(input[input[:, 2] .== "value", 4]))
        p = Int(maximum(input[input[:, 2] .== "value", 3]))
        labels = ["fgm-sparse-$i" for i in 1:n]
        nodes = [golden_sparse6(input,"input",j;subjects=1:n,ids=labels) for j in 1:p]
        noise = [only(reference(golden, "input", "noise_variance", j)) for j in 1:p]
        # Match FGM's full positive-rank denominator and relative cutoff.
        fitted = StatsAPI.fit(FPCA, MultivariateFunctionalData(nodes);
            basis=:marginal, grid, ncomponents=2, max_k=length(grid),
            mean_bandwidth=0.3, covariance_bandwidth=0.3, diagonal_bandwidth=0.3,
            noise_variance=noise, kernel=KernelSmoothing.GaussianKernel(),
            weighting=:subj, binning=nothing, eigen_tol=100eps(Float64))
        @test ids(fitted) == labels
        for j in 1:p
            univariate = component(fitted, j)
            @test univariate.method == :pace
            @test noise_variance(univariate) == noise[j]
            @test relative_error(vec(univariate.mean.values),
                reference(golden, "marginal", "mean", j)) < 1e-8
            expected_functions = reshape(reference(golden, "marginal",
                "eigenfunction", j), length(grid), 2)
            actual_functions = basis_matrix(eigenfunctions(fitted)[j], grid)
            signs = aligned_signs(actual_functions, expected_functions)
            @test all(abs.(signs) .== 1)
            @test relative_error(actual_functions .* transpose(signs), expected_functions) < 1e-8
            @test relative_error(eigvals(fitted)[j],
                reference(golden, "marginal", "eigenvalue", j)) < 1e-8
            expected_scores = reshape(reference(golden, "marginal", "score", j), n, 2)
            @test relative_error(scores(fitted, j) .* transpose(signs), expected_scores) < 1e-8
            @test relative_error(univariate.settings.full_projected_covariance,
                reshape(reference(golden, "marginal", "projected_covariance", j),
                    length(grid), length(grid))) < 1e-8
        end
        if get(ENV, "FDA_M4_REFERENCE_REPORT", "false") == "true"
            println("FGM sparse maximum relative error: ", maximum(observed_errors))
        end
    end
end
