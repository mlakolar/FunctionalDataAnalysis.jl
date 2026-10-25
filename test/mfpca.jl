using Test, LinearAlgebra, DelimitedFiles
using FunctionalDataAnalysis
import StatsAPI

function mfpca_fixture()
    table = readdlm(joinpath(@__DIR__, "data", "m4_mfpca_input.csv"), ','; comments=true)
    function matrix(quantity, node)
        rows = table[(table[:, 1] .== quantity) .& (table[:, 2] .== node), :]
        reshape(Float64.(rows[:, 5]), Int(maximum(rows[:, 3])), Int(maximum(rows[:, 4])))
    end
    labels = ["mfpca-$i" for i in 1:24]
    data = MultivariateFunctionalData([GridFunctionalData(matrix("value", j),
        vec(matrix("grid", j)); ids=labels) for j in 1:3])
    (; data, matrix, labels)
end

@testset "Happ--Greven multivariate FPCA" begin
    @testset "Default marginal expansion and settings" begin
        data = mfpca_fixture().data
        fitted = @inferred StatsAPI.fit(MFPCA, data; ncomponents=3)
        marginal = StatsAPI.fit(FPCA, data; basis=:marginal)
        @test fitted.univariate isa MultivariateFPCA{Float64}
        @test all(f -> f isa FPCA{Float64}, fitted.univariate.fits)
        @test fitted.weights == ones(3)
        @test fitted.settings.univariate == (;)
        @test fitted.settings.eigen_tol == 1e-10
        @test fitted.settings.covariance_denominator == nobs(data)-1
        @test fitted.settings.score_centering == :covariance_only
        @test fitted.settings.normalization == :r_mfpca_1_3_11
        @test fitted.settings.eigenanalysis == :exact
        @test !fitted.settings.bootstrap && !fitted.settings.fit
        @test fitted.settings.univariate_counts == ncomponents(marginal)
        for j in 1:3
            @test scores(component(fitted, j)) == scores(component(marginal, j))
            @test component(fitted, j).selection.rule == :FVE
            @test component(fitted, j).selection.threshold == 0.99
        end
    end

    @testset "R MFPCA 1.3-11: given expansions and cross-method uFPCA" begin
        fixture = mfpca_fixture()
        golden = readdlm(joinpath(@__DIR__, "data", "m4_mfpca.csv"), ','; comments=true)
        function reference(mode, quantity, node=0)
            a = golden[(golden[:, 1] .== mode) .& (golden[:, 2] .== quantity) .&
                (golden[:, 3] .== node), :]
            reshape(Float64.(a[:, 6]), Int(maximum(a[:, 4])), Int(maximum(a[:, 5])))
        end
        relative(a, b) = norm(a-b)/norm(b)
        strict_errors, cross_errors = Float64[], Float64[]
        for mode in ("given_unit", "given_weighted", "cross_method")
            weights = mode == "given_unit" ? ones(3) : [0.7, 1.5, 2.2]
            fitted = @inferred StatsAPI.fit(MFPCA, fixture.data; ncomponents=4,
                univariate=(; ncomponents=2, maxK=2), weights)
            # Strict R values compare signed coefficients as well as functions
            # and scores. Cross-method signs align the whole multivariate function.
            signs = ones(4)
            if mode == "cross_method"
                alignment = sum(weights[j] .* vec(sum(
                    fitted.eigenfunctions[j].weights .* fitted.eigenfunctions[j].values .*
                    reference(mode, "function", j); dims=1)) for j in 1:3)
                signs = sign.(alignment)
                @test all(abs.(signs) .== 1)
            end
            errors = mode == "cross_method" ? cross_errors : strict_errors
            # This cross-method fixture has linear means and constant/linear
            # residuals in both smoothers' unpenalized spaces, so it is exact
            # up to numerical error and uses the same strict tolerance.
            tolerance = 1e-8
            push!(errors, relative(eigvals(fitted), vec(reference(mode, "eigenvalue"))))
            push!(errors, relative(scores(fitted) .* signs', reference(mode, "score")))
            if mode != "cross_method"
                push!(errors, relative(fitted.coefficients, reference(mode, "coefficient")))
                for j in 1:3
                    @test relative(scores(component(fitted, j)), fixture.matrix("score", j)) < 1e-8
                    @test relative(eigenfunctions(component(fitted, j)).values,
                        fixture.matrix("function", j)) < 1e-8
                end
            end
            recovered = reconstruct(fitted)
            for j in 1:3
                push!(errors, relative(eigenfunctions(fitted)[j].values .* signs',
                    reference(mode, "function", j)))
                push!(errors, relative(component(recovered, j).values, reference(mode, "fit", j)))
            end
            @test maximum(errors) < tolerance
        end
        println("MFPCA given-expansion maximum relative error: ", maximum(strict_errors))
        println("MFPCA cross-method uFPCA maximum relative error: ", maximum(cross_errors))
    end

    @testset "Analytic reductions, weighted geometry, inference and IDs" begin
        n = 16
        latent = [sqrt(2*(n-1)/n)*cos(2pi*k*(i-0.5)/n) for i in 1:n, k in 1:4]
        grids = [[0.0, 0.1, 0.3, 0.6, 1.0], [-2.0, -1.9, -1.3, -0.5, 0.0]]
        variances = [9.0, 4.0, 2.0, 0.5]
        labels = ["analytic-$i" for i in 1:n]
        nodes = [begin
            w = quadrature_weights(g)
            linear = g .- dot(w, g)/sum(w)
            phi = hcat(ones(length(g))/sqrt(sum(w)), linear/sqrt(dot(w, linear.^2)))
            GridFunctionalData(3 .+ g' .+ (latent[:, 2j-1:2j] .*
                sqrt.(variances[2j-1:2j])')*phi', g; ids=labels)
        end for (j,g) in enumerate(grids)]
        data = MultivariateFunctionalData(nodes)
        one = StatsAPI.fit(MFPCA, MultivariateFunctionalData(nodes[1:1]);
            ncomponents=2, univariate=(; ncomponents=2, maxK=2))
        uni = StatsAPI.fit(FPCA, nodes[1]; ncomponents=2, maxK=2)
        @test eigvals(one) ≈ eigvals(uni) rtol=1e-12
        @test scores(one) ≈ scores(uni) atol=1e-13
        @test eigenfunctions(one)[1].values ≈ eigenfunctions(uni).values atol=1e-13

        fitted = @inferred StatsAPI.fit(MFPCA, data; ncomponents=4,
            univariate=(; ncomponents=2, maxK=2))
        @test (@inferred ncomponents(fitted)) == 4
        @test (@inferred nobs(fitted)) == n
        @test (@inferred scores(fitted)) isa Matrix{Float64}
        @test (@inferred eigvals(fitted)) ≈ variances rtol=1e-12
        @test (@inferred eigenfunctions(fitted)) isa Vector{FPCBasis{Float64}}
        @test (@inferred StatsAPI.predict(fitted, data)) ≈ scores(fitted) atol=1e-13
        @test ids(fitted) == labels
        restored = @inferred reconstruct(fitted)
        @test ids(restored) == labels
        @test ids(reconstruct(fitted, copy(scores(fitted)))) === nothing
        @test ids(reconstruct(fitted, scores(fitted)[1:3, :])) === nothing
        for j in 1:2
            @test component(restored, j).values ≈ nodes[j].values atol=1e-13
            @test eigenfunctions(fitted)[j].grid == grids[j]
        end
        weighted = StatsAPI.fit(MFPCA, data; ncomponents=4, weights=[2.0, 3.0],
            univariate=(; ncomponents=2, maxK=2))
        @test eigvals(weighted) ≈ [18.0, 8.0, 6.0, 1.5] rtol=1e-12
        geometry = sum(weighted.weights[j] .* (b.values' * (b.weights .* b.values))
            for (j,b) in enumerate(eigenfunctions(weighted)))
        @test geometry ≈ I atol=1e-13
        @test scores(weighted)' * scores(weighted)/(n-1) ≈ Diagonal(eigvals(weighted)) atol=1e-12
        for m in 1:4
            c = weighted.coefficients[:, m]
            @test c[argmax(abs.(c))] > 0
        end
        # Global weight scaling rescales eigenvalues and scores, preserving fits.
        scaled = StatsAPI.fit(MFPCA, data; ncomponents=4, weights=[14.0, 21.0],
            univariate=(; ncomponents=2, maxK=2))
        @test eigvals(scaled) ≈ 7eigvals(weighted) rtol=1e-12
        @test scores(scaled) ≈ sqrt(7)*scores(weighted) atol=1e-12
        for j in 1:2
            @test eigenfunctions(scaled)[j].values ≈ eigenfunctions(weighted)[j].values/sqrt(7) atol=1e-13
            @test component(reconstruct(scaled), j).values ≈ nodes[j].values atol=1e-13
        end
        # Node unit changes are canceled by inverse-square weight changes.
        rescaled_data = MultivariateFunctionalData([nodes[1],
            GridFunctionalData(5nodes[2].values, grids[2]; ids=labels)])
        units = StatsAPI.fit(MFPCA, rescaled_data; ncomponents=4, weights=[2.0, 3/25],
            univariate=(; ncomponents=2, maxK=2))
        @test eigvals(units) ≈ eigvals(weighted) rtol=1e-12
        @test scores(units) ≈ scores(weighted) atol=1e-12

        points = [[0.05, 0.4, 0.8], [-1.7, -1.0, -0.3]]
        on_grid = @inferred reconstruct(fitted; grid=points)
        for j in 1:2
            @test component(on_grid, j).grid == points[j]
            @test component(on_grid, j).values ≈ evaluate(nodes[j], points[j], PiecewiseLinear()) atol=1e-13
        end
        report = sprint(show, fitted)
        for term in ("weights", "univariate", "eigen_tol", "score_centering", "largest_absolute_coefficient")
            @test occursin(term, report)
        end
        @test sprint(show, MIME"text/plain"(), fitted) == report
        # Fit input arrays and metadata must be independent.
        component(data, 1).values .= 100
        component(data, 1).grid[2] = 0.2
        component(data, 1).ids[1] = "changed"
        @test ids(fitted) == labels
        @test eigenfunctions(fitted)[1].grid == grids[1]
        @test component(reconstruct(fitted), 1).values ≈ nodes[1].values atol=1e-13
    end

    @testset "Mixed dense/PACE nodes: R centering and normalization" begin
        dense = component(mfpca_fixture().data, 1)[1:8]
        g = [2.0, 2.1, 2.4, 2.7, 3.0]
        times = [g[[1, 2+i%2, 5]] for i in 1:8]
        sparse = IrregularFunctionalData(times, [fill(5+cos(i), length(t)) for (i,t) in enumerate(times)];
            domain=(2,3), ids=ids(dense))
        data = MultivariateFunctionalData([dense, sparse])
        options = (; ncomponents=[2, 1], maxK=[2, 1], grid=[nothing, g],
            mean=[nothing, (; grid=g, values=zeros(5))],
            covariance=[nothing, (; grid=g, surface=ones(5, 5))],
            noise_variance=[nothing, 0.2])
        fitted = StatsAPI.fit(MFPCA, data; ncomponents=2, univariate=options, weights=[0.8, 1.4])
        marginal = StatsAPI.fit(FPCA, data; basis=:marginal, options...)
        @test component(fitted, 2).method == :pace
        @test fitted.settings.univariate_counts == [2, 1]
        xi = hcat(scores(marginal)...)
        a = xi .* [sqrt(0.8) sqrt(0.8) sqrt(1.4)]
        centered = a .- sum(a; dims=1)/8
        @test eigvals(fitted) ≈ reverse(eigvals(Symmetric(centered'centered/7)))[1:2] rtol=1e-12
        @test norm(sum(a; dims=1)) > 1 # Ensure centering matters in this fixture.
        @test norm(eigvals(fitted) - reverse(eigvals(Symmetric(a'a/7)))[1:2]) > 1
        @test norm(fitted.settings.score_scales .- 1) > 0.1
        @test scores(fitted) ≈ (a * fitted.coefficients) .* fitted.settings.score_scales' atol=1e-13
        @test (@inferred StatsAPI.predict(fitted, data)) ≈ scores(fitted) atol=1e-13
        subset = data[2:5]
        @test StatsAPI.predict(fitted, subset) ≈ scores(fitted)[2:5, :] atol=1e-13
        @test ids(reconstruct(fitted)) == ids(data)

        golden = readdlm(joinpath(@__DIR__, "data", "m4_mfpca_pace.csv"), ','; comments=true)
        function reference(mode, quantity, node=0)
            rows = golden[(golden[:, 1] .== mode) .& (golden[:, 2] .== quantity) .&
                (golden[:, 3] .== node), :]
            reshape(Float64.(rows[:, 6]), Int(maximum(rows[:, 4])), Int(maximum(rows[:, 5])))
        end
        relative(a, b) = norm(a-b)/norm(b)
        errors = Float64[]
        for (mode, weights) in (("given_pace_unit", ones(2)),
                ("given_pace_weighted", [0.8, 1.4]))
            result = StatsAPI.fit(MFPCA, data; ncomponents=2, univariate=options, weights)
            push!(errors, relative(eigvals(result), vec(reference(mode, "eigenvalue"))))
            push!(errors, relative(result.coefficients, reference(mode, "coefficient")))
            push!(errors, relative(result.settings.normalization_factors,
                vec(reference(mode, "normalization_factor"))))
            push!(errors, relative(scores(result), reference(mode, "score")))
            recovered = reconstruct(result)
            for j in 1:2
                push!(errors, relative(eigenfunctions(result)[j].values,
                    reference(mode, "function", j)))
                push!(errors, relative(component(recovered, j).values, reference(mode, "fit", j)))
            end
            @test maximum(errors) < 1e-8
        end
        println("MFPCA nonzero-mean PACE given-expansion maximum relative error: ", maximum(errors))
        options.ncomponents[1] = 1
        @test fitted.settings.univariate.ncomponents == [2, 1]
    end

    @testset "Float32 and promoted prediction/reconstruction" begin
        data64 = mfpca_fixture().data
        data = MultivariateFunctionalData([GridFunctionalData(Float32.(component(data64,j).values),
            Float32.(component(data64,j).grid)) for j in 1:3])
        fitted = StatsAPI.fit(MFPCA, data; ncomponents=3, univariate=(; ncomponents=2, maxK=2))
        @test fitted isa MFPCA{Float32}
        @test scores(fitted) isa Matrix{Float32}
        @test eigenfunctions(fitted) isa Vector{FPCBasis{Float32}}
        @test reconstruct(fitted) isa MultivariateFunctionalData{Float32}
        precise = Float64.(scores(fitted))
        recovered = reconstruct(fitted, precise)
        perturbed = copy(precise)
        perturbed[1, 1] += 1e-8
        difference = component(reconstruct(fitted, perturbed), 1).values[1, :] -
            component(recovered, 1).values[1, :]
        @test difference ≈ 1e-8 .* Float64.(eigenfunctions(fitted)[1].values[:, 1]) atol=2e-15
        @test recovered isa MultivariateFunctionalData{Float64}
        # Float64 target points must not be rounded to the training precision.
        points = [[0.123456789, 0.456789123], [-0.123456789, 0.456789123], [2.123456789, 2.456789123]]
        target = reconstruct(fitted; grid=points)
        @test target isa MultivariateFunctionalData{Float64}
        @test component(target, 1).grid == points[1]
        # Higher precision node predictions promote the multivariate output.
        promoted = MultivariateFunctionalData([GridFunctionalData(Float64.(component(data,j).values),
            Float64.(component(data,j).grid)) for j in 1:3])
        @test StatsAPI.predict(fitted, promoted) isa Matrix{Float64}
        weighted = StatsAPI.fit(MFPCA, data; ncomponents=3, weights=ones(Float64, 3),
            univariate=(; ncomponents=2, maxK=2))
        @test weighted isa MFPCA{Float64,Float32}
        @test eigenfunctions(weighted) isa Vector{FPCBasis{Float64}}
    end

    @testset "Validation and copied weights" begin
        data = mfpca_fixture().data
        options = (; ncomponents=2, maxK=2)
        @test_throws UndefKeywordError StatsAPI.fit(MFPCA, data)
        for count in (0, -1, 1.5, true, FVE(0.9))
            @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=count)
        end
        @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=7, univariate=options)
        @test_throws DimensionMismatch StatsAPI.fit(MFPCA, data; ncomponents=2, weights=[1, 2])
        for weights in ([0, 1, 1], [-1, 1, 1], [Inf, 1, 1], [NaN, 1, 1], 2, ["a", "b", "c"])
            @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=2, weights)
        end
        @test_throws ArgumentError StatsAPI.fit(MFPCA, data[1:1]; ncomponents=1)
        @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=1, univariate=[])
        @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=1, univariate=(; basis=:pooled))
        @test_throws DimensionMismatch StatsAPI.fit(MFPCA, data; ncomponents=1, univariate=(; ncomponents=[1, 2]))
        for tol in (-1, NaN, 1)
            @test_throws ArgumentError StatsAPI.fit(MFPCA, data; ncomponents=1, eigen_tol=tol)
        end
        constant = MultivariateFunctionalData([GridFunctionalData(ones(6, 5), collect(0.0:0.25:1))])
        @test_throws ArgumentError StatsAPI.fit(MFPCA, constant; ncomponents=1)
        basis_data = MultivariateFunctionalData([BasisFunctionalData(ones(6, 2),
            MonomialBasis((0, 1), 1))])
        @test_throws ArgumentError StatsAPI.fit(MFPCA, basis_data; ncomponents=1)
        # More requested components than the joint positive rank must throw.
        duplicate = MultivariateFunctionalData(fill(component(data, 1), 2))
        @test_throws ArgumentError StatsAPI.fit(MFPCA, duplicate; ncomponents=3,
            univariate=(; ncomponents=1, maxK=2))
        @test_throws ArgumentError StatsAPI.fit(MFPCA, duplicate; ncomponents=2,
            univariate=(; ncomponents=1, maxK=2))
        weights = [0.7, 1.5, 2.2]
        fitted = StatsAPI.fit(MFPCA, data; ncomponents=3, univariate=options, weights)
        weights[1] = 10
        @test fitted.weights == [0.7, 1.5, 2.2]
        @test_throws DimensionMismatch StatsAPI.predict(fitted, MultivariateFunctionalData([component(data, 1)]))
        shifted = MultivariateFunctionalData([GridFunctionalData(component(data, j).values,
            component(data, j).grid .+ 0.1) for j in 1:3])
        @test_throws ArgumentError StatsAPI.predict(fitted, shifted)
        @test_throws DimensionMismatch reconstruct(fitted, zeros(2, 2))
        @test_throws ArgumentError reconstruct(fitted, zeros(0, 3))
        @test_throws ArgumentError reconstruct(fitted, fill(NaN, 2, 3))
        @test_throws DimensionMismatch reconstruct(fitted; grid=[[0.0, 0.5]])
        @test_throws ArgumentError reconstruct(fitted; grid=2)
        @test_throws ArgumentError reconstruct(fitted; grid=[nothing, nothing, nothing])
        @test_throws ArgumentError reconstruct(fitted; grid=[0.5, 0.1])
        @test_throws ArgumentError reconstruct(fitted; grid=[-100.0, 100.0])
        @test_throws ArgumentError reconstruct(fitted; grid=[0.0, NaN])
    end
end
