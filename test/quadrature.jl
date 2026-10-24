@testset "Trapezoidal quadrature" begin
    @testset "Weights" begin
        uniform = collect(range(-1.0, 1.0; length=9))
        weights = quadrature_weights(uniform)
        @test weights == [0.125; fill(0.25, 7); 0.125]
        @test sum(weights) ≈ last(uniform) - first(uniform)

        nonuniform = [0.0, 0.1, 0.4, 1.0]
        weights = quadrature_weights(nonuniform)
        @test weights ≈ [0.05, 0.2, 0.45, 0.3]
        @test sum(weights) ≈ last(nonuniform) - first(nonuniform)
        @test quadrature_weights([2.0, 5.0]) == [1.5, 1.5]
        @test quadrature_weights(range(-1.0, 1.0; length=9)) ==
              quadrature_weights(uniform)
    end

    @testset "Linear integrands" begin
        rng = Random.Xoshiro(20261024)
        lower, upper = -1.5, 2.75
        interior = sort!(lower .+ (upper - lower) .* rand(rng, 31))
        grid = [lower; interior; upper]
        weights = quadrature_weights(grid)

        @test dot(weights, ones(length(grid))) ≈ upper - lower atol=1e-13
        @test dot(weights, grid) ≈ (upper^2 - lower^2) / 2 atol=1e-13
        for (slope, intercept) in ((2.0, -3.0), (-0.7, 1.25), (0.0, 4.0))
            exact = slope * (upper^2 - lower^2) / 2 + intercept * (upper - lower)
            @test dot(weights, slope .* grid .+ intercept) ≈ exact atol=1e-13
        end
    end

    @testset "Types and inference" begin
        @test (@inferred quadrature_weights(Float32[0, 0.25, 1])) isa Vector{Float32}
        @test (@inferred quadrature_weights(Float64[0, 0.25, 1])) isa Vector{Float64}
        @test (@inferred quadrature_weights([0, 1, 4])) isa Vector{Float64}
        @test quadrature_weights(Float32[0, 0.25, 1]) == Float32[0.125, 0.5, 0.375]
        @test quadrature_weights(Float64[0, 0.25, 1]) == [0.125, 0.5, 0.375]
        @test quadrature_weights([0, 1, 4]) == [0.5, 2.0, 1.5]
    end

    @testset "Invalid grids" begin
        for grid in (Float64[], [0.0])
            @test_throws ArgumentError("the grid needs at least two points") quadrature_weights(grid)
        end
        for grid in ([0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.5])
            @test_throws ArgumentError("the grid must be strictly increasing") quadrature_weights(grid)
        end
        for grid in ([0.0, Inf], [-Inf, 0.0], [0.0, NaN], [NaN, 1.0])
            @test_throws ArgumentError("the grid must be finite") quadrature_weights(grid)
        end
    end
end
