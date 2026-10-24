@testset "FPCBasis" begin
    @testset "Interpolation and exact Gram matrix" begin
        grid = [0.0, 0.25, 1.0]
        values = [1.0 0.0; 1.0 0.25; 1.0 1.0]
        b = @inferred FPCBasis(grid, values)
        @test (@inferred nbasis(b)) == 2
        @test (@inferred domain(b)) == (0.0, 1.0)
        @test eltype(b) == Float64
        @test b.weights == [0.125, 0.5, 0.375]
        points = [1.0, 0.125, 0.0, 0.625, 0.25]
        @test (@inferred basismatrix(b, points)) == hcat(ones(5), points)
        @test basismatrix(b, grid; derivative=0) == values
        @test size(@inferred basismatrix(b, Float64[])) == (0, 2)
        @test (@inferred gram(b)) ≈ [1.0 0.5; 0.5 1 / 3]
        @test (@inferred penalty(b, 0)) == gram(b)

        # A discrete orthonormal pair has an analytic, nonidentity exact Gram.
        normalized = FPCBasis([0.0, 0.5, 1.0],
            [1.0 -sqrt(2); 1.0 0.0; 1.0 sqrt(2)])
        @test transpose(normalized.values) *
            (normalized.weights .* normalized.values) ≈ Matrix{Float64}(I, 2, 2)
        @test gram(normalized) ≈ [1.0 0.0; 0.0 2 / 3]

        # The exact constant-function integral remains finite even though the
        # unscaled function-value product would overflow (or underflow).
        narrow = FPCBasis([0.0, 1e-310], [1e155; 1e155;;])
        wide = FPCBasis([0.0, 1e308], [1e-170; 1e-170;;])
        @test only(gram(narrow)) ≈ 1.0
        @test only(gram(wide)) ≈ 1e-32

        single = @inferred FPCBasis(Float32[0, 0.25, 1],
            Float32[1 0; 1 0.25; 1 1])
        @test eltype(single) == Float32
        @test (@inferred basismatrix(single, Float32[0, 0.5, 1])) isa Matrix{Float32}
        @test (@inferred gram(single)) isa Matrix{Float32}
        @test (@inferred penalty(single, 0)) isa Matrix{Float32}
        @test eltype(@inferred FPCBasis([0, 1], [1 0; 1 1])) == Float64
        @test basismatrix(single, [0.5]) isa Matrix{Float64}

        fd = BasisFunctionalData([2.0 3.0], b)
        @test evaluate(fd, [0.0, 0.4, 1.0]) ≈ [2.0 3.2 5.0]
    end

    @testset "Copies, equality, hashing and display" begin
        grid = [0.0, 0.25, 1.0]
        values = [1.0 0.0; 1.0 0.25; 1.0 1.0]
        weights = [0.125, 0.5, 0.375]
        b = @inferred FPCBasis(grid, values; weights)
        original = deepcopy(b)
        grid[2] = 0.5
        values[1, 1] = 2
        weights[1] = 0.5
        @test b == original
        @test b.grid !== grid
        @test b.values !== values
        @test b.weights !== weights
        single = FPCBasis(Float32.(b.grid), Float32.(b.values);
            weights=Float32.(b.weights))
        @test b == single
        @test isequal(b, single)
        @test hash(b) == hash(single)
        @test hash(b, UInt(123)) == hash(deepcopy(b), UInt(123))
        @test Dict(b => :found)[deepcopy(b)] == :found
        @test deepcopy(b) in Set([b])
        otherweights = FPCBasis(b.grid, b.values; weights=ones(3))
        @test b != otherweights
        text = sprint(show, b)
        @test text == "FPCBasis(2 functions on [0.0, 1.0], 3 grid points)"
        @test sprint(show, MIME"text/plain"(), b) == text

        # Follow array isequal semantics, which distinguish signed zeros.
        positivezero = FPCBasis([0.0, 1.0], [0.0; 1.0;;])
        negativezero = FPCBasis([0.0, 1.0], [-0.0; 1.0;;])
        @test positivezero == negativezero
        @test !isequal(positivezero, negativezero)
        @test isequal(negativezero, deepcopy(negativezero))
        @test hash(negativezero) == hash(deepcopy(negativezero))
        @test length(Dict(positivezero => 1, negativezero => 2)) == 2
    end

    @testset "Validation" begin
        grid = [0.0, 0.5, 1.0]
        values = ones(3, 2)
        b = FPCBasis(grid, values)
        for bad in (Float64[], [0.0], [0.0, 0.0], [1.0, 0.0],
                [0.0, NaN], [0.0, Inf])
            @test_throws ArgumentError FPCBasis(bad, ones(length(bad), 1))
        end
        @test_throws ArgumentError FPCBasis(grid, zeros(3, 0))
        @test_throws DimensionMismatch FPCBasis(grid, ones(2, 1))
        @test_throws DimensionMismatch FPCBasis(grid, values; weights=ones(2))
        for badweights in ([0.0, 0.5, 0.5], [-0.1, 0.5, 0.6],
                [NaN, 0.5, 0.5], [Inf, 0.5, 0.5])
            @test_throws ArgumentError FPCBasis(grid, values; weights=badweights)
        end
        for badvalue in (NaN, Inf, -Inf)
            invalid = copy(values)
            invalid[1] = badvalue
            @test_throws ArgumentError FPCBasis(grid, invalid)
        end
        for points in ([-0.1], [1.1], [NaN], [Inf])
            @test_throws ArgumentError basismatrix(b, points)
        end
        for order in (-1, 0.5, 1, 2)
            @test_throws ArgumentError basismatrix(b, [0.5]; derivative=order)
            @test_throws ArgumentError penalty(b, order)
        end
    end
end
