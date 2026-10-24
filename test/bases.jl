@testset "Bases" begin
    @testset "B-splines" begin
        breaks = [0.0, 0.2, 0.6, 1.0]
        original = copy(breaks)
        b = @inferred BSplineBasis(4, breaks)
        @test breaks == original
        breaks[2] = 0.1
        @test b.breakpoints == original
        @test domain(b) == (0.0, 1.0)
        @test nbasis(b) == 6
        t = collect(range(0.0, 1.0; length=101))
        X = @inferred basismatrix(b, t)
        @test vec(sum(X; dims=2)) ≈ ones(length(t))
        @test X[1, 1] == X[end, end] == 1
        @test vec(sum(basismatrix(b, t; derivative=1); dims=2)) ≈
            zeros(length(t)) atol=1e-13
        points = [0.13, 0.33, 0.72]
        h = 1e-6
        for m in 1:2
            difference = (basismatrix(b, points .+ h; derivative=m - 1) -
                basismatrix(b, points .- h; derivative=m - 1)) / (2h)
            @test basismatrix(b, points; derivative=m) ≈ difference rtol=1e-8
        end
        @test basismatrix(b, reverse(t)) ≈ reverse(X; dims=1)
        @test basismatrix(b, t; derivative=4) == zeros(length(t), 6)
        @test penalty(b, 4) == zeros(6, 6)
        @test size(basismatrix(b, Float64[])) == (0, 6)
        @test nbasis(@inferred BSplineBasis(4, (0, 1), 7)) == 7
        @test eltype(@inferred BSplineBasis(2, Float32[0, 0.5, 1])) == Float32
        @test eltype(@inferred BSplineBasis(2, [0, 1])) == Float64
        @test eltype(@inferred BSplineBasis(2, (0.0f0, 1.0f0), 3)) == Float32
    end

    @testset "Analytic bases" begin
        f = @inferred FourierBasis((0, 2), 5)
        @test nbasis(f) == 5
        @test domain(f) == (0.0, 2.0)
        @test gram(f) ≈ Matrix{Float64}(I, 5, 5)
        @test diag(penalty(f, 1)) ≈ [0, pi^2, pi^2, (2pi)^2, (2pi)^2]
        @test basismatrix(f, [0.0]) ≈ reshape([inv(sqrt(2)), 0, 1, 0, 1], 1, :)
        t = [0.17, 0.63, 1.21]
        h = 1e-6
        @test basismatrix(f, t; derivative=1) ≈
            (basismatrix(f, t .+ h) - basismatrix(f, t .- h)) / (2h) rtol=1e-8
        b = @inferred MonomialBasis((0, 1), 3)
        @test nbasis(b) == 4
        @test gram(b) ≈ [1 / (i + j + 1) for i in 0:3, j in 0:3]
        @test penalty(b, 2) ≈ [0 0 0 0; 0 0 0 0; 0 0 4 6; 0 0 6 12]
        @test penalty(b, 4) == zeros(4, 4)
        @test basismatrix(b, [0.0, 1.0]; derivative=2) == [0 0 2 0; 0 0 2 6]
        @test eltype(@inferred FourierBasis((0.0f0, 1.0f0), 3)) == Float32
        @test eltype(@inferred MonomialBasis((0.0f0, 1.0f0), 2)) == Float32
    end

    @testset "Quadrature and inference" begin
        bases = (BSplineBasis(4, [-1.0, -0.5, 0.25, 2.0]),
            FourierBasis((-1.0, 2.0), 5), MonomialBasis((-1.0, 2.0), 4))
        t = collect(range(-1.0, 2.0; length=100001))
        w = quadrature_weights(t)
        for b in bases
            @test (@inferred nbasis(b)) isa Int
            @test (@inferred domain(b)) == (-1.0, 2.0)
            @test (@inferred gram(b)) isa Matrix{Float64}
            for m in 0:2
                X = @inferred basismatrix(b, t; derivative=m)
                P = @inferred penalty(b, m)
                numerical = transpose(X) * (w .* X)
                @test norm(numerical - P) / norm(P) < 1e-6
            end
            for badpoints in ([-1.01], [2.01], [NaN], [Inf])
                @test_throws ArgumentError basismatrix(b, badpoints)
            end
            @test_throws ArgumentError basismatrix(b, [0.0]; derivative=-1)
            @test_throws ArgumentError basismatrix(b, [0.0]; derivative=0.5)
            @test_throws ArgumentError penalty(b, -1)
            @test_throws ArgumentError penalty(b, 0.5)
        end
        for b in (BSplineBasis(3, Float32[0, 0.5, 1]),
                FourierBasis((0.0f0, 1.0f0), 3), MonomialBasis((0.0f0, 1.0f0), 2))
            @test (@inferred basismatrix(b, Float32[0, 0.25, 1])) isa Matrix{Float32}
            @test (@inferred gram(b)) isa Matrix{Float32}
            @test (@inferred penalty(b, 1)) isa Matrix{Float32}
            @test b == deepcopy(b)
        end
    end

    @testset "Invalid bases" begin
        for bad in (Float64[], [0.0], [0.0, 0.0], [1.0, 0.0], [0.0, Inf], [0.0, NaN])
            @test_throws ArgumentError BSplineBasis(4, bad)
        end
        @test_throws ArgumentError BSplineBasis(0, [0.0, 1.0])
        @test_throws ArgumentError BSplineBasis(4, (0, 1), 3)
        for bounds in ((1, 0), (0, 0), (0.0, Inf), (NaN, 1.0))
            @test_throws ArgumentError BSplineBasis(2, bounds, 3)
            @test_throws ArgumentError FourierBasis(bounds, 3)
            @test_throws ArgumentError MonomialBasis(bounds, 2)
        end
        @test_throws ArgumentError FourierBasis((0, 1), 0)
        @test_throws ArgumentError FourierBasis((0, 1), 2)
        @test_throws ArgumentError MonomialBasis((0, 1), -1)
    end
end
