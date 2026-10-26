import BSplineKit
using DelimitedFiles

@testset "Bases" begin
    @testset "B-splines" begin
        breaks = [0.0, 0.2, 0.6, 1.0]
        original = copy(breaks)
        # The backend type includes the order, so construction is inferred when
        # the order is known to the compiler; runtime orders have distinct types.
        b = @inferred (x -> BSplineBasis(4, x))(breaks)
        @test isconcretetype(fieldtype(typeof(b), :backend))
        @test breaks == original
        breaks[2] = 0.1
        @test b.breakpoints == original
        @test domain(b) == (0.0, 1.0)
        @test nbasis(b) == 6
        t = collect(range(0.0, 1.0; length=101))
        X = @inferred basis_matrix(b, t)
        @test vec(sum(X; dims=2)) ≈ ones(length(t))
        @test X[1, 1] == X[end, end] == 1
        @test vec(sum(basis_matrix(b, t; derivative=1); dims=2)) ≈
            zeros(length(t)) atol=1e-13
        points = [0.13, 0.33, 0.72]
        h = 1e-6
        for m in 1:2
            difference = (basis_matrix(b, points .+ h; derivative=m - 1) -
                basis_matrix(b, points .- h; derivative=m - 1)) / (2h)
            @test basis_matrix(b, points; derivative=m) ≈ difference rtol=1e-8
        end
        @test basis_matrix(b, reverse(t)) ≈ reverse(X; dims=1)
        @test basis_matrix(b, t; derivative=4) == zeros(length(t), 6)
        @test penalty(b, 4) == zeros(6, 6)
        @test size(basis_matrix(b, Float64[])) == (0, 6)
        @test nbasis(@inferred (() -> BSplineBasis(4, (0, 1), 7))()) == 7
        @test eltype(@inferred (x -> BSplineBasis(2, x))(Float32[0, 0.5, 1])) == Float32
        @test eltype(@inferred (x -> BSplineBasis(2, x))([0, 1])) == Float64
        @test eltype(@inferred (() -> BSplineBasis(2, (0.0f0, 1.0f0), 3))()) == Float32
    end

    @testset "Concrete B-spline backend" begin
        for T in (Float32, Float64), order in (2, 3, 4, 6)
            b = BSplineBasis(order, T[0, 0.2, 0.6, 1])
            @test isconcretetype(typeof(b))
            @test fieldtype(typeof(b), :backend) === typeof(b.backend)
            # Exercise the existing partial constructor used by FACE as well.
            @test typeof(BSplineBasis{T}(b.backend, b.breakpoints, order)) === typeof(b)
            t = T[0, 0.13, 0.33, 0.72, 1]
            for m in 0:order
                X = @inferred basis_matrix(b, t; derivative=m)
                P = @inferred penalty(b, m)
                @test X isa Matrix{T}
                @test P isa Matrix{T}
                if m < order
                    # Direct BSplineKit calls retain the previous arithmetic.
                    expected = zeros(T, length(t), nbasis(b))
                    BSplineKit.collocation_matrix!(expected, b.backend, t,
                        BSplineKit.Derivative(m); clip_threshold=0)
                    @test X == expected
                    expected_penalty = zeros(T, nbasis(b), nbasis(b))
                    d = BSplineKit.Derivative(m)
                    BSplineKit.galerkin_matrix!(expected_penalty, b.backend, (d, d))
                    @test P == (expected_penalty + transpose(expected_penalty)) / 2
                else
                    @test iszero(X)
                    @test iszero(P)
                end
            end
        end
    end

    @testset "Analytic bases" begin
        f = @inferred FourierBasis((0, 2), 5)
        @test nbasis(f) == 5
        @test domain(f) == (0.0, 2.0)
        @test gram(f) ≈ Matrix{Float64}(I, 5, 5)
        @test diag(penalty(f, 1)) ≈ [0, pi^2, pi^2, (2pi)^2, (2pi)^2]
        @test basis_matrix(f, [0.0]) ≈ reshape([inv(sqrt(2)), 0, 1, 0, 1], 1, :)
        t = [0.17, 0.63, 1.21]
        h = 1e-6
        @test basis_matrix(f, t; derivative=1) ≈
            (basis_matrix(f, t .+ h) - basis_matrix(f, t .- h)) / (2h) rtol=1e-8
        b = @inferred MonomialBasis((0, 1), 3)
        @test nbasis(b) == 4
        @test gram(b) ≈ [1 / (i + j + 1) for i in 0:3, j in 0:3]
        @test penalty(b, 2) ≈ [0 0 0 0; 0 0 0 0; 0 0 4 6; 0 0 6 12]
        @test penalty(b, 4) == zeros(4, 4)
        @test basis_matrix(b, [0.0, 1.0]; derivative=2) == [0 0 2 0; 0 0 2 6]
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
                X = @inferred basis_matrix(b, t; derivative=m)
                P = @inferred penalty(b, m)
                numerical = transpose(X) * (w .* X)
                @test norm(numerical - P) / norm(P) < 1e-6
            end
            for badpoints in ([-1.01], [2.01], [NaN], [Inf])
                @test_throws ArgumentError basis_matrix(b, badpoints)
            end
            @test_throws ArgumentError basis_matrix(b, [0.0]; derivative=-1)
            @test_throws ArgumentError basis_matrix(b, [0.0]; derivative=0.5)
            @test_throws ArgumentError penalty(b, -1)
            @test_throws ArgumentError penalty(b, 0.5)
        end
        for b in (BSplineBasis(3, Float32[0, 0.5, 1]),
                FourierBasis((0.0f0, 1.0f0), 3), MonomialBasis((0.0f0, 1.0f0), 2))
            @test (@inferred basis_matrix(b, Float32[0, 0.25, 1])) isa Matrix{Float32}
            @test (@inferred gram(b)) isa Matrix{Float32}
            @test (@inferred penalty(b, 1)) isa Matrix{Float32}
            @test b == deepcopy(b)
        end
    end

    @testset "Hash and display" begin
        pairs = ((BSplineBasis(3, [0.0, 0.5, 1.0]), BSplineBasis(3, Float32[0, 0.5, 1])),
            (FourierBasis((0.0, 1.0), 3), FourierBasis((0.0f0, 1.0f0), 3)),
            (MonomialBasis((0.0, 1.0), 2), MonomialBasis((0.0f0, 1.0f0), 2)))
        for (b, single) in pairs
            @test hash(b) == hash(deepcopy(b))
            @test hash(b, UInt(123)) == hash(deepcopy(b), UInt(123))
            @test b == single
            @test hash(b) == hash(single)
            @test Dict(b => :found)[deepcopy(b)] == :found
            @test deepcopy(b) in Set([b])
            text = sprint(show, b)
            @test !occursin('\n', text)
            @test length(text) < 100
            @test sprint(show, MIME"text/plain"(), b) == text
        end
        @test sprint(show, first(pairs[1])) ==
            "BSplineBasis(order 3, 4 functions on [0.0, 1.0])"
        @test sprint(show, first(pairs[2])) == "FourierBasis(3 functions on [0.0, 1.0])"
        @test sprint(show, first(pairs[3])) ==
            "MonomialBasis(degree 2, 3 functions on [0.0, 1.0])"
    end

    @testset "mgcv P-splines" begin
        reference = readdlm(joinpath(@__DIR__, "data", "psplines_mgcv.csv"), ',', Any; comments=true)
        for (case, degree, difference_order) in ((1, 3, 1), (2, 3, 1), (3, 4, 2))
            rows(q) = reference[(reference[:, 1] .== case) .& (reference[:, 2] .== q), :]
            values(q) = Float64.(rows(q)[:, 5])
            function matrix(q)
                r = rows(q)
                out = zeros(maximum(Int.(r[:, 3])), maximum(Int.(r[:, 4])))
                for row in eachrow(r)
                    out[Int(row[3]), Int(row[4])] = Float64(row[5])
                end
                out
            end
            range_ = values("range")
            k = size(matrix("S"), 1)
            b = PSplineBasis((range_[1], range_[2]), k; degree, difference_order)
            x = values("points")
            @test nbasis(b) == k
            @test domain(b) == (range_[1], range_[2])
            @test b.knots ≈ values("knots") rtol=1e-14
            @test basis_matrix(b, x) ≈ matrix("value") rtol=1e-12
            @test basis_matrix(b, x; derivative=1) ≈ matrix("derivative") rtol=1e-12
            @test difference_penalty(b) == matrix("S")
        end
        # Integrated products over the domain. With Greville coefficients the
        # basis reproduces 1 and t exactly, so G and the derivative penalties
        # have analytic quadratic forms.
        b = PSplineBasis((-1.0, 2.5), 9; degree=3, difference_order=2)
        a, z = domain(b)
        one_ = ones(nbasis(b))
        line = [sum(b.knots[j+1:j+3]) / 3 for j in 1:nbasis(b)]
        t = collect(range(a, z; length=17))
        @test basis_matrix(b, t) * one_ ≈ ones(17)
        @test basis_matrix(b, t) * line ≈ t
        G = gram(b)
        @test G ≈ transpose(G)
        @test dot(one_, G, one_) ≈ z - a rtol=1e-12
        @test dot(line, G, line) ≈ (z^3 - a^3) / 3 rtol=1e-12
        @test dot(line, G, one_) ≈ (z^2 - a^2) / 2 rtol=1e-12
        @test norm(penalty(b, 1) * one_) < 1e-12
        @test dot(line, penalty(b, 1), line) ≈ z - a rtol=1e-12
        @test norm(penalty(b, 2) * line) < 1e-10
        @test penalty(b, 4) == zeros(9, 9)
        @test basis_matrix(b, t; derivative=4) == zeros(17, 9)
        # The integrated penalty matches a fine composite Simpson rule.
        fine = collect(range(a, z; length=4001))
        w = fill(2.0, 4001); w[2:2:end] .= 4.0; w[[1, end]] .= 1.0; w .*= (fine[2] - fine[1]) / 3
        D2 = basis_matrix(b, fine; derivative=2)
        @test penalty(b, 2) ≈ transpose(D2) * (w .* D2) rtol=1e-6
        # Cubic data are reproduced exactly by an unpenalized fit.
        cubic = GridFunctionalData(reshape(t .^ 3, 1, :), t)
        @test evaluate(to_basis(cubic, b), t) ≈ reshape(t .^ 3, 1, :)
        @test difference_penalty(b, 0) == Matrix(1.0I, 9, 9)
        @test difference_penalty(b, 1) == difference_penalty(PSplineBasis((-1.0, 2.5), 9; difference_order=1))
        @test eltype(PSplineBasis((0.0f0, 1.0f0), 6)) == Float32
        @test eltype(difference_penalty(PSplineBasis((0.0f0, 1.0f0), 6))) == Float32
        @test PSplineBasis((0, 1), 5; degree=3, difference_order=1) == PSplineBasis((0.0, 1.0), 5; difference_order=1)
        @test PSplineBasis((0, 1), 5) != PSplineBasis((0, 1), 5; difference_order=1)
        @test hash(PSplineBasis((0, 1), 5)) == hash(PSplineBasis((0.0, 1.0), 5))
        @test sprint(show, PSplineBasis((0, 1), 5; difference_order=1)) ==
            "PSplineBasis(degree 3, 5 functions on [0.0, 1.0], difference order 1)"
        @test sprint(show, MIME"text/plain"(), b) == sprint(show, b)
        @test_throws ArgumentError PSplineBasis((0, 1), 3)
        @test_throws ArgumentError PSplineBasis((0, 1), 5; degree=-1)
        @test_throws ArgumentError PSplineBasis((0, 1), 5; difference_order=5)
        @test_throws ArgumentError PSplineBasis((1, 0), 5)
        @test_throws ArgumentError basis_matrix(b, [3.0])
        @test_throws ArgumentError difference_penalty(b, 9)
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
