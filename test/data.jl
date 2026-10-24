struct TestLinearBasis <: AbstractBasis end
FunctionalDataAnalysis.domain(::TestLinearBasis) = (0.0, 1.0)
FunctionalDataAnalysis.nbasis(::TestLinearBasis) = 2
function FunctionalDataAnalysis.basismatrix(::TestLinearBasis, t; derivative=0)
    derivative == 0 && return hcat(ones(length(t)), t)
    derivative == 1 && return hcat(zeros(length(t)), ones(length(t)))
    zeros(length(t), 2)
end
FunctionalDataAnalysis.gram(::TestLinearBasis) = [1.0 0.5; 0.5 1/3]
FunctionalDataAnalysis.penalty(b::TestLinearBasis, order) =
    order == 0 ? gram(b) : order == 1 ? [0.0 0.0; 0.0 1.0] : zeros(2, 2)

@testset "Functional data" begin
    @testset "Grid constructors and masks" begin
        grid = [0.0, 0.1, 0.4, 1.0]
        values = reshape(collect(1.0:12.0), 3, 4)
        fd = @inferred GridFunctionalData(values, grid)
        values[1] = -10
        grid[1] = -1
        @test fd.values[1] == 1
        @test fd.grid[1] == 0
        @test fd.weights ≈ [0.05, 0.2, 0.45, 0.3]
        @test fd.observed === nothing
        @test nobs(fd) == length(fd) == 3
        @test nobs_per_curve(fd) == [4, 4, 4]
        @test domain(fd) == (0.0, 1.0)
        @test eltype(fd) == Float64
        masked = @inferred GridFunctionalData([1.0 missing 3.0; missing 2.0 4.0], [0.0, 0.5, 1.0])
        @test masked.observed == Bool[1 0 1; 0 1 1]
        @test masked.values == [1 0 3; 0 2 4]
        @test nobs_per_curve(masked) == [2, 2]
        @test masked[1] == ([0.0, 1.0], [1.0, 3.0])
        @test (@inferred GridFunctionalData(Float32[1 2], Float32[0, 1])) isa GridFunctionalData{Float32}
        @test (@inferred GridFunctionalData([1 2], [0, 1])) isa GridFunctionalData{Float64}
        @test (@inferred GridFunctionalData(fill(missing, 1, 2), [0, 1])).observed == falses(1, 2)
        for bad in (fill(NaN, 1, 2), fill(Inf, 1, 2))
            @test_throws ArgumentError GridFunctionalData(bad, [0, 1])
        end
        @test_throws DimensionMismatch GridFunctionalData(ones(3, 3), fd.grid)
        @test_throws ArgumentError GridFunctionalData(zeros(0, 4), fd.grid)
        @test_throws ArgumentError GridFunctionalData(ones(1, 1), [0.0])
        @test_throws ArgumentError GridFunctionalData(ones(1, 2), [0.0, 0.0])
        @test_throws ArgumentError GridFunctionalData(ones(1, 2), [1.0, 0.0])
        @test_throws ArgumentError GridFunctionalData(ones(1, 2), [0.0, Inf])
        @test_throws ArgumentError GridFunctionalData(ones(1, 2), [0.0, NaN])
    end

    @testset "Irregular constructors" begin
        times = [[1.0, 0.0, 0.5], [0.2, 0.8]]
        values = [[3.0, 1.0, 2.0], [5.0, 6.0]]
        fd = @inferred IrregularFunctionalData(times, values)
        @test fd[1] == ([0.0, 0.5, 1.0], [1.0, 2.0, 3.0])
        @test fd.offsets == [1, 4, 6]
        @test fd.domain == domain(fd) == (0.0, 1.0)
        times[1][1] = 99
        values[1][1] = 99
        @test fd[1][2] == [1, 2, 3]
        rebuilt = @inferred IrregularFunctionalData(fd.times, fd.values, fd.offsets; domain=domain(fd))
        @test rebuilt.times == fd.times
        @test rebuilt.values == fd.values
        @test rebuilt.offsets == fd.offsets
        rebuilt.times[1] = -1
        rebuilt.values[1] = -1
        rebuilt.offsets[2] = 2
        @test fd.times[1] == 0 && fd.values[1] == 1 && fd.offsets[2] == 4
        tied = @inferred IrregularFunctionalData([[1, 0, 1]], [[2, 1, 4]]; ties=:mean)
        @test tied[1] == ([0.0, 1.0], [1.0, 3.0])
        @test (@inferred IrregularFunctionalData([Float32[0, 1]], [Float32[1, 2]])) isa IrregularFunctionalData{Float32}
        @test (@inferred IrregularFunctionalData(Float32[0, 1], Float32[1, 2], [1, 3])) isa IrregularFunctionalData{Float32}
        @test (@inferred IrregularFunctionalData([[1]], [[2]])).domain == (1.0, 1.0)
        @test_throws DimensionMismatch IrregularFunctionalData([[0.0]], [[1.0], [2.0]])
        @test_throws DimensionMismatch IrregularFunctionalData([[0.0, 1.0]], [[1.0]])
        @test_throws DimensionMismatch IrregularFunctionalData([0.0], [1.0, 2.0], [1, 2])
        @test_throws ArgumentError IrregularFunctionalData(Vector{Float64}[], Vector{Float64}[])
        @test_throws ArgumentError IrregularFunctionalData([Float64[]], [Float64[]])
        @test_throws ArgumentError IrregularFunctionalData([[0.0, 0.0]], [[1.0, 2.0]])
        @test_throws ArgumentError IrregularFunctionalData([[0.0]], [[NaN]])
        @test_throws ArgumentError IrregularFunctionalData([[Inf]], [[1.0]])
        @test_throws ArgumentError IrregularFunctionalData([[NaN]], [[1.0]])
        @test_throws ArgumentError IrregularFunctionalData([[0.0]], [[Inf]])
        @test_throws ArgumentError IrregularFunctionalData([[0.0]], [[1.0]]; ties=:drop)
        @test_throws ArgumentError IrregularFunctionalData([[0.0, 1.0]], [[1.0, 2.0]]; domain=(0.1, 1.0))
        @test_throws ArgumentError IrregularFunctionalData([[0.0]], [[1.0]]; domain=(1.0, 0.0))
        @test_throws ArgumentError IrregularFunctionalData([[0.0]], [[1.0]]; domain=(0.0, Inf))
        for offsets in ([0, 3], [1, 2], [1, 1, 3], [1, 4, 3], Int[])
            @test_throws ArgumentError IrregularFunctionalData([0, 1], [1, 2], offsets)
        end
    end

    @testset "Basis data" begin
        b = MonomialBasis((0, 1), 2)
        coefs = [1.0 2.0 3.0; 4.0 5.0 6.0]
        fd = @inferred BasisFunctionalData(coefs, b)
        coefs[1] = 99
        @test fd.coefs[1] == 1
        @test fd.basis == b
        @test (@inferred fd[1]) isa BasisFunctionalData
        @test fd[1].coefs == [1 2 3]
        @test domain(fd) == (0.0, 1.0)
        @test nobs_per_curve(fd) == [3, 3]
        t = [0.0, 0.25, 1.0]
        @test (@inferred evaluate(fd, t)) == fd.coefs * transpose(basismatrix(b, t))
        @test (@inferred fd(t)) == evaluate(fd, t)
        @test (@inferred evaluate(fd, t; derivative=1)) == fd.coefs * transpose(basismatrix(b, t; derivative=1))
        @test (@inferred BasisFunctionalData(Float32[1 2], MonomialBasis((0.0f0, 1.0f0), 1))) isa BasisFunctionalData{Float32}
        spline = BSplineBasis(3, [0.0, 0.5, 1.0])
        sf = BasisFunctionalData(ones(1, nbasis(spline)), spline)
        @test sf.basis === spline
        @test sf[1].basis === spline
        @test first(collect(sf)).basis === spline
        @test_throws DimensionMismatch BasisFunctionalData(ones(2, 2), b)
        @test_throws ArgumentError BasisFunctionalData(zeros(0, 3), b)
        @test_throws ArgumentError BasisFunctionalData(fill(NaN, 1, 3), b)
        @test_throws ArgumentError evaluate(fd, [-0.1])
        custom = @inferred BasisFunctionalData([1 2], TestLinearBasis())
        @test (@inferred evaluate(custom, [0.5])) == reshape([2.0], 1, 1)
        @test to_basis(GridFunctionalData([1 2 3], [0.0, 0.5, 1.0]), TestLinearBasis()).coefs ≈ [1 2]
    end

    @testset "Evaluation and conversion" begin
        times = [[0.0, 0.25, 1.0], [0.25, 0.75]]
        values = [2 .* t .+ 1 for t in times]
        ir = IrregularFunctionalData(times, values; domain=(0, 1))
        t = [0.0, 0.25, 0.5, 0.75, 1.0]
        y = @inferred evaluate(ir, t, PiecewiseLinear())
        @test y[1, :] == 2t .+ 1
        @test y[2, 2:4] == 2t[2:4] .+ 1
        @test isnan(y[2, 1]) && isnan(y[2, end])
        @test (@inferred evaluate(ir, [0.25], NoInterpolation())) == reshape([1.5, 1.5], 2, 1)
        @test_throws ArgumentError evaluate(ir, [0.5], NoInterpolation())
        @test_throws ArgumentError evaluate(ir, t)
        @test_throws ArgumentError evaluate(ir, [NaN], PiecewiseLinear())
        @test_throws ArgumentError evaluate(ir, [Inf], NoInterpolation())
        gap = GridFunctionalData([1 missing 5], [0.0, 0.5, 1.0])
        @test evaluate(gap, [0.5], PiecewiseLinear()) == reshape([3.0], 1, 1)
        gr = @inferred to_grid(ir, t, PiecewiseLinear())
        @test gr.observed == Bool[1 1 1 1 1; 0 1 1 1 0]
        @test all(isfinite, gr.values)
        exact = @inferred to_grid(ir, t, NoInterpolation())
        @test exact.observed == Bool[1 1 0 0 1; 0 1 0 1 0]
        @test_throws ArgumentError to_grid(ir, [0.0, 0.5, 1.0], NoInterpolation())
        shifted = IrregularFunctionalData([[0.0, 0.5 + eps(), 1.0]], [[1.0, 2.0, 3.0]])
        @test_throws ArgumentError to_grid(shifted, [0.0, 0.5, 1.0], NoInterpolation())
        for fd in (GridFunctionalData([1 2 3; 4 5 6], [0.0, 0.5, 1.0]),
                GridFunctionalData([missing 2 3; 4 missing 6], [0.0, 0.5, 1.0]))
            roundtrip = @inferred to_grid(@inferred(to_irregular(fd)), fd.grid, NoInterpolation())
            @test roundtrip.grid == fd.grid
            @test roundtrip.values == fd.values
            @test roundtrip.observed == fd.observed
            @test evaluate(fd, [0.5], PiecewiseLinear()) isa Matrix{Float64}
        end
        emptycurve = GridFunctionalData(fill(missing, 1, 2), [0, 1])
        @test_throws ArgumentError to_irregular(emptycurve)
        @test all(isnan, evaluate(emptycurve, [0.0], PiecewiseLinear()))
        singleton = IrregularFunctionalData([[0.5]], [[2.0]])
        @test isequal(evaluate(singleton, [0.0, 0.5, 1.0], PiecewiseLinear()), [NaN 2.0 NaN])
        bfd = BasisFunctionalData([1 2], MonomialBasis((0, 1), 1))
        bg = @inferred to_grid(bfd, t)
        @test bg.values == reshape(2t .+ 1, 1, :)
        bi = @inferred to_irregular(bfd; grid=t)
        @test bi[1] == (t, vec(bg.values))
        @test domain(to_irregular(bfd; grid=[0.25, 0.5, 0.75])) == domain(bfd)
        @test (@inferred to_irregular(ir)).values == ir.values
        @test_throws ArgumentError to_grid(ir, [0.5], PiecewiseLinear())
        @test (@inferred evaluate(GridFunctionalData(Float32[1 2], Float32[0, 1]), Float32[0.5], PiecewiseLinear())) isa Matrix{Float32}
        @test (@inferred evaluate(IrregularFunctionalData([Float32[0, 1]], [Float32[1, 2]]), Float32[0], NoInterpolation())) isa Matrix{Float32}
    end

    @testset "Basis fitting" begin
        t = collect(range(0.0, 1.0; length=31))
        for basis in (MonomialBasis((0, 1), 2), FourierBasis((0, 1), 5), BSplineBasis(4, (0, 1), 6))
            c = reshape(collect(1.0:2nbasis(basis)), 2, :)
            values = c * transpose(basismatrix(basis, t))
            fd = GridFunctionalData(values, t)
            fit = @inferred to_basis(fd, basis)
            @test fit.coefs ≈ c rtol=1e-10
            @test evaluate(fit, t) ≈ values rtol=1e-12
            irregular = to_irregular(fd)
            @test to_basis(irregular, basis).coefs ≈ c rtol=1e-10
        end
        b = MonomialBasis((0, 1), 2)
        values = reshape(sin.(3t), 1, :)
        raw = GridFunctionalData(values, t)
        X = basismatrix(b, t)
        lambda = 0.3
        expected = (transpose(X) * X + lambda * penalty(b, 2)) \ (transpose(X) * vec(values))
        @test to_basis(raw, b; lambda).coefs ≈ transpose(expected)
        short = IrregularFunctionalData([[0.0, 1.0]], [[1.0, 2.0]])
        @test_throws ArgumentError to_basis(short, b)
        @test all(isfinite, to_basis(short, b; lambda=0.1).coefs)
        @test_throws ArgumentError to_basis(short, b; lambda=0.1, order=3)
        @test_throws ArgumentError to_basis(raw, b; lambda=-1)
        @test_throws ArgumentError to_basis(raw, b; lambda=Inf)
        @test_throws ArgumentError to_basis(raw, b; order=-1)
        @test_throws ArgumentError to_basis(BasisFunctionalData([1 2 3], b), b)
        @test_throws ArgumentError to_basis(IrregularFunctionalData([[0.0, 1.0, 2.0]], [[1, 2, 3]]), b)
        rankdeficient = GridFunctionalData([1 2 3], [0.0, 0.5, 1.0])
        @test_throws ArgumentError to_basis(rankdeficient, FourierBasis((0, 1), 3))
    end

    @testset "Subject operations and design" begin
        grid = GridFunctionalData([1 2 3; 4 5 6; 7 8 9], [0.0, 0.5, 1.0])
        ir = to_irregular(grid)
        basis = to_basis(grid, MonomialBasis((0, 1), 2))
        for fd in (grid, ir, basis)
            @test (@inferred fd[1:2]) isa typeof(fd)
            @test (@inferred fd[[3, 1]]) isa typeof(fd)
            @test (@inferred fd[Bool[1, 0, 1]]) isa typeof(fd)
            @test nobs(fd[[3, 1]]) == 2
            @test nobs(vcat(fd[1:1], fd[2:3])) == 3
            @test length(collect(fd)) == 3
            @test nobs(fd) == length(fd)
            @test !occursin('\n', sprint(show, fd))
            @test occursin("n=3", sprint(show, fd))
            @test sprint(show, MIME"text/plain"(), fd) == sprint(show, fd)
            @test_throws BoundsError fd[0]
            @test_throws BoundsError fd[[1, 4]]
            @test_throws DimensionMismatch fd[Bool[1, 0]]
            @test_throws ArgumentError fd[Int[]]
            @test_throws ArgumentError fd[falses(3)]
        end
        observed = @inferred grid[1]
        observed[1][1] = -1
        observed[2][1] = -1
        @test grid.grid[1] == 0 && grid.values[1] == 1
        @test (@inferred ir[1])[1] == grid.grid
        @test collect(basis)[1] isa BasisFunctionalData
        @test vcat(grid[1:1], grid[2:3]).values == grid.values
        @test vcat(ir[1:1], ir[2:3]).times == ir.times
        @test vcat(basis[1:1], basis[2:3]).coefs ≈ basis.coefs
        @test_throws ArgumentError vcat(grid, GridFunctionalData([1 2], [0, 1]))
        @test_throws ArgumentError vcat(ir, IrregularFunctionalData([[0, 2]], [[1, 2]]))
        @test_throws ArgumentError vcat(basis, BasisFunctionalData([1 2], MonomialBasis((0, 1), 1)))
        @test sampling_design(grid) == sampling_design(basis) == :regular
        dense = IrregularFunctionalData([collect(range(0, 1; length=n)) for n in (19, 21)], [ones(n) for n in (19, 21)])
        @test sampling_design(dense) == :dense_irregular
        @test sampling_design(dense; threshold=20.1) == :sparse
        @test sampling_design(ir) == :sparse
        masked = Matrix{Union{Missing,Float64}}(ones(2, 22))
        masked[1, 1:3] .= missing
        masked[2, 1] = missing
        @test sampling_design(GridFunctionalData(masked, collect(1:22))) == :dense_irregular
        masked[2, 2] = missing
        @test sampling_design(GridFunctionalData(masked, collect(1:22))) == :sparse
        for threshold in (0, -1, Inf, NaN)
            @test_throws ArgumentError sampling_design(ir; threshold)
        end
    end
end

@testset "Multivariate functional data" begin
    grid = [0.0, 0.5, 1.0]
    a = GridFunctionalData([1 2 3; 4 5 6], grid)
    b = GridFunctionalData([7 8 9; 10 11 12], grid)
    fd = @inferred MultivariateFunctionalData([a, b])
    @test nobs(fd) == length(fd) == 2
    @test ncomponents(fd) == 2
    @test component(fd, 1).values == a.values
    @test domain(fd) == [(0.0, 1.0), (0.0, 1.0)]
    @test nobs_per_curve(fd) == [[3, 3], [3, 3]]
    @test sampling_design(fd) == [:regular, :regular]
    array = @inferred to_array(fd)
    @test size(array) == (2, 2, 3)
    @test array[:, 1, :] == a.values
    @test array[:, 2, :] == b.values
    array[1] = -1
    a.values[1] = -1
    @test to_array(fd)[1] == 1
    @test (@inferred evaluate(fd, grid, NoInterpolation())) == to_array(fd)
    @test (@inferred evaluate(fd, [0.25], PiecewiseLinear()))[:, 1, :] == reshape([1.5, 4.5], 2, 1)
    @test (@inferred fd[1])[1] == component(fd, 1)[1]
    @test (@inferred fd[Bool[0, 1]]) isa MultivariateFunctionalData
    @test (@inferred fd[[2, 1]]) isa MultivariateFunctionalData
    @test (@inferred fd[1:1]) isa MultivariateFunctionalData
    @test length(collect(fd)) == 2
    @test nobs(vcat(fd[1:1], fd[2:2])) == 2
    @test !occursin('\n', sprint(show, fd))
    @test_throws ArgumentError evaluate(fd, grid)
    @test_throws ArgumentError evaluate(fd, [-0.1], PiecewiseLinear())
    @test_throws ArgumentError MultivariateFunctionalData(GridFunctionalData{Float64}[])
    @test_throws DimensionMismatch MultivariateFunctionalData([a, b[1:1]])
    @test_throws BoundsError component(fd, 3)
    ir = IrregularFunctionalData([[0.1, 0.9], [0.25, 0.75]], [[1, 2], [3, 4]]; domain=(0, 1))
    bf = BasisFunctionalData([1 2; 3 4], MonomialBasis((0, 2), 1))
    mixed = @inferred MultivariateFunctionalData(AbstractFunctionalData{Float64}[a, ir, bf])
    @test sampling_design(mixed; threshold=3) == [:regular, :sparse, :regular]
    @test nobs_per_curve(mixed) == [[3, 3], [2, 2], [2, 2]]
    @test mixed[1][3] isa BasisFunctionalData
    @test_throws ArgumentError to_array(mixed)
    @test_throws ArgumentError evaluate(mixed, [1.5], PiecewiseLinear())
    points = [[0.0, 0.5, 1.0], [0.25, 0.5, 0.75], [0.0, 1.0, 2.0]]
    outputs = @inferred evaluate(mixed, points, PiecewiseLinear())
    @test outputs[1] == evaluate(a, points[1], PiecewiseLinear())
    @test outputs[2] == evaluate(ir, points[2], PiecewiseLinear())
    @test outputs[3] == evaluate(bf, points[3])
    @test_throws DimensionMismatch evaluate(mixed, points[1:2], PiecewiseLinear())
    @test_throws ArgumentError evaluate(mixed, [[0.0], [1.1], [0.0]], PiecewiseLinear())
    basesonly = MultivariateFunctionalData([bf, bf])
    @test (@inferred evaluate(basesonly, [0.5]))[:, 1, :] == evaluate(bf, [0.5])
    @test (@inferred evaluate(basesonly, [[0.0], [2.0]]))[2] == evaluate(bf, [2.0])
    @test_throws ArgumentError to_array(MultivariateFunctionalData([a, GridFunctionalData([missing 1 2; 3 4 5], grid)]))
    @test_throws ArgumentError to_array(MultivariateFunctionalData([a, GridFunctionalData([1 2; 3 4], [0, 1])]))
    @test_throws DimensionMismatch vcat(fd, MultivariateFunctionalData([a]))
    @test_throws ArgumentError vcat(fd, MultivariateFunctionalData([to_irregular(a), to_irregular(b)]))
    f32 = GridFunctionalData(Float32[1 2; 3 4], Float32[0, 1])
    @test (@inferred MultivariateFunctionalData([f32, f32])) isa MultivariateFunctionalData{Float32}
    @test eltype(MultivariateFunctionalData(AbstractFunctionalData[f32, a])) == Float64
end
