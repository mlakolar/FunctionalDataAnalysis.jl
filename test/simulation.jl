function check_eigenfunction_generator(generator::F) where {F}
    domain = (-1.0, 2.0)
    points = collect(range(domain...; length=100001))
    weights = quadrature_weights(points)
    functions = @inferred generator(domain, 6)
    X = [f(t) for t in points, f in functions]
    @test norm(transpose(X) * (weights .* X) - I) / sqrt(6) < 1e-6
    @test (@inferred first(functions)(0.5)) isa Float64
    @test isempty(generator(domain, 0))
    @test_throws ArgumentError generator(domain, -1)
    @test_throws ArgumentError generator((1, 0), 2)
    @test_throws ArgumentError functions[1](-1.1)
    @test_throws ArgumentError functions[1](NaN)
    f32 = @inferred generator((0.0f0, 1.0f0), 3)
    @test (@inferred f32[1](0.5f0)) isa Float32
end

@testset "Simulation" begin
    @testset "Orthonormal eigenfunctions" begin
        for generator in (fourier_eigenfunctions, legendre_eigenfunctions)
            check_eigenfunction_generator(generator)
        end
    end

    @testset "Sampling and reproducibility" begin
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        regular = @inferred RegularSampling(grid)
        grid[1] = -1
        @test regular.grid[1] == 0
        sparse = @inferred SparseSampling((0, 1), 2:5)
        functions = fourier_eigenfunctions((0, 1), 3)
        for sampling in (regular, sparse)
            a = simulate_kl(Random.Xoshiro(42), 20; mean=t -> 2t + 1,
                eigenfunctions=functions, eigenvalues=[1.0, 0.5, 0.25], noise_sd=0.1, sampling)
            b = simulate_kl(Random.Xoshiro(42), 20; mean=t -> 2t + 1,
                eigenfunctions=functions, eigenvalues=[1.0, 0.5, 0.25], noise_sd=0.1, sampling)
            @test a.scores == b.scores
            @test a.data.values == b.data.values
            @test a.truth.values == b.truth.values
            @test a.data isa typeof(a.truth)
            noisefree = simulate_kl(Random.Xoshiro(42), 20; mean=t -> 2t + 1,
                eigenfunctions=functions, eigenvalues=[1.0, 0.5, 0.25], noise_sd=0, sampling)
            @test noisefree.data.values == noisefree.truth.values
            @test noisefree.scores == a.scores
            @test noisefree.data.values !== noisefree.truth.values
            @test size(a.scores) == (20, 3)
            @test nobs(a.data) == 20
            if a.data isa IrregularFunctionalData
                @test a.data.times == b.data.times == a.truth.times == noisefree.data.times
                @test all(n -> 2 <= n <= 5, nobs_per_curve(a.data))
                @test domain(a.data) == (0.0, 1.0)
                @test all(i -> issorted(a.data[i][1]), 1:nobs(a.data))
            else
                @test a.data.grid == b.data.grid == a.truth.grid
            end
        end
        f32 = simulate_kl(Random.Xoshiro(1), 2; mean=t -> zero(t),
            eigenfunctions=fourier_eigenfunctions((0.0f0, 1.0f0), 1),
            eigenvalues=Float32[1], sampling=RegularSampling(Float32[0, 1]))
        @test f32.data isa GridFunctionalData{Float32}
        @test f32.scores isa Matrix{Float32}
        meanonly = simulate_kl(Random.Xoshiro(1), 2; mean=t -> 2t,
            eigenfunctions=Function[], eigenvalues=Float64[], sampling=regular)
        @test size(meanonly.scores) == (2, 0)
        @test meanonly.data.values[1, :] == 2regular.grid
        stepped = simulate_kl(Random.Xoshiro(3), 30; mean=t -> 0.0,
            eigenfunctions=Function[], eigenvalues=Float64[], sampling=SparseSampling((0, 1), 2:2:6))
        @test all(n -> n in (2, 4, 6), nobs_per_curve(stepped.data))
    end

    @testset "Monte Carlo validation (six standard errors)" begin
        n = 20000
        functions = legendre_eigenfunctions((0, 1), 3)
        eigenvalues = [1.0, 0.5, 0.2]
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = simulate_kl(Random.Xoshiro(20261024), n; mean=t -> 1 + 2t,
            eigenfunctions=functions, eigenvalues, noise_sd=0.3, sampling=RegularSampling(grid))
        for k in eachindex(eigenvalues)
            scores = result.scores[:, k]
            average = sum(scores) / n
            variance = sum(abs2, scores .- average) / (n - 1)
            @test abs(average) < 6sqrt(eigenvalues[k] / n)
            @test abs(variance - eigenvalues[k]) < 6sqrt(2 / (n - 1)) * eigenvalues[k]
        end
        for (j, t) in enumerate(grid)
            average = sum(result.data.values[:, j]) / n
            variance = sum(eigenvalues[k] * functions[k](t)^2 for k in eachindex(eigenvalues)) + 0.3^2
            @test abs(average - (1 + 2t)) < 6sqrt(variance / n)
        end
        residuals = vec(result.data.values - result.truth.values)
        residualmean = sum(residuals) / length(residuals)
        residualvariance = sum(abs2, residuals .- residualmean) / (length(residuals) - 1)
        @test abs(residualvariance - 0.3^2) < 6sqrt(2 / (length(residuals) - 1)) * 0.3^2
    end

    @testset "Errors" begin
        @test_throws ArgumentError RegularSampling([0.0])
        @test_throws ArgumentError RegularSampling([0.0, 0.0])
        @test_throws ArgumentError RegularSampling([0.0, Inf])
        @test_throws ArgumentError SparseSampling((1, 0), 2:5)
        @test_throws ArgumentError SparseSampling((0, 1), 0:2)
        @test_throws ArgumentError SparseSampling((0, 1), 2:1)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 1; mean=t -> 0.0f0,
            eigenfunctions=Function[], eigenvalues=Float32[],
            sampling=SparseSampling((1.0f0, nextfloat(1.0f0)), 3:3))
        regular = RegularSampling([0.0, 1.0])
        functions = fourier_eigenfunctions((0, 1), 1)
        for eigenvalues in ([-1.0], [NaN], [Inf])
            @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0, eigenfunctions=functions, eigenvalues, sampling=regular)
        end
        for noise_sd in (-1, NaN, Inf)
            @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0, eigenfunctions=functions, eigenvalues=[1.0], noise_sd, sampling=regular)
        end
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 0; mean=t -> 0.0, eigenfunctions=functions, eigenvalues=[1.0], sampling=regular)
        @test_throws DimensionMismatch simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0, eigenfunctions=functions, eigenvalues=[1.0, 2.0], sampling=regular)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> Inf, eigenfunctions=functions, eigenvalues=[1.0], sampling=regular)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0, eigenfunctions=[t -> NaN], eigenvalues=[1.0], sampling=regular)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=0.0, eigenfunctions=functions, eigenvalues=[1.0], sampling=regular)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0, eigenfunctions=[1.0], eigenvalues=[1.0], sampling=regular)
        sparse = SparseSampling((0, 1), 2:5)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=0.0,
            eigenfunctions=functions, eigenvalues=[1.0], sampling=sparse)
        @test_throws ArgumentError simulate_kl(Random.Xoshiro(1), 2; mean=t -> 0.0,
            eigenfunctions=[1.0], eigenvalues=[1.0], sampling=sparse)
    end
end
