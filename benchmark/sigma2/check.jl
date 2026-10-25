# Focused harness verification, deliberately outside the package test suite.
# Run: julia --project=. benchmark/sigma2/check.jl
include("common.jl")
using .Sigma2Benchmark
using FunctionalDataAnalysis, LinearAlgebra, Random, Test
import StatsAPI
import KernelSmoothing

@testset "Benchmark exact grouping and estimators" begin
    for design in (:dense, :sparse), weighting in (:obs, :subj)
        sampling = design == :dense ? RegularSampling(collect(0.:.2:1.)) : SparseSampling((0., 1.), 3:6)
        fd = simulate_kl(MersenneTwister(53), 8; mean=t -> 1+t,
            eigenfunctions=fourier_eigenfunctions((0., 1.), 3),
            eigenvalues=[1., .5, .25], noise_sd=.3, sampling).data
        grid = collect(range(FunctionalDataAnalysis._pace_observed_range(fd)...; length=9))
        kernel = KernelSmoothing.GaussianKernel()
        mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=.3, weighting, kernel)
        raw = FunctionalDataAnalysis._covariance_pool(fd, mean, weighting, nothing)
        grouped = Sigma2Benchmark.collapse_pairs(raw)
        @test sum(grouped.weights) ≈ sum(raw.weights)
        @test sum(grouped.weights .* grouped.z) ≈ sum(raw.weights .* raw.z)
        @test sum(grouped.weights .* grouped.z.^2 + grouped.scatter) ≈ sum(raw.weights .* raw.z.^2)
        selector = CurveCV([.2, .4]; folds=2, rng=MersenneTwister(71))
        direct = FunctionalDataAnalysis._covariance_selection(selector, fd, raw, kernel)
        collapsed = Sigma2Benchmark.covariance_cv(selector, fd, raw, kernel)
        @test collapsed.errors ≈ direct.errors rtol=1e-12
        @test collapsed.assignments == direct.assignments
        t, y, w, subjects = FunctionalDataAnalysis._mean_pool(fd, weighting)
        direct_mean = FunctionalDataAnalysis._mean_selection(selector, fd,
            t, y, w, subjects, kernel, 1)
        collapsed_mean = Sigma2Benchmark.mean_cv(selector, fd, weighting, kernel)
        @test collapsed_mean.errors ≈ direct_mean.errors rtol=1e-12
        @test collapsed_mean.assignments == direct_mean.assignments
        h = .3
        surface = FunctionalDataAnalysis._local_linear_surface(grouped, grid, grid, h, kernel)
        surface = (surface + surface')/2
        diagonal = FunctionalDataAnalysis._mean_predict(raw.times, raw.squares,
            raw.diagonal_weights, grid, LocalPolynomial(; bandwidth=h, kernel))
        prepared = (; grid, kernel, mean, pool=merge(raw, grouped), surface, diagonal, h)
        for (estimator, method, interval) in ((:surface_middle_half, :surface_diagonal, :middle_half),
                (:surface_full, :surface_diagonal, :full), (:rotated_middle_half, :rotated, :middle_half),
                (:rotated_full, :rotated, :full))
            fit = StatsAPI.fit(SmoothedCovariance, fd, mean; grid, bandwidth=h, kernel,
                weighting, noise_method=method, noise_interval=interval)
            estimate = Sigma2Benchmark.estimate(prepared, estimator)
            @test estimate.raw ≈ fit.noise_details.raw_variance rtol=1e-10
            @test surface ≈ fit.surface rtol=1e-12
        end
        # Independent full weighted PSD reconstruction, without the helper's
        # eigen step, verifies the benchmark-only reference.
        root = sqrt.(quadrature_weights(grid))
        decomposition = eigen(Symmetric(surface .* (root*root')))
        keep = decomposition.values .> 100eps()*maximum(abs, decomposition.values)
        projected = ((decomposition.vectors[:,keep] .* decomposition.values[keep]') *
            decomposition.vectors[:,keep]') ./ (root*root')
        projected = (projected+projected')/2
        interior = ceil(Int, length(grid)/4):floor(Int, 3length(grid)/4)
        expected_raw = sum(diagonal[interior] - diag(projected)[interior])/length(interior)
        @test Sigma2Benchmark.estimate(prepared, :projected_fgm).raw ≈ expected_raw
    end
end
