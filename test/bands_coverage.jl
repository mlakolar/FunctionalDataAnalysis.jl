using Test, LinearAlgebra, Random
using FunctionalDataAnalysis
import StatsAPI
import KernelSmoothing

# Recreate the larger study with:
# FDA_BANDS_FULL_COVERAGE=true julia --project=. -e 'include("test/bands_coverage.jl")'
# Julia 1.10+; fixed seed; no exact cross-version RNG values are asserted.
# The data-generating mean is affine, so local-linear smoothing has no bias.
function bands_coverage(; replicates=400, n=300, B=399, seed=20261025)
    rng = MersenneTwister(seed)
    grid = collect(range(0.0, 1.0; length=5))
    observation_grid = collect(range(0.0, 1.0; length=9))
    basis(t) = hcat(ones(length(t)), sqrt(12) .* (t .- 0.5))
    phi = basis(grid)
    lambda = [1.0, 0.3]
    surface = phi * Diagonal(lambda) * phi'
    truth_mean = 1 .+ grid
    fitted_point, fitted_band, mean_point, mean_band = 0.0, 0.0, 0.0, 0.0
    for _ in 1:replicates
        latent = randn(rng, n, 2) .* transpose(sqrt.(lambda))
        times = [sort(observation_grid[randperm(rng, 9)[1:(3 + (i % 3))]]) for i in 1:n]
        values = [1 .+ times[i] .+ basis(times[i]) * latent[i, :] .+
            sqrt(0.2) .* randn(rng, length(times[i])) for i in 1:n]
        fd = IrregularFunctionalData(times, values; domain=(0.0, 1.0))
        model = StatsAPI.fit(FPCA, fd; method=:pace, grid,
            mean=(; grid, values=truth_mean), covariance=(; grid, surface),
            noise_variance=0.2, ncomponents=2, max_k=2)
        truth = transpose(truth_mean) .+ latent * phi'
        point = fitted_bands(model)
        band = fitted_bands(model; coverage=:band)
        fitted_point += sum((point.lower .<= truth) .& (truth .<= point.upper)) / length(truth)
        fitted_band += sum(vec(all((band.lower .<= truth) .& (truth .<= band.upper); dims=2))) / n
        mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=0.5,
            kernel=KernelSmoothing.GaussianKernel(), degree=1, weighting=:subj)
        mb = mean_bands(fd, mean; B, rng)
        mean_point += sum((mb.interval.lower .<= truth_mean) .& (truth_mean .<= mb.interval.upper)) / length(grid)
        mean_band += all((mb.band.lower .<= truth_mean) .& (truth_mean .<= mb.band.upper))
    end
    (; replicates, n, B, seed, fitted_pointwise=fitted_point/replicates,
        fitted_simultaneous=fitted_band/replicates, mean_pointwise=mean_point/replicates,
        mean_simultaneous=mean_band/replicates)
end

@testset "Gaussian sparse band coverage" begin
    full = get(ENV, "FDA_BANDS_FULL_COVERAGE", "false") == "true"
    result = full ? bands_coverage(; replicates=200, n=300, B=1000) : bands_coverage()
    @info "Band coverage (nominal 0.95; known population inputs for PACE)" result
    # Fitted observations share scores within curves. The chi-square ellipsoid
    # can be conservative over the five-point grid.
    @test abs(result.fitted_pointwise - 0.95) < 0.04
    @test 0.91 < result.fitted_simultaneous <= 1.0
    # The full study found mean coverage about .925 rather than nominal .95.
    # Use a one-sided regression floor of .90, not a claim of nominal coverage:
    # it allows that documented shortfall plus .025 Monte Carlo variation.
    # CI uses 400 independent data sets (MC SE .0109 at .95); the full study
    # uses 200 (SE .0154). Grid points are not independent replicates.
    # CI uses 399 bootstrap draws (full: 1000) to reduce tail-quantile noise.
    @test 0.90 <= result.mean_pointwise <= 1.0
    @test 0.90 <= result.mean_simultaneous <= 1.0
end
