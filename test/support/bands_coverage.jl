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
