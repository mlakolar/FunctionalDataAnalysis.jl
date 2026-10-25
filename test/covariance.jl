import KernelSmoothing
import StatsAPI
using DelimitedFiles

@testset "Covariance with an FPCA grid mean" begin
    grid = [0., .3, .7, 1.]
    fd = GridFunctionalData([i*(1+t) for i in 1:4, t in grid], grid)
    smoothmean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=2.)
    gridmean = GridFunctionalData(reshape(copy(smoothmean.values),1,:),grid)
    fromgrid = @inferred StatsAPI.fit(SmoothedCovariance,fd,gridmean; grid,bandwidth=2.)
    fromsmooth = StatsAPI.fit(SmoothedCovariance,fd,smoothmean; grid,bandwidth=2.)
    @test fromgrid.surface == fromsmooth.surface
    @test fromgrid.noise_variance == fromsmooth.noise_variance
    for badmean in (GridFunctionalData(zeros(2,4),grid),
            GridFunctionalData([0. missing 0. 0.],grid),
            GridFunctionalData(zeros(1,2),[.1,.9]))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance,fd,badmean; grid,bandwidth=2.)
    end
end

@testset "Noise methods: fdapace 0.6.0 PC_CovE" begin
    input = readdlm(joinpath(@__DIR__, "data", "m2_input.csv"), ','; comments=true)
    times = [Float64.(input[input[:,1] .== i,2]) for i in 1:4]
    ys = [Float64.(input[input[:,1] .== i,3]) for i in 1:4]
    fd = IrregularFunctionalData(times, ys)
    golden = readdlm(joinpath(@__DIR__, "data", "m5_fdapace_sigma2.csv"), ',', Any; comments=true)
    grids = (aligned=collect(0.:.125:1.), unaligned=[0., .18, .31, .56, .69, .84, 1.],
        extended=[-.1, .1, .35, .55, .7, .9, 1.1])
    for weighting in (:obs, :subj)
        m = StatsAPI.fit(SmoothedMean, fd;
            grid=sort(unique(vcat(times...))), bandwidth=.65, weighting)
        for (kernel_name, kernel) in (("epan", KernelSmoothing.EpanechnikovKernel()),
                ("gauss", KernelSmoothing.GaussianKernel())), h in (.35, .65),
                grid_name in keys(grids), interval in (:middle_half, :full)
            grid = grids[grid_name]
            c = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; grid,
                bandwidth=h, kernel, weighting, noise_method=:rotated, noise_interval=interval)
            rows = (golden[:,1] .== string(weighting)) .& (golden[:,2] .== kernel_name) .&
                (golden[:,3] .== h) .& (golden[:,4] .== string(grid_name)) .&
                (golden[:,5] .== string(interval))
            expected_raw = only(Float64.(golden[rows .& (golden[:,6] .== "raw_variance"),9]))
            @test abs(c.noise_details.raw_variance - expected_raw)/abs(expected_raw) < 1e-8
            @test c.noise_variance == max(0, c.noise_details.raw_variance)
            @test c.noise_details.truncated == (expected_raw < 0)
            for (quantity, actual) in (("latent_diagonal", c.noise_details.latent_diagonal),
                    ("observed_diagonal", c.noise_details.diagonal[indexin(c.noise_details.integration_grid, grid)]))
                selected = rows .& (golden[:,6] .== quantity)
                expected = Float64.(golden[selected,9])
                @test norm(actual - expected)/norm(expected) < 1e-8
                @test c.noise_details.integration_grid == Float64.(golden[selected,8])
            end
            @test c.noise_details.domain == (0., 1.)
            @test c.noise_details.nominal_length == (interval == :full ? 1. : .5)
            @test c.noise_details.boundary_rule == :retained_grid
            @test c.noise_details.rotation.applied
            @test c.noise_details.rotation.bandwidth == h
            @test c.noise_details.noise_method == :rotated
            @test c.noise_details.noise_interval == interval
            @test occursin("noise_method = :rotated", sprint(show, c))
        end
        grid = grids.unaligned
        default = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65, weighting)
        explicit = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65, weighting,
            noise_method=:rotated, noise_interval=:full)
        full = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65, weighting,
            noise_method=:surface_diagonal, noise_interval=:full)
        rotated = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65, weighting,
            noise_method=:rotated, diagonal_bandwidth=.8)
        @test default.noise_variance == explicit.noise_variance
        @test default.noise_details == explicit.noise_details
        @test default.noise_details.noise_method == :rotated
        @test default.noise_details.noise_interval == :full
        @test default.surface == full.surface == rotated.surface
        @test full.noise_details.integration_grid == grid
        @test full.noise_details.bounds == (0., 1.)
        @test !full.noise_details.rotation.applied
        @test full.noise_details.raw_variance ≈ dot(quadrature_weights(grid),
            full.noise_details.diagonal - diag(full.surface))
        @test rotated.diagonal_bandwidth == .8
        @test rotated.noise_details.rotation.bandwidth == .65
        # Snapping and within-curve aggregation preserve weighted rotated fits.
        raw = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, nothing)
        snapped = merge(raw, (; x=FunctionalDataAnalysis._covariance_snap(raw.x, 5),
            y=FunctionalDataAnalysis._covariance_snap(raw.y, 5)))
        grouped = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, 5)
        ungrouped = FunctionalDataAnalysis._rotated_covariance_diagonal(snapped, grid, 1.2, full.kernel)
        binned = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=1.2,
            weighting, binning=5, noise_method=:rotated, noise_interval=:full)
        @test binned.noise_details.latent_diagonal ≈ ungrouped
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; grid, weighting,
            bandwidth=CurveCV([.65, 1.2]; folds=4), noise_method=:rotated)
        @test selected.noise_details.rotation.bandwidth == selected.diagonal_bandwidth == selected.bandwidth
    end
    m = StatsAPI.fit(SmoothedMean, fd; grid=grids.aligned, bandwidth=.65)
    for bad in (:projected, :unknown, nothing, missing, "rotated")
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid=grids.aligned, bandwidth=.65, noise_method=bad)
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid=grids.aligned, bandwidth=.65, noise_interval=bad)
    end
    @test_throws r"at least two work-grid points" StatsAPI.fit(SmoothedCovariance, fd, m;
        grid=[0., .5, 1.], bandwidth=.65, noise_interval=:middle_half)
    # A compact kernel and three observation times cannot identify the rotated
    # quadratic at this bandwidth. The default must report this, without fallback.
    smallgrid = [0., .5, 1.]
    small = GridFunctionalData([1. 2 3; -1 -2 -3], smallgrid)
    smallmean = StatsAPI.fit(SmoothedMean, small; grid=smallgrid, bandwidth=1.05)
    @test_throws r"singular or undefined rotated diagonal fit .*use a larger bandwidth" StatsAPI.fit(
        SmoothedCovariance, small, smallmean; grid=smallgrid, bandwidth=1.05)
    surface = StatsAPI.fit(SmoothedCovariance, small, smallmean;
        grid=smallgrid, bandwidth=1.05, noise_method=:surface_diagonal)
    @test all(isfinite, surface.surface)
    @test isfinite(surface.noise_variance)
    @test surface.noise_details.noise_method == :surface_diagonal
    wider = StatsAPI.fit(SmoothedCovariance, small, smallmean;
        grid=smallgrid, bandwidth=2.)
    @test isfinite(wider.noise_variance)
    @test wider.noise_details.noise_method == :rotated
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
        grid=[-2., -1.], bandwidth=4., noise_method=:rotated, noise_interval=:full)
    # Analytic rotated polynomial: intercept at (t,t) is 2 + 4sqrt(2)t.
    # Unequal positive pair weights must not change exact polynomial recovery.
    s = repeat([0., .2, .6, 1.]; inner=4)
    t = repeat([0., .2, .6, 1.]; outer=4)
    u, v = (s-t)/sqrt(2), (s+t)/sqrt(2)
    pool = (; x=s, y=t, z=2 .+ 3u.^2 .+ 4v, weights=collect(1.:16.))
    points = [0., .17, .61, 1.]
    actual = @inferred FunctionalDataAnalysis._rotated_covariance_diagonal(
        pool, points, 2., KernelSmoothing.EpanechnikovKernel())
    @test actual ≈ 2 .+ 4sqrt(2).*points
    # The analytic diagonal plus constant noise must integrate to that noise.
    # Test the helper directly: runtime symbols must not leave Any fields that
    # break inference in covariance, PACE, or multivariate fits on Julia 1.10.
    for T in (Float32, Float64)
        workgrid = T[0, .25, .5, .75, 1]
        typed_pool = (; x=T.(s), y=T.(t), z=T.(pool.z), weights=T.(pool.weights),
            original_times=T.(s))
        surface = T[2 + 2sqrt(2)*(s+t) for s in workgrid, t in workgrid]
        diagonal = diag(surface) .+ T(.5)
        for method in (:surface_diagonal, :rotated), interval in (:middle_half, :full)
            details = @inferred FunctionalDataAnalysis._covariance_noise_integral(
                typed_pool, workgrid, surface, diagonal, T(2),
                KernelSmoothing.EpanechnikovKernel(), method, interval)
            @test details.raw_variance ≈ T(.5)
            @test details.raw_variance isa T
            @test details.latent_diagonal isa Vector{T}
            @test details.integration_grid isa Vector{T}
        end
    end
    singular = (; x=[0., 1.], y=[1., 0.], z=[1., 1.], weights=ones(2))
    @test_throws ArgumentError FunctionalDataAnalysis._rotated_covariance_diagonal(
        singular, [.5], 2., KernelSmoothing.GaussianKernel())
    grid32 = Float32[0, .25, .5, .75, 1]
    fd32 = GridFunctionalData(Float32[1 2 1 3 1; -1 -2 -1 -3 -1], grid32)
    m32 = StatsAPI.fit(SmoothedMean, fd32; grid=grid32, bandwidth=2f0)
    c32 = @inferred StatsAPI.fit(SmoothedCovariance, fd32, m32; grid=grid32,
        bandwidth=2f0, noise_method=:rotated, noise_interval=:full)
    @test eltype(c32.noise_details.latent_diagonal) == Float32
    @test c32.noise_variance isa Float32
end

@testset "Pooled covariance" begin
    input = readdlm(joinpath(@__DIR__, "data", "m2_input.csv"), ','; comments=true)
    times = [Float64.(input[input[:,1] .== i,2]) for i in 1:4]
    ys = [Float64.(input[input[:,1] .== i,3]) for i in 1:4]
    fd = IrregularFunctionalData(times, ys)
    golden = readdlm(joinpath(@__DIR__, "data", "m2_fdapace_cov.csv"), ',', Any; comments=true)
    grid = collect(0.0:.125:1.0)
    for weighting in (:obs, :subj)
        # The fixture centers at direct mean estimates: store the mean at every
        # observation time so interpolation is exact there.
        m = StatsAPI.fit(SmoothedMean, fd; grid=sort(unique(vcat(times...))), bandwidth=.65, weighting)
        c = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=.65, weighting)
        rows = golden[:,1] .== string(weighting)
        expected = reshape(Float64.(golden[rows,5]), length(grid), length(grid))
        @test norm(c.surface - expected)/norm(expected) < 1e-8
        @test c.surface == transpose(c.surface)
        @test (@inferred noise_variance(c)) == c.noise_variance
        @test c.weights == quadrature_weights(grid)
        @test c.grid !== grid
        @test (@inferred evaluate(c, .125, .25)) == c.surface[2,3]
        @test evaluate(c, .0625, .1875) ≈ sum(c.surface[1:2,2:3])/4
        @test evaluate(c, 1.0, 1.0) == c.surface[end,end]
        @test c.diagonal_bandwidth == c.noise_details.bandwidth == .65
        @test c.noise_details.selection === nothing
        @test c.noise_details.folds == 0
        @test occursin("diagonal_bandwidth=0.65", sprint(show, c))
        @test occursin("noise_details", sprint(show, c))
        for q in ((-1., 0.), (0., 2.), (NaN, 0.), (0., Inf))
            @test_throws ArgumentError evaluate(c, q...)
        end
        pool = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, nothing)
        for i in 1:4
            ix = pool.subjects .== i
            n = length(times[i])
            @test all(pool.weights[ix] .≈ (weighting == :obs ? 1.0 : 1/(n*(n-1))))
            @test sum(pool.weights[ix]) ≈ (weighting == :obs ? n*(n-1) : 1)
        end
        cv = CurveCV([.8, 1.2]; folds=4)
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=cv, weighting)
        # Independent weighted-design QR prediction checks both pair weights and
        # leave-curve-out scoring (rather than reusing the FDA smoother).
        errors = map(cv.bandwidths) do h
            score = 0.0
            for i in eachindex(pool.z)
                train = pool.subjects .!= pool.subjects[i]
                u, v = (pool.x[train] .- pool.x[i])/h, (pool.y[train] .- pool.y[i])/h
                w = sqrt.(pool.weights[train] .* KernelSmoothing.EpanechnikovKernel().(u) .* KernelSmoothing.EpanechnikovKernel().(v))
                X = hcat(ones(length(u)), u, v)
                beta = (X .* w) \ (pool.z[train] .* w)
                score += pool.weights[i] * abs2(pool.z[i] - beta[1])
            end
            score/sum(pool.weights)
        end
        @test selected.selection.errors ≈ errors
        @test selected.bandwidth == cv.bandwidths[argmin(errors)]
        @test selected.selection.assignments == 1:4
    end
    m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=.65)
    default = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65)
    explicit = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=.65, diagonal_bandwidth=.65)
    @test default.diagonal_bandwidth == default.noise_details.bandwidth == .65
    @test default.noise_details.selection === nothing
    @test default.noise_details.diagonal_source == :covariance
    @test explicit.noise_details.diagonal_source == :user
    @test default.noise_variance == explicit.noise_variance
    @test default.surface == explicit.surface
    cvdefault = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=CurveCV([.5, .65, 1.]; folds=4))
    @test cvdefault.diagonal_bandwidth == cvdefault.bandwidth
    for h in (0., -1., Inf, NaN, :bad)
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=h)
    end
    for selector in (CurveCV([1.]; folds=5), CurveCV([1.]; folds=2), CurveCV([.001]; folds=4))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=selector)
    end
    # Independent weighted QR fits to raw squares. Test original held-out times
    # even when training times are snapped, and both observation weight rules.
    for weighting in (:obs, :subj), binning in (nothing, 5),
            selector in (CurveCV([.8, 1.2]; folds=4),
                         CurveCV([.8, 1.2]; folds=2, rng=MersenneTwister(17)))
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=selector, weighting, binning)
        reference = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=selector, weighting)
        assignments = reference.selection.assignments
        original = vcat(times...)
        squares = (vcat(ys...) - evaluate(m, original)).^2
        subjects = vcat([fill(i, length(times[i])) for i in 1:4]...)
        weights = weighting == :obs ? ones(length(original)) : 1 ./ length.(times)[subjects]
        training_times = binning === nothing ? original : round.(original .* 4, RoundNearestTiesUp) ./ 4
        errors = map(selector.bandwidths) do h
            score = 0.0
            for i in eachindex(original)
                train = assignments[subjects] .!= assignments[subjects[i]]
                u = (training_times[train] .- original[i])/h
                w = sqrt.(weights[train] .* KernelSmoothing.EpanechnikovKernel().(u))
                beta = (hcat(ones(length(u)), u) .* w) \ (squares[train] .* w)
                score += weights[i]*abs2(squares[i]-beta[1])
            end
            score/sum(weights)
        end
        @test selected.noise_details.selection.bandwidths == selector.bandwidths
        @test selected.noise_details.selection.bandwidths !== selector.bandwidths
        @test selected.noise_details.selection.assignments == assignments
        @test selected.noise_details.selection.errors ≈ errors
        @test selected.diagonal_bandwidth == selector.bandwidths[argmin(errors)]
        @test selected.noise_details.bandwidth == selected.diagonal_bandwidth
        @test selected.noise_details.folds == selector.folds
        @test selected.surface == StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, bandwidth=.65, diagonal_bandwidth=.65, weighting, binning).surface
    end
    binned = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=1.2, binning=5)
    @test binned.binning == 5
    pool = FunctionalDataAnalysis._covariance_pool(fd, m, :subj, 5)
    @test all(t -> t in 0:.25:1, pool.times)
    @test length(pool.z) < sum(n*(n-1) for n in length.(times))
    # Expand snapped raw pairs independently. Aggregation must preserve both
    # fixed-bandwidth estimates and the original raw-product CV objective.
    for weighting in (:obs, :subj)
        raw = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, nothing)
        snap(t) = round(t*4, RoundNearestTiesUp)/4
        x, y = snap.(raw.x), snap.(raw.y)
        grouped = FunctionalDataAnalysis._covariance_pool(fd, m, weighting, 5)
        @test sum(grouped.weights) ≈ sum(raw.weights)
        @test any(>(0), grouped.scatter)
        for i in eachindex(grouped.z)
            cell = (raw.subjects .== grouped.subjects[i]) .&
                (x .== grouped.x[i]) .& (y .== grouped.y[i])
            w, z = raw.weights[cell], raw.z[cell]
            @test grouped.weights[i] ≈ sum(w)
            @test grouped.z[i] ≈ dot(w, z)/sum(w)
            @test grouped.scatter[i] ≈ sum(w .* (z .- grouped.z[i]).^2) atol=1e-14
        end
        fit = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=1.2, weighting, binning=5)
        expected = KernelSmoothing.local_polynomial((x, y), raw.z, (grid, grid);
            bandwidth=1.2, kernel=fit.kernel, weights=raw.weights)
        @test fit.surface ≈ (expected + expected')/2
        selector = CurveCV([.8, 1.2]; folds=2, rng=MersenneTwister(3))
        selected = @inferred StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65,
            grid, bandwidth=selector, weighting, binning=5)
        assignments = selected.selection.assignments
        errors = map(selector.bandwidths) do h
            score = 0.0
            for i in eachindex(raw.z)
                train = assignments[raw.subjects] .!= assignments[raw.subjects[i]]
                prediction = only(KernelSmoothing.local_polynomial((x[train], y[train]),
                    raw.z[train], [(x[i], y[i])]; bandwidth=h, kernel=fit.kernel, weights=raw.weights[train]))
                score += raw.weights[i]*abs2(raw.z[i]-prediction)
            end
            score/sum(raw.weights)
        end
        @test selected.selection.errors ≈ errors
        @test selected.bandwidth == selector.bandwidths[argmin(errors)]
    end
    for kwargs in ((; bandwidth=.001), (; bandwidth=0), (; bandwidth=Inf),
            (; bandwidth=1., weighting=:bad), (; bandwidth=1., binning=1),
            (; bandwidth=1., binning=2.5), (; bandwidth=1., kernel=:bad),
            (; bandwidth=CurveCV([1.]; folds=5)), (; bandwidth=CurveCV([1.]; folds=2)))
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, kwargs...)
    end
    for g in ([0.], [1.,0.], [0.,Inf])
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid=g, bandwidth=1.)
    end
    single = IrregularFunctionalData([[0.]], [[1.]])
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, single, m; diagonal_bandwidth=.65, grid, bandwidth=1.)
    narrow_mean = StatsAPI.fit(SmoothedMean, fd; grid=[.25,.75], bandwidth=1.)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, narrow_mean; diagonal_bandwidth=.65, grid, bandwidth=1.)
    a = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=CurveCV([1.]; folds=2, rng=MersenneTwister(3)))
    b = StatsAPI.fit(SmoothedCovariance, fd, m; diagonal_bandwidth=.65, grid, bandwidth=CurveCV([1.]; folds=2, rng=MersenneTwister(3)))
    @test a.selection == b.selection

    # Two normalized Legendre eigenfunctions give a curved covariance with
    # distinct eigenvalues. Orthogonal sign columns give exactly centered,
    # uncorrelated scores and measurement errors, independent of RNG versions.
    workgrid = collect(0.0:.1:1.0)
    eigenfunctions = [t -> sqrt(3)*(2t-1), t -> sqrt(5)*(6t^2-6t+1)]
    eigenvalues = [.3, .1]
    truth = [sum(eigenvalues[k]*eigenfunctions[k](s)*eigenfunctions[k](t)
        for k in 1:2) for s in workgrid, t in workgrid]
    # Nonconstant columns of a 16 x 16 Walsh matrix: two score columns and
    # eleven noise columns. Their empirical second moments are the identity.
    signs = [isodd(count_ones(i & j)) ? -1.0 : 1.0 for i in 0:15, j in 1:13]
    scores = signs[:,1:2] .* sqrt.(eigenvalues)'
    clean = scores * [f(t) for t in workgrid, f in eigenfunctions]'
    noise = .6 .* signs[:,3:13]
    @test vec(sum(clean; dims=1)) ≈ zeros(11) atol=1e-14
    @test clean' * clean / 16 ≈ truth
    @test clean' * noise / 16 ≈ zeros(11, 11) atol=1e-14
    @test noise' * noise / 16 ≈ .36 .* Matrix{Float64}(I, 11, 11)
    fits = map((clean, clean + noise)) do values
        data = GridFunctionalData(values, workgrid)
        meanfit = StatsAPI.fit(SmoothedMean, data; grid=workgrid, bandwidth=.205)
        # This bound checks the original surface-diagonal, middle-half estimator.
        StatsAPI.fit(SmoothedCovariance, data, meanfit; diagonal_bandwidth=.205,
            grid=workgrid, bandwidth=.205, noise_method=:surface_diagonal, noise_interval=:middle_half)
    end
    noisy = GridFunctionalData(clean + noise, workgrid)
    noisy_mean = StatsAPI.fit(SmoothedMean, noisy; grid=workgrid, bandwidth=.205)
    different_diagonal = StatsAPI.fit(SmoothedCovariance, noisy, noisy_mean;
        grid=workgrid, bandwidth=.205, diagonal_bandwidth=.405,
        noise_method=:surface_diagonal, noise_interval=:middle_half)
    @test different_diagonal.surface == fits[2].surface
    @test different_diagonal.diagonal_bandwidth == .405
    @test !isapprox(different_diagonal.noise_variance, fits[2].noise_variance; atol=1e-8)
    for (covfit, noise) in zip(fits, (0., .6))
        @test maximum(abs, covfit.surface - truth) < .18
        @test abs(covfit.noise_variance - noise^2) < .07
    end
    # Only deterministic smoothing bias remains in the analytic recovery bound;
    # noise changes diagonal squares but leaves pooled off-diagonal products equal.
    @test fits[2].surface ≈ fits[1].surface atol=1e-13
    # Unequal observation counts can make the untruncated difference negative:
    # the two nonzero curves receive relatively more pair than diagonal weight.
    ts = [workgrid, workgrid, [0.,1.], [0.,1.]]
    unequal = IrregularFunctionalData(ts, [ones(11), -ones(11), zeros(2), zeros(2)])
    um = StatsAPI.fit(SmoothedMean, unequal; grid=workgrid, bandwidth=2.)
    uc = StatsAPI.fit(SmoothedCovariance, unequal, um; diagonal_bandwidth=2.,
        grid=[0.,.2,.6,1.], bandwidth=2., noise_method=:surface_diagonal, noise_interval=:middle_half)
    @test uc.noise_details.truncated
    @test uc.noise_details.raw_variance < 0
    @test uc.noise_variance == 0
    @test uc.noise_details.integration_grid == [.25,.6,.75]
    # Independent trapezoids at clipped boundaries of a nonuniform work grid.
    d = uc.noise_details.diagonal - diag(uc.surface)
    left = d[2] + (.25-.2)/(.6-.2)*(d[3]-d[2])
    right = d[3] + (.75-.6)/(1-.6)*(d[4]-d[3])
    expected = 2*((.6-.25)*(left+d[3])/2 + (.75-.6)*(d[3]+right)/2)
    @test uc.noise_details.raw_variance ≈ expected
    f32 = GridFunctionalData(Float32[1 2 3; -1 -2 -3], Float32[0,.5,1])
    m32 = StatsAPI.fit(SmoothedMean, f32; grid=Float32[0,.5,1], bandwidth=2f0)
    c32 = @inferred StatsAPI.fit(SmoothedCovariance, f32, m32; diagonal_bandwidth=2f0, grid=Float32[0,.5,1], bandwidth=2f0)
    @test eltype(c32.surface) == Float32
    # The diagonal smoother uses covariance-specific context for undefined
    # windows, while ordinary mean fits retain their default label.
    method = LocalPolynomial(; bandwidth=.01)
    for label in (nothing, "noise variance diagonal smoother")
        kwargs = label === nothing ? NamedTuple() : (; label)
        error = try
            FunctionalDataAnalysis._mean_predict([0.,1.], [1.,1.], [1.,1.], [.5],
                method; kwargs...)
        catch err
            err
        end
        expected_label = label === nothing ? "pooled mean" : label
        @test error isa ArgumentError
        @test occursin("$expected_label has an undefined estimate at point 0.5; use a larger bandwidth",
            sprint(showerror, error))
    end
    # Only two pair locations cannot identify three local-linear coefficients.
    endpoints = GridFunctionalData([1. -1.; -1. 1.], [0.,1.])
    em = StatsAPI.fit(SmoothedMean, endpoints; grid=[0.,1.], bandwidth=2.)
    @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, endpoints, em; diagonal_bandwidth=.65, grid=[0.,1.], bandwidth=2.)
end

@testset "Noise diagonal bandwidth rules and pilots" begin
    times = [collect(range(0., 1.; length=n)) for n in (21, 25, 29, 33)]
    ys = [sin.(4t) .+ .15i .* cos.(17t) for (i, t) in enumerate(times)]
    fd = IrregularFunctionalData(times, ys)
    grid = collect(0.:.1:1.)
    kernel = KernelSmoothing.GaussianKernel()
    m = StatsAPI.fit(SmoothedMean, fd; grid, kernel, bandwidth=.3)
    original = vcat(times...)
    squares = (vcat(ys...) - evaluate(m, original)).^2
    for weighting in (:obs, :subj), binning in (nothing, 9)
        x = binning === nothing ? original : round.(original .* 8, RoundNearestTiesUp) ./ 8
        w = vcat([fill(weighting == :obs ? 1. : 1/length(t), length(t)) for t in times]...)
        for rule in (KernelSmoothing.RuleOfThumb(), KernelSmoothing.IMSEOptimal())
            direct = KernelSmoothing.select_bandwidth(rule,
                KernelSmoothing.SmoothingData(x, squares; weights=w), kernel; degree=1)
            # KernelSmoothing selectors' `details` types depend on the data (and on
            # the KernelSmoothing version), so selector fits are not inferred.
            c = StatsAPI.fit(SmoothedCovariance, fd, m;
                grid, kernel, bandwidth=.5, diagonal_bandwidth=rule, weighting, binning)
            @test c.diagonal_bandwidth == c.noise_details.selection.bandwidth == direct.h
            @test c.noise_details.selection.rule == rule
            @test c.noise_details.selection.details == direct.details
            @test c.noise_details.folds == 0
            @test occursin(string(nameof(typeof(rule))), sprint(show, c))
        end
        pilot = KernelSmoothing.RuleOfThumb()
        multipliers = [1., 2., 4.]
        cv = CurveCV(; pilot, multipliers, folds=2, rng=MersenneTwister(52))
        c = StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, kernel, bandwidth=.5, diagonal_bandwidth=cv, weighting, binning)
        h = KernelSmoothing.select_bandwidth(pilot,
            KernelSmoothing.SmoothingData(x, squares; weights=w), kernel; degree=1).h
        candidates = multipliers .* h
        explicit = StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, kernel, bandwidth=.5, weighting, binning,
            diagonal_bandwidth=CurveCV(candidates; folds=2, rng=MersenneTwister(52)))
        selection = c.noise_details.selection
        @test selection.pilot == pilot
        @test selection.h_pilot == h
        @test selection.multipliers == multipliers
        @test selection.bandwidths == candidates
        @test selection.errors == explicit.noise_details.selection.errors
        @test selection.assignments == explicit.noise_details.selection.assignments
        @test c.diagonal_bandwidth == explicit.diagonal_bandwidth
        @test c.noise_variance == explicit.noise_variance
    end
    for bandwidth in (KernelSmoothing.RuleOfThumb(), CurveCV(; pilot=KernelSmoothing.RuleOfThumb()))
        err = try
            StatsAPI.fit(SmoothedCovariance, fd, m; grid, kernel, bandwidth, diagonal_bandwidth=.5)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("univariate", sprint(showerror, err))
        @test occursin("CurveCV with explicit candidates", sprint(showerror, err))
    end
    for rule in (KernelSmoothing.NormalReference(), KernelSmoothing.MSEOptimal())
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, kernel, bandwidth=.5, diagonal_bandwidth=rule)
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance, fd, m;
            grid, kernel, bandwidth=.5, diagonal_bandwidth=CurveCV(; pilot=rule, folds=4))
    end
end
