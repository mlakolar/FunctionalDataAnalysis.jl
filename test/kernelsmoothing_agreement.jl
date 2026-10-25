import KernelSmoothing
import StatsAPI
const FDA = FunctionalDataAnalysis

# The smoothers FDA used before it switched to KernelSmoothing's localpolynomial,
# smoothermatrix and bivariate smoother (FDA 539b944). The new results must agree
# with these to 1e-10 for the same kernel and bandwidth.
module OldSmoothers
using LinearAlgebra
import KernelSmoothing

predict1d(t, y, w, points; bandwidth, kernel, degree, derivative=0) = KernelSmoothing.coef(
    KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, t, y; bandwidth, kernel,
        degree, weights=w, points, derivative, minobs=0))

function local_linear2d(x, y, z, weights, s, t, h, kernel)
    h1, h2 = h isa Real ? (h, h) : h
    A, b = zeros(3, 3), zeros(3)
    for i in eachindex(z)
        u, v = (x[i] - s)/h1, (y[i] - t)/h2
        w = weights[i] * kernel(u) * kernel(v)
        w == 0 && continue
        row = (1.0, u, v)
        for j in 1:3
            b[j] += w * row[j] * z[i]
            for k in 1:3
                A[j, k] += w * row[j] * row[k]
            end
        end
    end
    factor = cholesky(Symmetric(A); check=false)
    isposdef(factor) && minimum(abs2, diag(factor.U)) > 100eps() * maximum(abs, A) || return NaN
    (factor \ b)[1]
end

surface(pool, g1, g2, h, kernel) =
    [local_linear2d(pool.x, pool.y, pool.z, pool.weights, s, t, h, kernel) for s in g1, t in g2]

function cv_errors(pool, assignments, folds, hs, kernel)
    map(hs) do h
        score = 0.0
        for fold in 1:folds
            held = assignments[pool.subjects] .== fold
            train = .!held
            x, y, z, w = pool.x[train], pool.y[train], pool.z[train], pool.weights[train]
            for i in findall(held)
                predicted = local_linear2d(x, y, z, w, pool.x[i], pool.y[i], h, kernel)
                score += pool.weights[i] * abs2(pool.z[i] - predicted) + pool.scatter[i]
            end
        end
        score / sum(pool.weights)
    end
end
end

# Sparse random times, a common grid (`dense`), or one random time in each of
# `stratified` equal cells, which keeps every window of width 2/stratified nonempty
# whatever the random stream of the Julia version.
function agreement_data(rng, n; dense=false, stratified=0)
    times = [dense ? collect(range(0, 1; length=25)) :
        stratified > 0 ? ((0:stratified-1) .+ rand(rng, stratified)) ./ stratified :
        sort(rand(rng, rand(rng, 3:7))) for _ in 1:n]
    values = [sin.(2π .* t) .+ randn(rng) .* cos.(π .* t) .+ 0.2 .* randn(rng, length(t)) for t in times]
    IrregularFunctionalData(times, values)
end

const AGREEMENT_KERNELS = (KernelSmoothing.EpanechnikovKernel(), KernelSmoothing.GaussianKernel())

@testset "KernelSmoothing smoothers agree with FDA's former smoothers" begin
    rng = MersenneTwister(20261025)
    fd = agreement_data(rng, 40)
    grid = collect(range(0.05, 0.95; length=21))

    @testset "pooled mean and derivatives" begin
        for kernel in AGREEMENT_KERNELS, weighting in (:obs, :subj), derivative in 0:2
            h = derivative == 0 ? 0.12 : 0.25
            m = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=h, kernel, derivative, weighting)
            t, y, w, _ = FDA._mean_pool(fd, weighting)
            old = OldSmoothers.predict1d(t, y, w, grid; bandwidth=h, kernel,
                degree=derivative + 1, derivative)
            @test m.values ≈ old rtol=0 atol=1e-10
        end
    end

    @testset "curve smoothing" begin
        dense = agreement_data(rng, 6; dense=true)
        g = dense[1][1]
        observed = trues(6, length(g))
        observed[2, [3, 9, 10]] .= false
        values = reduce(vcat, [dense[i][2]' for i in 1:6])
        for data in (GridFunctionalData(Matrix(values), g),
                GridFunctionalData(Union{Missing,Float64}[observed[i, j] ? values[i, j] : missing
                    for i in 1:6, j in eachindex(g)], g), agreement_data(rng, 6; stratified=15))
            for kernel in AGREEMENT_KERNELS, derivative in 0:1
                h = derivative == 0 ? 0.2 : 0.4
                smoothed = smooth_curves(data, LocalPolynomial(; bandwidth=h, derivative, kernel); grid)
                for i in 1:nobs(data)
                    t, y = data[i]
                    inside = findall(x -> first(t) <= x <= last(t), grid)
                    old = OldSmoothers.predict1d(t, y, ones(length(t)), grid[inside];
                        bandwidth=h, kernel, degree=derivative + 1, derivative)
                    times, estimates = smoothed[i]
                    @test times == grid[inside]
                    @test estimates ≈ old rtol=0 atol=1e-10
                end
            end
        end
    end

    @testset "derivative columns" begin
        g = collect(range(0, 1; length=41))
        values = [sin(k*π*x) for x in g, k in 1:3]
        for kernel in AGREEMENT_KERNELS, order in 1:2
            method = LocalPolynomial(; bandwidth=0.2, derivative=order, degree=order + 1, kernel)
            new = FDA._differentiate_columns(g, values, method)
            for j in 1:3
                old = OldSmoothers.predict1d(g, values[:, j], ones(length(g)), g;
                    bandwidth=0.2, kernel, degree=order + 1, derivative=order)
                @test new[:, j] ≈ old rtol=0 atol=1e-10
            end
        end
    end

    @testset "covariance surface" begin
        m = StatsAPI.fit(SmoothedMean, fd; grid=collect(range(0, 1; length=51)), bandwidth=0.15)
        for kernel in AGREEMENT_KERNELS, weighting in (:obs, :subj), binning in (nothing, 15)
            c = StatsAPI.fit(SmoothedCovariance, fd, m; grid, bandwidth=0.2, kernel, weighting, binning)
            pool = FDA._covariance_pool(fd, m, weighting, binning)
            old = OldSmoothers.surface(pool, grid, grid, 0.2, kernel)
            @test c.surface ≈ (old + old')/2 rtol=0 atol=1e-10
        end
    end

    @testset "leave-curves-out covariance CV" begin
        m = StatsAPI.fit(SmoothedMean, fd; grid=collect(range(0, 1; length=51)), bandwidth=0.15)
        hs = [0.35, 0.2, 0.25, 0.5]
        for kernel in AGREEMENT_KERNELS, binning in (nothing, 15),
                (folds, cvrng) in ((nobs(fd), nothing), (5, MersenneTwister(7)))
            c = StatsAPI.fit(SmoothedCovariance, fd, m; grid, binning, kernel,
                bandwidth=CurveCV(hs; folds, rng=cvrng))
            pool = FDA._covariance_pool(fd, m, :obs, binning)
            old = OldSmoothers.cv_errors(pool, c.selection.assignments, folds, hs, kernel)
            @test c.selection.errors ≈ old rtol=1e-10 atol=0
            @test c.bandwidth == hs[argmin(old)]
        end
    end

    @testset "cross-covariance with a bandwidth pair" begin
        fd2 = IrregularFunctionalData([fd[i][1] for i in 1:nobs(fd)], [fd[i][2] .^ 2 for i in 1:nobs(fd)])
        m1 = StatsAPI.fit(SmoothedMean, fd; grid=collect(range(0, 1; length=51)), bandwidth=0.15)
        m2 = StatsAPI.fit(SmoothedMean, fd2; grid=collect(range(0, 1; length=51)), bandwidth=0.15)
        g1, g2 = collect(range(0, 1; length=21)), collect(range(0, 1; length=13))
        for kernel in AGREEMENT_KERNELS, h in (0.25, (0.2, 0.3))
            c = StatsAPI.fit(SmoothedCrossCovariance, fd, fd2, m1, m2; grid1=g1, grid2=g2,
                bandwidth=h, kernel)
            pool = FDA._cross_pool(fd, fd2, m1, m2, :obs, false, nothing)
            @test c.surface ≈ OldSmoothers.surface(pool, g1, g2, h, kernel) rtol=0 atol=1e-10
        end
    end
end
