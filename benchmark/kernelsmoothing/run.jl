# Timings of FDA's smoothers through the public API, so that the same script runs
# on the code before and after the switch to KernelSmoothing's smoothers.
# Run: julia --project=<FDA checkout> run.jl <label> [reps]
# Prints CSV rows label,case,threads,seconds with the minimum over reps.
using FunctionalDataAnalysis, Random
import KernelSmoothing, StatsAPI

const LABEL = ARGS[1]
const REPS = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 5

simulate(seed, n, sampling) = simulate_kl(MersenneTwister(seed), n; mean=t -> sin(2π*t),
    eigenfunctions=fourier_eigenfunctions((0., 1.), 3), eigenvalues=[1., .5, .25],
    noise_sd=.3, sampling).data

function timed(f)
    f()
    minimum(1:REPS) do _
        GC.gc()
        @elapsed f()
    end
end

const sparse = simulate(1, 500, SparseSampling((0., 1.), 2:8))
const sparse200 = simulate(2, 200, SparseSampling((0., 1.), 2:8))
const dense = simulate(3, 200, RegularSampling(collect(range(0, 1; length=101))))
const dense50 = simulate(4, 50, RegularSampling(collect(range(0, 1; length=51))))
const irregular = simulate(5, 200, SparseSampling((0., 1.), 20:40))
const grid = collect(range(0, 1; length=51))
const gauss = KernelSmoothing.GaussianKernel()
const msparse = StatsAPI.fit(SmoothedMean, sparse; grid, bandwidth=.1)
const msparse200 = StatsAPI.fit(SmoothedMean, sparse200; grid, bandwidth=.1)
const mdense50 = StatsAPI.fit(SmoothedMean, dense50; grid, bandwidth=.05)
const pace = StatsAPI.fit(FPCA, sparse200; method=:pace)

const cases = [
    "mean_fixed_sparse500" => () -> StatsAPI.fit(SmoothedMean, sparse; grid, bandwidth=.1),
    "mean_curvecv_sparse500" => () -> StatsAPI.fit(SmoothedMean, sparse; grid,
        bandwidth=CurveCV(collect(range(.05, .3; length=10)); folds=10, rng=MersenneTwister(1))),
    "mean_derivative_sparse500" => () -> StatsAPI.fit(SmoothedMean, sparse; grid, bandwidth=.2, derivative=1),
    "smooth_curves_dense200x101" => () -> smooth_curves(dense, LocalPolynomial(; bandwidth=.05); grid),
    "smooth_curves_irregular200" => () -> smooth_curves(irregular, LocalPolynomial(; bandwidth=.1, kernel=gauss); grid),
    "covariance_fixed_sparse500" => () -> StatsAPI.fit(SmoothedCovariance, sparse, msparse; grid, bandwidth=.15),
    "covariance_fixed_gauss_sparse500" => () -> StatsAPI.fit(SmoothedCovariance, sparse, msparse; grid,
        bandwidth=.15, kernel=gauss),
    "covariance_fixed_dense50x51" => () -> StatsAPI.fit(SmoothedCovariance, dense50, mdense50; grid, bandwidth=.1),
    "covariance_curvecv10_sparse200" => () -> StatsAPI.fit(SmoothedCovariance, sparse200, msparse200; grid,
        bandwidth=CurveCV([.1, .15, .2, .3, .4]; folds=10, rng=MersenneTwister(2))),
    "covariance_curvecvloo_sparse200" => () -> StatsAPI.fit(SmoothedCovariance, sparse200, msparse200; grid,
        bandwidth=CurveCV([.1, .15, .2, .3, .4]; folds=nobs(sparse200))),
    "cross_covariance_sparse200" => () -> StatsAPI.fit(SmoothedCrossCovariance, sparse200, sparse200,
        msparse200, msparse200; grid1=grid, bandwidth=.2),
    "pace_fpca_sparse200" => () -> StatsAPI.fit(FPCA, sparse200; method=:pace),
    "fpca_derivative_sparse200" => () -> FunctionalDataAnalysis.derivative(pace, 1),
]

only_cases = get(ENV, "CASES", "")
for (name, f) in cases
    isempty(only_cases) || occursin(name, only_cases) || continue
    println(join((LABEL, name, Threads.nthreads(), timed(f)), ","))
    flush(stdout)
end
