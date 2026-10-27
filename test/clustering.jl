import StatsAPI

# Goldens: test/reference/clustering_m8.R (fdapace 0.6.0, fda.usc 2.2.0,
# EMCluster 0.2-18, stats::kmeans), all compared at relative error 1e-8.
const _M8 = golden_read("m8_clustering.csv", Any)
_m8(name) = golden_values4(_M8, name)
_m8_matrix(name) = golden_matrix4(_M8, name)
_m8_labels(name) = Int.(_m8(name))
_m8_cluster(name, c) = Float64.(_M8[(_M8[:, 1] .== name) .& (_M8[:, 3] .== c), 4])

function _m8_data()
    GridFunctionalData(_m8_matrix("y"), _m8("grid"))
end

# Deterministic sparse curves from two clusters with different modes of variation.
function _m8_sparse()
    fine = collect(range(0, 1; length=41))
    n = 40
    truth = repeat([1, 2]; inner=20)
    times = [fine[sort(unique(vcat(1, 41, [((i * 7 + 5j) % 39) + 2 for j in 1:12])))] for i in 1:n]
    curve(i, t) = truth[i] == 1 ? 1 + t + 1.5cos(2pi * i / n) * sqrt(2) * sin(pi * t) :
        1.5 + t + 1.5cos(2pi * i / n + 1) * sqrt(2) * sin(3pi * t)
    IrregularFunctionalData(times, [curve.(i, times[i]) for i in 1:n]), truth
end

# Well-separated grid curves: three shifted groups whose two scores are normal
# quantiles of low-discrepancy sequences (deterministic, no RNG stream).
function _m8_separated()
    grid = collect(range(0, 1; length=15))
    truth = repeat(1:3; inner=20)
    z(i, a) = sqrt(2) * FunctionalDataAnalysis.SpecialFunctions.erfinv(2 * mod(i * a + 0.5, 1) - 1)
    values = [3 * truth[i] + 0.8z(i, 0.6180339887) * sqrt(2) * sin(2pi * t) +
        0.4z(i, 0.7548776662) * sqrt(2) * cos(2pi * t) for i in eachindex(truth), t in grid]
    GridFunctionalData(values, grid), truth
end

# Three groups along one direction, six standard deviations apart, with
# normal quantiles as scores: one FPC, no lattice structure for extra mixture
# components to exploit, so BIC's choice does not depend on the random starts.
function _m8_separated_line()
    grid = collect(range(0, 1; length=15))
    truth = repeat(1:3; inner=20)
    z(i) = sqrt(2) * FunctionalDataAnalysis.SpecialFunctions.erfinv((2 * mod1(i, 20) - 1) / 20 - 1)
    values = [(6 * truth[i] + z(i)) * sqrt(2) * sin(2pi * t) for i in eachindex(truth), t in grid]
    GridFunctionalData(values, grid), truth
end

@testset "functional k-means" begin
    errors = Float64[]
    fd = _m8_data()
    seeds = _m8_labels("fdausc_seeds")
    m = @inferred StatsAPI.fit(FunctionalKMeans, fd; k=3, init=seeds)
    # fda.usc kmeans.fd with seed curves and trim=0: nearest seed, mean centres.
    @test m.assignments == _m8_labels("fdausc_cluster")
    @test golden_relative_error(errors, m.center_coordinates, _m8_matrix("fdausc_centers")) < 1e-8
    @test golden_relative_error(errors, m.centers.values, _m8_matrix("fdausc_centers")) < 1e-8
    @test m.centers.grid == fd.grid
    seed_distance = _m8_matrix("fdausc_seed_distance")
    @test m.costs ≈ [sum(fd.weights .* (fd.values[i, :] .- m.center_coordinates[m.assignments[i], :]).^2)
        for i in 1:nobs(fd)] rtol=1e-12
    # metric.lp distances to the seeds use the stored trapezoidal weights.
    own = [sqrt(sum(fd.weights .* (fd.values[i, :] .- fd.values[s, :]).^2)) for i in 1:nobs(fd), s in seeds]
    @test golden_relative_error(errors, own, seed_distance) < 1e-8
    @test m.seeds == seeds && m.converged && m.space === :L2 && m.k == 3
    @test sum(m.withinss) ≈ m.totalcost
    @test m.counts == [count(==(c), m.assignments) for c in 1:3]
    @test assignments(m) == m.assignments && nclusters(m) == 3 && nobs(m) == 36
    @test StatsAPI.predict(m, fd) == m.assignments
    @test m.settings.absolute_tol > 0 && m.settings.init == seeds
    @test occursin("FunctionalKMeans(space=L2", sprint(show, MIME("text/plain"), m))

    # stats::kmeans (Lloyd) on the FPC scores from seed rows.
    f = StatsAPI.fit(FPCA, fd; ncomponents=FVE(0.9))
    @test golden_relative_error(errors, scores(f), _m8_matrix("fpca_scores")) < 1e-8
    @test golden_relative_error(errors, eigvals(f), _m8("fpca_lambda")) < 1e-8
    s = @inferred StatsAPI.fit(FunctionalKMeans, f; k=3, init=_m8_labels("kmeans_seeds"))
    @test s.assignments == _m8_labels("kmeans_cluster")
    @test golden_relative_error(errors, s.center_coordinates, _m8_matrix("kmeans_centers")) < 1e-8
    @test golden_relative_error(errors, s.withinss, _m8("kmeans_withinss")) < 1e-8
    @test s.space === :fpc_scores && s.fpca === f
    @test s.centers.values ≈ reconstruct(f, s.center_coordinates).values
    @test StatsAPI.predict(s, fd) == s.assignments
    @test maximum(errors) < 1e-8

    # Random seeding, several starts and the fda.usc spread initializer.
    sep, truth = _m8_separated()
    for init in (:kmpp, :rand, :spread)
        r = StatsAPI.fit(FunctionalKMeans, sep; k=3, init, nstart=4, rng=Random.Xoshiro(11))
        @test randindex(r, truth)[1] ≈ 1
        @test length(r.start_costs) == 4 && r.totalcost ≈ minimum(r.start_costs)
        @test r.start_costs[r.best_start] == minimum(r.start_costs)
        @test length(r.seeds) == 3 && allunique(r.seeds)
    end
    # The relative tolerance makes the result invariant to the data's units.
    tiny = GridFunctionalData(1e-6 .* sep.values, sep.grid)
    @test StatsAPI.fit(FunctionalKMeans, tiny; k=3, init=[1, 21, 41]).assignments ==
        StatsAPI.fit(FunctionalKMeans, sep; k=3, init=[1, 21, 41]).assignments

    # Basis data use the exact Gram matrix.
    b = FourierBasis((0, 1), 5)
    coefs = [3 * truth[i] + (j == 1 ? 0 : 0.2 * cos(i * j)) for i in eachindex(truth), j in 1:5]
    bd = BasisFunctionalData(coefs, b)
    r = StatsAPI.fit(FunctionalKMeans, bd; k=3, init=[1, 21, 41])
    @test randindex(r, truth)[1] ≈ 1
    @test r.centers isa BasisFunctionalData && r.centers.basis == b
    fine = collect(range(0, 1; length=4001))
    values = evaluate(bd, fine)
    centres = evaluate(r.centers, fine)
    w = quadrature_weights(fine)
    numeric = [sum(w .* (values[i, :] .- centres[r.assignments[i], :]).^2) for i in eachindex(truth)]
    @test maximum(abs.(numeric .- r.costs) ./ r.costs) < 1e-5
    @test StatsAPI.predict(r, bd) == r.assignments

    # Errors.
    masked = GridFunctionalData([1.0 missing 2.0; 1.0 2.0 3.0; 4.0 5.0 6.0], [0.0, 0.5, 1.0])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, masked; k=2)
    sparse, _ = _m8_sparse()
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, sparse; k=2)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=37)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, init=[1, 1, 2])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, init=[1, 2])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, init=[1, 2, 40])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, init=seeds, nstart=2)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, init=:hartigan)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, nstart=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, maxiter=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, fd; k=3, tol=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKMeans, GridFunctionalData(ones(4, 3), [0, 0.5, 1]); k=2)
    @test_throws DimensionMismatch StatsAPI.predict(m, GridFunctionalData(fd.values[:, 1:5], fd.grid[1:5]))
    @test_throws ArgumentError StatsAPI.predict(m, GridFunctionalData(fd.values, fd.grid .+ 1))
    holes = Matrix{Union{Missing,Float64}}(fd.values)
    holes[1, 3] = missing
    @test_throws ArgumentError StatsAPI.predict(m, GridFunctionalData(holes, fd.grid))
    @test_throws ArgumentError StatsAPI.predict(m, bd)
    @test_throws ArgumentError StatsAPI.predict(r, BasisFunctionalData(coefs[:, 1:3], FourierBasis((0, 1), 3)))
end

@testset "kCFC" begin
    errors = Float64[]
    fd = _m8_data()
    initial = _m8_labels("kcfc_initial")
    m = @inferred StatsAPI.fit(KCFC, fd; k=3, initial)
    @test m.assignments == _m8_labels("kcfc_cluster")
    @test m.iterations == only(_m8_labels("kcfc_iterations"))
    @test m.status === :converged
    @test golden_relative_error(errors, m.costs, _m8_matrix("kcfc_costs")) < 1e-8
    @test reduce(hcat, m.history) == _m8_matrix("kcfc_history")
    for c in 1:3
        @test golden_relative_error(errors, vec(m.centers.values[c, :]), _m8_cluster("kcfc_mean", c)) < 1e-8
        @test golden_relative_error(errors, eigvals(m.fpcas[c]), _m8_cluster("kcfc_lambda", c)) < 1e-8
    end
    @test maximum(errors) < 1e-8
    @test m.initial_assignments == initial && m.initial_clustering === nothing
    @test ncomponents(m.initial_fpca) == 3 && m.settings.initial === :supplied
    @test m.counts == [count(==(c), m.assignments) for c in 1:3] && nobs(m) == 36
    @test assignments(m) == m.assignments && nclusters(m) == 3
    text = sprint(show, MIME("text/plain"), m)
    @test occursin("status=converged", text) && occursin("cluster_fpca_options", text)

    # Initial k-means of the FPC scores (Lloyd, k-means++).
    automatic = StatsAPI.fit(KCFC, fd; k=3, rng=Random.Xoshiro(5))
    @test automatic.initial_clustering isa FunctionalKMeans
    @test automatic.initial_assignments == automatic.initial_clustering.assignments
    @test automatic.initial_clustering.settings.maxiter == 125
    @test automatic.settings.initial === :kmeans_scores

    # Sparse data through PACE; fdapace 0.6.0 stops for sparse designs.
    sparse, truth = _m8_sparse()
    start = copy(truth)
    start[[1, 4, 22, 35]] .= 3 .- start[[1, 4, 22, 35]]
    s = StatsAPI.fit(KCFC, sparse; k=2, initial=start, cluster_fpca_options=(; ncomponents=1))
    @test s.assignments == truth && s.status === :converged
    @test s.settings.cluster_fpca_options.grid == s.initial_fpca.grid
    @test all(f -> f.method === :pace && f.grid == s.initial_fpca.grid, s.fpcas)
    @test size(s.centers.values) == (2, length(s.initial_fpca.grid))
    @test occursin("51-point vector", sprint(show, s))

    # Non-convergence warns and is recorded.
    capped = @test_logs (:warn, r"did not converge") StatsAPI.fit(KCFC, fd; k=3, initial, maxiter=1)
    @test capped.status === :maxiter && capped.iterations == 0
    tiny_start = copy(m.assignments)
    tiny_start[[2, 3, 14, 26]] .= 4
    tiny = @test_logs (:warn, r"at most max") StatsAPI.fit(KCFC, fd; k=4, initial=tiny_start)
    @test tiny.status === :tiny_cluster && minimum(tiny.counts) <= 3 && tiny.iterations == 0

    # Errors.
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=1)
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=3, maxiter=0)
    @test_throws DimensionMismatch StatsAPI.fit(KCFC, fd; k=3, initial=initial[1:10])
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=3, initial=fill(1, 36))
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=3, initial=Float64.(initial))
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=3, initial, kmeans_options=(; nstart=2))
    few = vcat(fill(1, 3), fill(2, 16), fill(3, 17))
    @test_throws ArgumentError StatsAPI.fit(KCFC, fd; k=3, initial=few)
    single = IrregularFunctionalData([[0.0], [0.0, 1.0]], [[1.0], [1.0, 2.0]])
    @test_throws ArgumentError StatsAPI.fit(KCFC, single; k=2)
    @test_throws ArgumentError StatsAPI.fit(KCFC, BasisFunctionalData(ones(4, 3), FourierBasis((0, 1), 3)); k=2)
end

@testset "Gaussian mixture on FPC scores" begin
    errors = Float64[]
    fd = _m8_data()
    f = StatsAPI.fit(FPCA, fd; ncomponents=FVE(0.9))
    initial = _m8_labels("em_initial")
    m = @inferred StatsAPI.fit(FunctionalGaussianMixture, f; k=3, init=initial, parameter_tol=1e-12)
    @test golden_relative_error(errors, m.weights, _m8("em_pi")) < 1e-8
    @test golden_relative_error(errors, m.means, _m8_matrix("em_mu")) < 1e-8
    @test golden_relative_error(errors, reduce(vcat, m.covariances), _m8_matrix("em_sigma")) < 1e-8
    @test golden_relative_error(errors, m.posterior, _m8_matrix("em_posterior")) < 1e-8
    @test golden_relative_error(errors, [m.loglikelihood, m.aic, m.bic],
        [only(_m8("em_loglik")), only(_m8("em_aic")), only(_m8("em_bic"))]) < 1e-8
    @test maximum(errors) < 1e-8
    @test m.assignments == _m8_labels("em_class")
    @test m.status === :converged
    @test m.iterations < 1000 && m.settings.parameter_tol == 1e-12
    @test StatsAPI.loglikelihood(m) == m.loglikelihood && StatsAPI.aic(m) == m.aic &&
        StatsAPI.bic(m) == m.bic && StatsAPI.dof(m) == 29 && nobs(m) == 36
    @test m.initialization.method === :supplied_labels && m.initialization.labels == initial
    @test m.centers.values ≈ reconstruct(f, m.means).values
    @test posterior_probabilities(m) == m.posterior
    @test all(sum(m.posterior; dims=2) .≈ 1)
    @test posterior_probabilities(m, fd) ≈ m.posterior
    @test StatsAPI.predict(m, fd) == m.assignments
    @test occursin("FunctionalGaussianMixture(n=36", sprint(show, MIME("text/plain"), m))

    # EMCluster's default relative log-likelihood rule stops near the fixed point.
    default = StatsAPI.fit(FunctionalGaussianMixture, f; k=3, init=initial)
    @test default.assignments == m.assignments
    @test abs(default.loglikelihood - m.loglikelihood) <= 1e-6 * abs(m.loglikelihood)
    @test default.iterations < m.iterations

    # em.EM initialization on separated data, from data or from a fitted FPCA.
    # The data make every assertion hold whatever the random starts are.
    sep, truth = _m8_separated_line()
    fitted = StatsAPI.fit(FunctionalGaussianMixture, sep; k=3, fpca_options=(; ncomponents=1),
        rng=Random.Xoshiro(7))
    @test randindex(fitted, truth)[1] ≈ 1
    @test fitted.status === :converged
    @test fitted.initialization.method === :emem && fitted.initialization.starts >= 1
    @test fitted.initialization.short_iterations >= 200 && fitted.initialization.attempts >= 1
    @test ncomponents(fitted.fpca) == 1
    selected = StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=1:4, rng=Random.Xoshiro(8))
    @test selected.k == 3 && selected.selection.chosen == 3
    @test selected.selection.values[3] == minimum(selected.selection.values)
    @test selected.selection.dof == [2, 5, 8, 11]
    # The fourth component gains far less log likelihood than BIC charges for it.
    gain = selected.selection.loglikelihood[4] - selected.selection.loglikelihood[3]
    @test isnan(gain) || gain < 1
    by_aic = StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=[2, 3], criterion=AIC(),
        rng=Random.Xoshiro(9))
    @test by_aic.selection.criterion === :AIC && by_aic.k == 3
    # Eight components on 60 one-dimensional scores collapse onto too few curves.
    overfit = StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=[3, 8], restarts=2, rng=Random.Xoshiro(10))
    @test overfit.k == 3 && overfit.selection.degenerate == [8]
    @test overfit.selection.values[2] == Inf && isnan(overfit.selection.loglikelihood[2])
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=8, restarts=2,
        rng=Random.Xoshiro(10))
    # Initialization failures are skipped in a range too: 30 classes of at least two
    # curves each are almost never drawn, and 31 classes cannot be filled.
    toolarge = StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=[3, 30, 31], restarts=1,
        rng=Random.Xoshiro(11))
    @test toolarge.k == 3 && toolarge.selection.degenerate == [30, 31]
    @test toolarge.selection.values[2:3] == [Inf, Inf]
    @test_throws "only 60 curves" StatsAPI.fit(FunctionalGaussianMixture, fitted.fpca; k=31)
    @test FunctionalDataAnalysis._gmm_emem(fitted.fpca.scores, 30, Random.Xoshiro(12), 200, 0.01) isa String

    # Errors.
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=2:3, init=initial)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=3, init=:kmeans)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=3, criterion=FVE(0.9))
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=Int[])
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=[2, 2])
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=2.5)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; tol=-1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; short_tol=Inf)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; parameter_tol=-1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; parameter_tol=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; maxiter=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; short_iter=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; restarts=0)
    small = vcat(fill(1, 3), fill(2, 16), fill(3, 17))
    @test_throws ArgumentError StatsAPI.fit(FunctionalGaussianMixture, f; k=3, init=small)
    @test_throws DimensionMismatch StatsAPI.fit(FunctionalGaussianMixture, f; k=3, init=initial[1:5])

    # A component without posterior weight: short runs skip it, the final EM throws.
    G = FunctionalDataAnalysis.GaussianMixtures
    X = reshape(collect(0.0:0.1:0.9), :, 1)
    collapsed() = G.GMM([0.5, 0.5], reshape([0.45, 100.0], 2, 1),
        [UpperTriangular(fill(1 / 0.3, 1, 1)), UpperTriangular(fill(10.0, 1, 1))], G.History[], 10)
    em = (; maxiter=10, tol=1e-6, parameter_tol=nothing, short=false)
    @test FunctionalDataAnalysis._gmm_em!(collapsed(), X; em..., strict=false).status === :degenerate
    @test_throws ArgumentError FunctionalDataAnalysis._gmm_em!(collapsed(), X; em..., strict=true)
end
