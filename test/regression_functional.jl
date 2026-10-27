import StatsAPI
using DelimitedFiles

_m7b(name) = readdlm(joinpath(@__DIR__, "data", "m7b_$(name).csv"), ',', Any; comments=true)
_m7b_input(name) = readdlm(joinpath(@__DIR__, "data", "$(name).csv"), ','; comments=true)
_m7b_vector(table, quantity) = Float64.(table[table[:, 1] .== quantity, 4])
function _m7b_matrix(table, quantity)
    rows = table[table[:, 1] .== quantity, :]
    out = zeros(maximum(Int.(rows[:, 2])), maximum(Int.(rows[:, 3])))
    for row in eachrow(rows)
        out[Int(row[2]), Int(row[3])] = Float64(row[4])
    end
    out
end
_m7b_relative(actual, expected) = norm(actual - expected) / norm(expected)
_m7b_sparse(rows) = IrregularFunctionalData(
    [rows[rows[:, 1] .== i, 2] for i in 1:Int(maximum(rows[:, 1]))],
    [rows[rows[:, 1] .== i, 3] for i in 1:Int(maximum(rows[:, 1]))])

# Quantities shared by the dense and sparse functional-response references.
function fof_golden_checks(f, newx, golden; scalar_covariates=nothing)
    ref(q) = _m7b_vector(golden, q)
    mat(q) = _m7b_matrix(golden, q)
    @test StatsAPI.dof_residual(f) == only(ref("df_residual"))
    @test abs(StatsAPI.r2(f) / only(ref("r2")) - 1) < 1e-8
    @test _m7b_relative(StatsAPI.fitted(f).values, mat("fitted")) < 1e-8
    @test _m7b_relative(f.response, mat("response_scores")) < 1e-8
    @test _m7b_relative(f.residual_covariance, mat("residual_covariance")) < 1e-8
    intercept = reconstruct(f.response_fpca, reshape(StatsAPI.coef(f)[1, :], 1, :))
    @test _m7b_relative(vec(intercept.values), ref("intercept_function")) < 1e-8
    # The sine response eigenfunctions vanish at both endpoints, where R²(s) is
    # 0/0: NaN here, and NaN or roundoff noise in R (sin(pi) is not zero there).
    pointwise = pointwise_r2(f)
    expected = ref("pointwise_r2")
    @test findall(isnan, pointwise.values) == [1, length(expected)]
    @test _m7b_relative(pointwise.values[2:end-1], expected[2:end-1]) < 1e-8
    bands = coefficient_bands(f)
    critical = only(ref("critical_value"))
    for (j, beta) in enumerate(coefficient_functions(f))
        @test _m7b_relative(beta.values, mat("beta$j")) < 1e-8
        @test _m7b_relative(bands[j].estimate, mat("beta$j")) < 1e-8
        @test _m7b_relative(bands[j].se, mat("beta$(j)_se")) < 1e-8
        @test _m7b_relative(bands[j].lower, mat("beta$j") - critical .* mat("beta$(j)_se")) < 1e-8
        @test bands[j].response_grid == f.response_fpca.grid
    end
    predicted = StatsAPI.predict(f, newx; scalar_covariates)
    @test predicted isa GridFunctionalData
    @test _m7b_relative(predicted.values, mat("prediction")) < 1e-8
    confidence = StatsAPI.predict(f, newx; scalar_covariates, interval=:confidence)
    prediction = StatsAPI.predict(f, newx; scalar_covariates, interval=:prediction)
    se, sY = mat("prediction_se"), ref("response_score_variance")
    @test abs(confidence.critical_value / critical - 1) < 1e-8
    @test _m7b_relative(confidence.se, se) < 1e-8
    @test _m7b_relative(confidence.upper, mat("prediction") + critical .* se) < 1e-8
    @test _m7b_relative(prediction.lower,
        mat("prediction") - critical .* sqrt.(se.^2 .+ transpose(sY))) < 1e-8
end

@testset "FPC function-on-function regression" begin
    @testset "fdapace FLM dense golden values" begin
        grid = vec(_m7b_input("m7_dense_grid"))
        x1 = GridFunctionalData(_m7b_input("m7_dense_input1"), grid)
        x2 = GridFunctionalData(_m7b_input("m7_dense_input2"), grid)
        new1 = GridFunctionalData(_m7b_input("m7_dense_heldout1"), grid)
        new2 = GridFunctionalData(_m7b_input("m7_dense_heldout2"), grid)
        z = _m7b_input("m7_dense_response")[:, 2]
        znew = vec(_m7b_input("m7_dense_heldout_scalar"))
        y = GridFunctionalData(_m7b_input("m7b_fof_dense_response"), grid)
        golden = _m7b("fof_dense")
        f = StatsAPI.fit(FPCRegression, (x1, x2), y; scalar_covariates=z,
            ncomponents=(3, 2), fpca_options=((; method=:dense, max_k=3), (; method=:dense, max_k=2)),
            response_ncomponents=2, response_fpca_options=(; method=:dense, max_k=2))
        fof_golden_checks(f, (new1, new2), golden; scalar_covariates=znew)
        # R orders scores as intercept, functional scores, scalar; Julia as
        # intercept, scalar, functional scores.
        permutation = [1, 7, 2, 3, 4, 5, 6]
        @test _m7b_relative(StatsAPI.coef(f), _m7b_matrix(golden, "score_coefficient")[permutation, :]) < 1e-8
        @test _m7b_relative(StatsAPI.coef(f)[2, :], _m7b_vector(golden, "scalar_coefficient")) < 1e-8
        # FPCRegression's defaults fit FACE (here with short-grid knots) for the
        # covariates and the response; the goldens above name the dense route.
        automatic = @inferred StatsAPI.fit(FPCRegression, (x1, x2), y; scalar_covariates=z)
        knots = length(grid) - 4
        @test automatic.settings.fpca_method == (:face, :face)
        @test automatic.settings.fpca_method_source == (:short_grid_knots, :short_grid_knots)
        @test automatic.settings.fpca_knots == [knots, knots]
        @test automatic.settings.response_fpca_method === :face
        @test automatic.settings.response_fpca_method_source === :short_grid_knots
        @test automatic.settings.response_fpca_knots == [knots]
        @test automatic.settings.response_fpca_options == (;)
        face = (; method=:face, face=(; knots))
        @test StatsAPI.coef(automatic) == StatsAPI.coef(StatsAPI.fit(FPCRegression, (x1, x2), y;
            scalar_covariates=z, fpca_options=face, response_fpca_options=face))
        @test occursin("response_fpca_method_source = :short_grid_knots", sprint(show, automatic))
        dense_response = StatsAPI.fit(FPCRegression, (x1, x2), y; scalar_covariates=z,
            response_fpca_options=(; method=:dense))
        @test dense_response.response_fpca.method === :dense
        @test dense_response.settings.response_fpca_method_source === :fpca_options
        @test dense_response.settings.response_fpca_knots == [nothing]
        @test_throws r"does not accept response_fpca_options mean_bandwidth" StatsAPI.fit(FPCRegression,
            (x1, x2), y; scalar_covariates=z, response_fpca_options=(; mean_bandwidth=0.1))
        @test StatsAPI.nobs(f) == 24
        @test StatsAPI.dof(f) == 14
        @test size(StatsAPI.vcov(f)) == (14, 14)
        @test StatsAPI.stderror(f) ≈ reshape(sqrt.(diag(StatsAPI.vcov(f))), 7, 2)
        @test StatsAPI.vcov(f) ≈ kron(f.residual_covariance, inv(transpose(f.design) * f.design))
        @test f.sigma2 ≈ tr(f.residual_covariance)
        residual = StatsAPI.residuals(f)
        reconstructed = reconstruct(f.response_fpca)
        @test residual.values ≈ reconstructed.values - StatsAPI.fitted(f).values atol=1e-12
        bands = fitted_bands(f)
        @test bands.estimate ≈ StatsAPI.fitted(f).values
        @test bands.grid == grid
        coarse = collect(range(0.0, 1.0; length=6))
        @test StatsAPI.fitted(f; grid=coarse).grid == coarse
        @test size(coefficient_bands(f; grid=(coarse, coarse), response_grid=coarse)[1].estimate) == (6, 6)
        @test StatsAPI.predict(f, (x1, x2); scalar_covariates=z).values ≈ StatsAPI.fitted(f).values
        displayed = sprint(show, f)
        for choice in ("response_ncomponents", "response_fpca", "settings")
            @test occursin(choice, displayed)
        end
        @test sprint(show, MIME"text/plain"(), f) == displayed
    end

    @testset "fdapace FLM sparse golden values" begin
        read_m3(name) = _m7b_input("m3_sparse_$name")
        mu, surface = read_m3("mean"), read_m3("surface")
        x, newx = _m7b_sparse(read_m3("input")), _m7b_sparse(read_m3("heldout"))
        y = _m7b_sparse(_m7b_input("m7b_fof_sparse_response"))
        ymean = _m7b_input("m7b_fof_sparse_response_mean")
        golden = _m7b("fof_sparse")
        options = (; grid=mu[:, 1], mean=(; grid=mu[:, 1], values=mu[:, 2]),
            covariance=(; grid=mu[:, 1], surface), noise_variance=only(read_m3("noise")), max_k=5)
        response_options = (; grid=ymean[:, 1], mean=(; grid=ymean[:, 1], values=ymean[:, 2]),
            covariance=(; grid=ymean[:, 1], surface=_m7b_input("m7b_fof_sparse_response_surface")),
            noise_variance=only(_m7b_vector(golden, "response_noise")), max_k=5)
        f = @inferred StatsAPI.fit(FPCRegression, x, y; ncomponents=3, fpca_options=options,
            response_ncomponents=2, response_fpca_options=response_options)
        fof_golden_checks(f, newx, golden)
        @test _m7b_relative(StatsAPI.coef(f), _m7b_matrix(golden, "score_coefficient")) < 1e-8
        # A supplied response FPCA reuses its training scores and may be truncated.
        yfpca = StatsAPI.fit(FPCA, y; response_options...)
        supplied = StatsAPI.fit(FPCRegression, x, y; ncomponents=3, fpca_options=options,
            response_fpca=yfpca, response_ncomponents=2)
        @test StatsAPI.coef(supplied) ≈ StatsAPI.coef(f)
        @test supplied.settings.response_fpca_source == :supplied
        @test size(StatsAPI.coef(StatsAPI.fit(FPCRegression, x, y; ncomponents=3,
            fpca_options=options, response_fpca=yfpca, response_ncomponents=1)), 2) == 1
    end

    @testset "Analytic surfaces, conditional and bootstrap intervals" begin
        signs(j) = [isodd(count_ones((i - 1) & j)) ? -1.0 : 1.0 for i in 1:16]
        a, b, e = 2signs(1), signs(2), 0.2signs(3)
        grid = [0.0, 0.1, 0.35, 0.65, 0.9, 1.0]
        weights = quadrature_weights(grid)
        slope = (grid .- 0.5) ./ sqrt(dot(weights, (grid .- 0.5).^2))
        x = GridFunctionalData(transpose(2 .+ 3slope) .+ a .+ b * transpose(slope), grid)
        # A noise-free rank-one response: every resample refits the same surface.
        eta = 1.5a .- 0.75b
        y = GridFunctionalData(2 .+ eta * transpose(slope), grid)
        f = StatsAPI.fit(FPCRegression, x, y; fpca_options=(; method=:dense),
            response_fpca_options=(; method=:dense))
        @test f.counts == [2]
        @test size(StatsAPI.coef(f), 2) == 1
        expected = (1.5 .- 0.75slope) * transpose(slope)
        @test only(coefficient_functions(f)).values ≈ expected atol=1e-12
        @test StatsAPI.r2(f) ≈ 1 atol=1e-12
        rng = Random.Xoshiro(20261026)
        before = copy(rng)
        boot = only(coefficient_bands(f; method=:bootstrap, B=12, rng))
        @test rng == before
        @test boot.lower ≈ expected atol=1e-10
        @test boot.upper ≈ expected atol=1e-10
        @test boot.se ≈ zeros(6, 6) atol=1e-10
        @test boot.critical_value === nothing
        @test boot.settings.B == 12 && boot.settings.rng_state == before
        @test boot.response_grid == grid

        # Scalar response: the same collapse, plus a manual replay of the resamples.
        response = 4 .+ 1.5a .- 0.75b
        scalar = StatsAPI.fit(FPCRegression, x, response; fpca_options=(; method=:dense))
        collapsed = only(coefficient_bands(scalar; method=:bootstrap, B=10, rng=Random.Xoshiro(3)))
        @test collapsed.lower ≈ 1.5 .- 0.75slope atol=1e-10
        @test collapsed.upper ≈ 1.5 .- 0.75slope atol=1e-10
        noisy = StatsAPI.fit(FPCRegression, x, response .+ e; fpca_options=(; method=:dense))
        replay = Random.Xoshiro(11)
        draws = zeros(4, length(grid))
        for r in 1:4
            index = rand(replay, 1:16, 16)
            refit = StatsAPI.fit(FPCRegression, GridFunctionalData(x.values[index, :], grid),
                (response .+ e)[index]; fpca_options=(; method=:dense))
            draws[r, :] = vec(evaluate(only(coefficient_functions(refit)), grid))
        end
        interval = only(coefficient_bands(noisy; method=:bootstrap, B=4, rng=Random.Xoshiro(11)))
        quantile7(v, p) = (s = sort(v); h = (length(s) - 1)p; s[floor(Int, h)+1] +
            (h - floor(h)) * (s[min(floor(Int, h)+2, end)] - s[floor(Int, h)+1]))
        @test interval.lower ≈ [quantile7(draws[:, j], 0.025) for j in eachindex(grid)]
        @test interval.upper ≈ [quantile7(draws[:, j], 0.975) for j in eachindex(grid)]
        @test band_data(interval).target == :coefficient
        conditional = only(coefficient_bands(noisy))
        @test conditional.settings.distribution == :student_t

        # Basis covariates resample their coefficient rows; the noise-free
        # response again collapses every replicate onto the estimate.
        basisfit = StatsAPI.fit(FPCRegression, to_basis(x, MonomialBasis((0.0, 1.0), 2)), response)
        basisband = only(coefficient_bands(basisfit; method=:bootstrap, B=3, rng=Random.Xoshiro(3)))
        @test basisband.lower ≈ basisband.estimate atol=1e-10
        @test basisband.upper ≈ basisband.estimate atol=1e-10
        # Sparse covariates: only subject 1 is observed at t = 0, so resamples
        # without it lose part of the observed range and are redrawn (FLMCI).
        sparse = to_irregular(x)
        sparse = IrregularFunctionalData([i == 1 ? sparse[i][1] : sparse[i][1][2:end] for i in 1:16],
            [i == 1 ? sparse[i][2] : sparse[i][2][2:end] for i in 1:16])
        # Replay the draws to pick a seed (independent of the Julia version's
        # stream) whose two accepted resamples follow at least one rejection.
        function rejections(seed)
            replay, accepted, rejected = Random.Xoshiro(seed), 0, 0
            while accepted < 2
                1 in rand(replay, 1:16, 16) ? (accepted += 1) : (rejected += 1)
            end
            rejected
        end
        seed = findfirst(s -> rejections(s) > 0, 1:100)
        @test seed !== nothing
        sparsefit = StatsAPI.fit(FPCRegression, sparse, response .+ e; ncomponents=1)
        sparseband = only(coefficient_bands(sparsefit; method=:bootstrap, B=2, rng=Random.Xoshiro(seed)))
        @test all(isfinite, sparseband.se) && all(sparseband.lower .<= sparseband.upper)
        @test sparseband.settings.range_check == :all_functional_variables
        # A replicate whose refit fails is reported with its number, not dropped:
        # resamples without subject 1 have a constant response.
        spike = [1.0; zeros(15)]
        spiked = StatsAPI.fit(FPCRegression, x, spike; fpca_options=(; method=:dense))
        failure = try
            coefficient_bands(spiked; method=:bootstrap, B=50, rng=Random.Xoshiro(1))
            nothing
        catch err
            err
        end
        @test failure isa ArgumentError
        @test occursin("regression bootstrap replicate", failure.msg)
    end

    @testset "Functional response validation" begin
        grid = vec(_m7b_input("m7_dense_grid"))
        x1 = GridFunctionalData(_m7b_input("m7_dense_input1"), grid)
        y = GridFunctionalData(_m7b_input("m7b_fof_dense_response"), grid)
        z = _m7b_input("m7_dense_response")[:, 2]
        options = (; ncomponents=3, fpca_options=(; max_k=3))
        for selection in (AIC(), BIC(), :loocv)
            @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; selection, fpca_options=(; max_k=3))
        end
        @test_throws DimensionMismatch StatsAPI.fit(FPCRegression, x1, y[1:23]; options...)
        labelled = GridFunctionalData(y.values, grid; ids=string.(24:-1:1))
        @test_throws ArgumentError StatsAPI.fit(FPCRegression,
            GridFunctionalData(x1.values, grid; ids=string.(1:24)), labelled; options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; response_ncomponents=true, options...)
        yfpca = StatsAPI.fit(FPCA, y; max_k=2)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; response_fpca=yfpca,
            response_fpca_options=(; max_k=2), options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; response_fpca=yfpca,
            response_ncomponents=5, options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; response_fpca=42, options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, MultivariateFunctionalData([y, y]))
        f = StatsAPI.fit(FPCRegression, x1, y; scalar_covariates=z, options...)
        @test_throws ArgumentError coefficient_bands(f; method=:unknown)
        @test_throws ArgumentError coefficient_bands(f; B=10)
        @test_throws ArgumentError coefficient_bands(f; method=:bootstrap, B=1)
        @test_throws ArgumentError coefficient_bands(f; method=:bootstrap, rng=Random.RandomDevice())
        @test_throws ArgumentError coefficient_bands(f; response_grid=[0.5, 2.0])
        @test_throws ArgumentError StatsAPI.predict(f, x1; scalar_covariates=z, interval=:band)
        @test_throws ArgumentError fitted_bands(f; interval=:none)
        @test_throws ArgumentError pointwise_r2(f; grid=[0.0])
        supplied = StatsAPI.fit(FPCRegression, x1, y; fpca=StatsAPI.fit(FPCA, x1; max_k=3, ncomponents=3))
        @test_throws ArgumentError coefficient_bands(supplied; method=:bootstrap, B=5)
        selected = StatsAPI.fit(FPCRegression, x1, z; selection=AIC(), fpca_options=(; max_k=2))
        @test_throws ArgumentError coefficient_bands(selected; method=:bootstrap, B=5)
        scalar = StatsAPI.fit(FPCRegression, x1, z; options...)
        @test_throws ArgumentError coefficient_bands(scalar; response_grid=grid)
        @test_throws ArgumentError coefficient_bands(scalar; method=:bootstrap, B=5, response_grid=grid)
        @test_throws ArgumentError StatsAPI.predict(scalar, x1; grid)
        @test_throws ArgumentError StatsAPI.fitted(scalar; grid)
        @test_throws ArgumentError StatsAPI.residuals(scalar; grid)
        @test_throws ArgumentError fitted_bands(scalar; grid)
    end
end

@testset "Concurrent regression (fdapace FCReg)" begin
    input = _m7b_input("m7b_concurrent_dense_input")
    cgrid = collect(range(0.0, 1.0; length=11))
    x1 = GridFunctionalData(input[:, 1:11], cgrid)
    x2 = GridFunctionalData(input[:, 12:22], cgrid)
    y = GridFunctionalData(input[:, 23:33], cgrid)
    z = input[:, 34]
    @testset "Dense golden values" begin
        golden = _m7b("concurrent_dense")
        f = @inferred StatsAPI.fit(ConcurrentRegression, (x1, x2), y; scalar_covariates=z,
            mean_bandwidth=0.15, covariance_bandwidth=0.2)
        @test f.grid == cgrid
        @test f.settings.grid_source == :union_of_observation_times
        @test _m7b_relative(transpose(f.coefficients), _m7b_matrix(golden, "beta")) < 1e-8
        @test _m7b_relative(f.intercept, _m7b_vector(golden, "beta0")) < 1e-8
        @test _m7b_relative(pointwise_r2(f).values, _m7b_vector(golden, "r2")) < 1e-8
        @test StatsAPI.coef(f) == vcat(transpose(f.intercept), f.coefficients)
        functions = coefficient_functions(f)
        @test length(functions) == 4
        @test vec(functions[2].values) == f.coefficients[1, :]
        manual = transpose(f.intercept) .+ x1.values .* transpose(f.coefficients[1, :]) .+
            x2.values .* transpose(f.coefficients[2, :]) .+ z .* transpose(f.coefficients[3, :])
        @test StatsAPI.predict(f, (x1, x2); scalar_covariates=z).values ≈ manual
        @test StatsAPI.fitted(f).values ≈ manual
        @test StatsAPI.nobs(f) == 20
        explicit = StatsAPI.fit(ConcurrentRegression, MultivariateFunctionalData([x1, x2]), y;
            scalar_covariates=reshape(z, :, 1), mean_bandwidth=0.15, covariance_bandwidth=0.2, grid=cgrid)
        @test explicit.coefficients ≈ f.coefficients
        @test explicit.settings.grid_source == :supplied
        noerror = StatsAPI.fit(ConcurrentRegression, x1, y; mean_bandwidth=0.15,
            covariance_bandwidth=0.2, measurement_error=false)
        reference = _m7b("concurrent_dense_noerror")
        @test _m7b_relative(vec(noerror.coefficients), _m7b_vector(reference, "beta")) < 1e-8
        @test _m7b_relative(noerror.intercept, _m7b_vector(reference, "beta0")) < 1e-8
        @test _m7b_relative(noerror.r_squared, _m7b_vector(reference, "r2")) < 1e-8
        displayed = sprint(show, f)
        for choice in ("mean_bandwidth", "covariance_bandwidth", "measurement_error", "FCReg")
            @test occursin(choice, displayed)
        end
        @test sprint(show, MIME"text/plain"(), f) == displayed
    end

    @testset "Sparse golden values" begin
        rows = _m7b_input("m7b_concurrent_sparse_input")
        variable(v) = _m7b_sparse(rows[rows[:, 1] .== v, 2:4])
        golden = _m7b("concurrent_sparse")
        sgrid = _m7b_vector(golden, "grid")
        f = StatsAPI.fit(ConcurrentRegression, (variable(1), variable(2)), variable(3);
            mean_bandwidth=0.2, covariance_bandwidth=0.25, grid=sgrid)
        @test _m7b_relative(transpose(f.coefficients), _m7b_matrix(golden, "beta")) < 1e-8
        @test _m7b_relative(f.intercept, _m7b_vector(golden, "beta0")) < 1e-8
        @test _m7b_relative(f.r_squared, _m7b_vector(golden, "r2")) < 1e-8
        # Sparse curves do not cover the grid: no evaluator recovers their values.
        @test_throws ArgumentError StatsAPI.fitted(f)
        @test_throws ArgumentError StatsAPI.fitted(f; evaluator=PiecewiseLinear())
    end

    @testset "Bootstrap intervals and validation" begin
        f = StatsAPI.fit(ConcurrentRegression, x1, y; scalar_covariates=z,
            mean_bandwidth=0.15, covariance_bandwidth=0.2)
        rng = Random.Xoshiro(7)
        before = copy(rng)
        bands = coefficient_bands(f; B=4, rng)
        @test rng == before
        @test length(bands) == 3
        @test bands == coefficient_bands(f; B=4, rng=Random.Xoshiro(7))
        replay = Random.Xoshiro(7)
        draws = zeros(4, 11)
        for r in 1:4
            index = rand(replay, 1:20, 20)
            refit = StatsAPI.fit(ConcurrentRegression, GridFunctionalData(x1.values[index, :], cgrid),
                GridFunctionalData(y.values[index, :], cgrid); scalar_covariates=z[index],
                mean_bandwidth=0.15, covariance_bandwidth=0.2)
            draws[r, :] = refit.coefficients[1, :]
        end
        @test bands[2].lower ≈ [minimum(draws[:, t]) + 0.075 * (sort(draws[:, t])[2] - minimum(draws[:, t]))
            for t in 1:11]
        @test bands[2].estimate == f.coefficients[1, :]
        @test bands[1].settings.rng_state == before
        @test band_data(bands[1]).target == :coefficient
        @test_throws ArgumentError coefficient_bands(f; B=1)
        @test_throws ArgumentError coefficient_bands(f; alpha=1)
        basis = BasisFunctionalData(randn(Random.Xoshiro(1), 20, 3), FourierBasis((0, 1), 3))
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, basis, y;
            mean_bandwidth=0.1, covariance_bandwidth=0.1)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, basis;
            mean_bandwidth=0.1, covariance_bandwidth=0.1)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, y;
            mean_bandwidth=-1, covariance_bandwidth=0.1)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, y;
            mean_bandwidth=0.1, covariance_bandwidth=Inf)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, y;
            mean_bandwidth=0.1, covariance_bandwidth=0.1, measurement_error=1)
        @test_throws DimensionMismatch StatsAPI.fit(ConcurrentRegression, x1, y[1:19];
            mean_bandwidth=0.1, covariance_bandwidth=0.1)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, y; scalar_covariates=hcat(z, z),
            mean_bandwidth=0.15, covariance_bandwidth=0.2)
        @test_throws ArgumentError StatsAPI.fit(ConcurrentRegression, x1, y;
            mean_bandwidth=0.15, covariance_bandwidth=0.2, grid=[0.5, 0.2])
        @test_throws DimensionMismatch StatsAPI.predict(f, (x1, x2); scalar_covariates=z)
        @test_throws DimensionMismatch StatsAPI.predict(f, x1)
        @test_throws ArgumentError StatsAPI.predict(f, to_irregular(x1); scalar_covariates=z)
        @test StatsAPI.predict(f, to_irregular(x1); scalar_covariates=z,
            evaluator=NoInterpolation()).values ≈ StatsAPI.fitted(f).values
        @test_throws ArgumentError StatsAPI.predict(f, x1; scalar_covariates=z, evaluator=:linear)
        # Basis covariates are evaluated on the fitted grid without an evaluator.
        basisx = BasisFunctionalData(randn(Random.Xoshiro(2), 20, 3), FourierBasis((0, 1), 3))
        @test StatsAPI.predict(f, basisx; scalar_covariates=z).values ≈
            transpose(f.intercept) .+ evaluate(basisx, cgrid) .* transpose(f.coefficients[1, :]) .+
            z .* transpose(f.coefficients[2, :])
        # Resamples without subject 1 have a constant scalar covariate, so the
        # replicate's covariance is singular and the failure names the replicate.
        spiked = StatsAPI.fit(ConcurrentRegression, x1, y; scalar_covariates=[1.0; zeros(19)],
            mean_bandwidth=0.15, covariance_bandwidth=0.2)
        failure = try
            coefficient_bands(spiked; B=50, rng=Random.Xoshiro(1))
            nothing
        catch err
            err
        end
        @test failure isa ArgumentError
        @test occursin("concurrent bootstrap replicate", failure.msg)
    end
end

@testset "Function-on-scalar regression (fda fRegress)" begin
    function fos_inputs(name, basis)
        golden = _m7b(name)
        y = BasisFunctionalData(_m7b_matrix(golden, "response_coef"), basis)
        z = hcat(_m7b_vector(golden, "z1"), _m7b_vector(golden, "z2"))
        golden, y, z
    end
    function fos_checks(f, golden; native=true)
        mat(q) = _m7b_matrix(golden, q)
        ref(q) = _m7b_vector(golden, q)
        k = nbasis(f.bases[1])
        native && @test _m7b_relative(StatsAPI.coef(f), vec(mat("native_beta_coef"))) < 1e-8
        native && @test _m7b_relative(StatsAPI.fitted(f).coefs, mat("native_fitted_coef")) < 1e-8
        @test _m7b_relative(StatsAPI.coef(f), vec(mat("beta_coef"))) < 1e-8
        @test _m7b_relative(f.residual_covariance, mat("residual_covariance")) < 1e-8
        @test _m7b_relative(StatsAPI.vcov(f), mat("coefficient_covariance")) < 1e-8
        grid = ref("grid")
        bands = coefficient_bands(f; grid)
        critical = only(ref("critical_value"))
        for (j, band) in enumerate(bands)
            @test _m7b_relative(band.se, mat("beta_se")[:, j]) < 1e-8
            @test _m7b_relative(band.estimate,
                basis_matrix(f.bases[1], grid) * mat("beta_coef")[:, j]) < 1e-8
            @test abs(band.critical_value / critical - 1) < 1e-8
        end
        znew = mat("new_covariates")[:, 2:3]
        confidence = StatsAPI.predict(f, znew; interval=:confidence, grid)
        prediction = StatsAPI.predict(f, znew; interval=:prediction, grid)
        @test _m7b_relative(confidence.estimate, transpose(mat("prediction"))) < 1e-8
        @test _m7b_relative(confidence.se, transpose(mat("prediction_se"))) < 1e-8
        @test _m7b_relative(prediction.se.^2,
            transpose(mat("prediction_se")).^2 .+ transpose(ref("residual_variance"))) < 1e-8
        @test StatsAPI.predict(f, znew).coefs * transpose(basis_matrix(f.bases[1], grid)) ≈ confidence.estimate
    end

    @testset "Polynomial B-spline: fRegress itself is exact" begin
        basis = BSplineBasis(5, (0.0, 1.0), 5)
        golden, y, z = fos_inputs("fos_polynomial", basis)
        f = @inferred StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0)
        fos_checks(f, golden)
        @test f.penalty_order == fill(3, 3)
        @test f.lambda == zeros(3)
        @test StatsAPI.nobs(f) == 20 && StatsAPI.dof(f) == 15 && StatsAPI.dof_residual(f) == 17
        @test StatsAPI.stderror(f) ≈ sqrt.(diag(StatsAPI.vcov(f)))
        @test StatsAPI.residuals(f).coefs ≈ y.coefs - StatsAPI.fitted(f).coefs
        @test length(coefficient_functions(f)) == 3
        @test vec(coefficient_functions(f)[2].coefs) == StatsAPI.coef(f)[6:10]
        @test length(coefficient_bands(f)[1].grid) == 51
        # fda's bsplinepen uses a monomial-basis penalty for this knot-free basis;
        # compare the B-spline penalty system recomputed exactly in R instead.
        penalized_golden, _, _ = fos_inputs("fos_polynomial_penalized", basis)
        penalized = StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=[0, 1e-3, 1e-2])
        fos_checks(penalized, penalized_golden; native=false)
        @test penalized.penalty_order == fill(3, 3)
        @test _m7b_relative(StatsAPI.coef(penalized),
            vec(_m7b_matrix(penalized_golden, "native_beta_coef"))) > 0.1
        # A supplied coefficient covariance uses normal critical values.
        supplied = StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0,
            residual_covariance=_m7b_matrix(golden, "residual_covariance"))
        @test StatsAPI.vcov(supplied) ≈ StatsAPI.vcov(f)
        @test coefficient_bands(supplied)[1].critical_value ≈ 1.959963984540054 rtol=1e-12
        @test supplied.settings.residual_covariance == :supplied
        displayed = sprint(show, f)
        for choice in ("lambda", "penalty_order", "fRegress")
            @test occursin(choice, displayed)
        end
        @test sprint(show, MIME"text/plain"(), f) == displayed
    end

    @testset "Cubic B-spline with interior knots" begin
        # fRegress's Romberg inner products carry errors near 1e-5 here; R's
        # exact bsplinepen system is the documented golden value.
        golden, y, z = fos_inputs("fos_cubic", BSplineBasis(4, (0.0, 1.0), 7))
        f = StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=1e-2, penalty_order=1)
        fos_checks(f, golden; native=false)
        @test f.settings.penalty_source == :supplied
        @test StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0).penalty_order == fill(2, 3)
    end

    @testset "Analytic least squares and validation" begin
        rng = Random.Xoshiro(5)
        basis = FourierBasis((0.0, 2.0), 5)
        z = randn(rng, 12, 2)
        Y = randn(rng, 12, 5)
        y = BasisFunctionalData(Y, basis)
        f = StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0)
        X = hcat(ones(12), z)
        # An orthonormal basis separates the coefficients: ordinary least squares.
        @test reshape(StatsAPI.coef(f), 5, 3) ≈ transpose(X \ Y)
        @test f.penalty_order == fill(:harmonic_acceleration, 3)
        @test StatsAPI.fit(FunctionOnScalarRegression, z, BasisFunctionalData(Y[:, 1:3],
            MonomialBasis((0.0, 1.0), 2)); lambda=0).penalty_order == zeros(Int, 3)
        meanonly = StatsAPI.fit(FunctionOnScalarRegression, nothing, y; lambda=0)
        @test vec(StatsAPI.coef(meanonly)) ≈ vec(sum(Y; dims=1) / 12)
        @test StatsAPI.predict(meanonly, zeros(2, 0)).coefs ≈ repeat(sum(Y; dims=1) / 12, 2)
        noint = StatsAPI.fit(FunctionOnScalarRegression, z, y; intercept=false, lambda=0)
        @test reshape(StatsAPI.coef(noint), 5, 2) ≈ transpose(z \ Y)
        grid = to_grid(y, collect(range(0.0, 2.0; length=9)))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, grid; lambda=0, method=:fRegress)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, nothing, y; intercept=false)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; intercept=1)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=-1)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=[1, 2])
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0, penalty_order=-1)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z,
            BasisFunctionalData(Y, BSplineBasis(4, (0.0, 2.0), 5)); lambda=0, penalty_order=:harmonic_acceleration)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, z[1:11, :], y; lambda=0)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, hcat(z, z), y; lambda=0)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, fill(NaN, 12), y; lambda=0)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z[1:3, :], y[1:3]; lambda=0)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0,
            residual_covariance=Matrix(1.0I, 4, 4))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0,
            residual_covariance=triu(ones(5, 5)))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0,
            residual_covariance=-Matrix(1.0I, 5, 5))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y; lambda=0, residual_covariance=:sample)
        # No default penalty order exists for other bases.
        fpcy = BasisFunctionalData(Y[:, 1:2], FPCBasis(collect(range(0.0, 2.0; length=5)), randn(rng, 5, 2)))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, fpcy; lambda=0)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, fpcy)
        @test_throws DimensionMismatch StatsAPI.predict(f, z[:, 1])
        @test_throws ArgumentError StatsAPI.predict(f, nothing)
        @test_throws ArgumentError StatsAPI.predict(f, z; interval=:band)
        @test_throws ArgumentError StatsAPI.predict(f, z; grid=[0.0, 1.0])
        @test_throws ArgumentError coefficient_bands(f; grid=[0.0, 3.0])
        @test_throws ArgumentError coefficient_bands(f; alpha=0)
    end
end
