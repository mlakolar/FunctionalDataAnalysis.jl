import StatsAPI

const _PFR_GOLDEN = golden_read("m7_refund_pfr.csv")
_pfr_matrix(q) = golden_matrix4(_PFR_GOLDEN, q)
_pfr_values(q) = vec(_pfr_matrix(q))
_pfr_rel(a, b) = golden_relative_error(a, b)

# Smooth random curves (fixed seed, compared only with themselves or analytic values).
function _pfr_curves(rng, n, grid; noise=0.05)
    u = (grid .- first(grid)) ./ (last(grid) - first(grid))
    basis = hcat(ones(length(u)), u, sinpi.(u), cospi.(u), sinpi.(2u), cospi.(2u))
    randn(rng, n, 6) * Diagonal(1 ./ sqrt.(1:6)) * transpose(basis) .+ noise .* randn(rng, n, length(u))
end

# Deterministic, Julia-version-independent pseudo-random values in (-1/2, 1/2)
# (additive recurrence with plastic-number increments).
_pfr_uniform(n, m, seed) = [mod(i * 0.7548776662466927 + j * 0.5698402909980532 +
    seed * 0.4142135623730951 + 0.37 * sinpi(0.123 * i * j), 1) - 0.5 for i in 1:n, j in 1:m]
function _pfr_deterministic_curves(n, grid, seed)
    u = (grid .- first(grid)) ./ (last(grid) - first(grid))
    basis = hcat(ones(length(u)), u, sinpi.(u), cospi.(u), sinpi.(2u), cospi.(2u))
    sqrt(12) .* _pfr_uniform(n, 6, seed) * Diagonal(1 ./ sqrt.(1:6)) * transpose(basis) .+
        0.1 .* _pfr_uniform(n, length(u), seed + 1)
end

@testset "Penalized scalar-on-function regression" begin
    @testset "Thin plate regression basis matches mgcv 1.9-3" begin
        for name in ("g1", "small", "knots")
            points = _pfr_values("tp_$(name)_points")
            reference = _pfr_matrix("tp_$(name)_design")
            k = size(reference, 2)
            b = ThinPlateRegressionBasis(points, k)
            @test nbasis(b) == k
            @test domain(b) == extrema(points)
            B = basis_matrix(b, points)
            # Columns agree up to sign (eigenvector signs are arbitrary).
            signs = Diagonal(sign.(vec(sum(B .* reference; dims=1))))
            @test _pfr_rel(B * signs, reference) < 1e-8
            @test _pfr_rel(signs * b.penalty * signs, _pfr_matrix("tp_$(name)_penalty")) < 1e-8
        end
        b = ThinPlateRegressionBasis(_pfr_values("tp_g1_points"))
        # Natural cubic splines: the mgcv penalty is ∫f''² over the domain.
        @test penalty(b, 2) ≈ b.penalty rtol=1e-9
        # Linear functions are in the unpenalized null space.
        t = collect(range(0, 1; length=7))
        B = basis_matrix(b, t)
        coefficients = vcat(zeros(8), B[:, 9:10] \ (2 .+ 3t))
        @test B * coefficients ≈ 2 .+ 3t atol=1e-12
        @test dot(coefficients, b.penalty * coefficients) ≈ 0 atol=1e-10
        @test basis_matrix(b, t; derivative=2) * coefficients ≈ zeros(7) atol=1e-10
        @test basis_matrix(b, t; derivative=4) == zeros(7, 10)
        # Derivatives agree with central differences.
        h = 1e-6
        D = (basis_matrix(b, [0.4 + h]) - basis_matrix(b, [0.4 - h])) / 2h
        @test vec(D) ≈ vec(basis_matrix(b, [0.4]; derivative=1)) rtol=1e-6
        @test b == ThinPlateRegressionBasis(_pfr_values("tp_g1_points"))
        @test hash(b) == hash(ThinPlateRegressionBasis(_pfr_values("tp_g1_points")))
        @test occursin("ThinPlateRegressionBasis(10 functions", sprint(show, b))
        @test_throws ArgumentError ThinPlateRegressionBasis(t, 2)
        @test_throws ArgumentError ThinPlateRegressionBasis(t, 8)
        @test_throws ArgumentError ThinPlateRegressionBasis([0.0, NaN, 1.0], 3)
        @test_throws ArgumentError ThinPlateRegressionBasis(Float64[], 3)
        @test_throws ArgumentError basis_matrix(b, [1.5])
    end

    @testset "refund 0.1-40 pfr golden values" begin
        g1, g2, g3 = _pfr_values("grid1"), _pfr_values("grid2"), _pfr_values("grid3")
        X1, X2, X3, X4 = _pfr_matrix("X1"), _pfr_matrix("X2"), _pfr_matrix("X3"), _pfr_matrix("X4")
        z = _pfr_values("z")
        train, test = 1:50, 51:56
        grid(X, g, rows) = GridFunctionalData(X[rows, :], g)
        cases = [
            ("one", "one", (grid(X1, g1, train),), (grid(X1, g1, test),), (;), true),
            ("gcv", "one", (grid(X1, g1, train),), (grid(X1, g1, test),), (; criterion=:GCV), false),
            ("two", "two", (grid(X1, g1, train), grid(X2, g2, train)),
                (grid(X1, g1, test), grid(X2, g2, test)), (;), true),
            # refund's defaults: Simpson-type weights on a uniform grid.
            ("simpson", "simpson", (grid(X3, g3, train),), (grid(X3, g3, test),),
                (; integration=:refund_simpson), false),
            ("presmooth", "presmooth", (grid(X4, g1, train),), (grid(X4, g1, test),),
                (; presmooth=:bspline), true),
        ]
        for (case, response, x, xnew, options, compare_vc) in cases
            y = _pfr_values("y_$response")[train]
            free = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z[train], options...)
            # Smoothing parameters per unscaled penalty, from mgcv's sp and S.scale.
            lambda = _pfr_values("$(case)_sp") ./ _pfr_values("$(case)_S_scale") .* free.penalty_scale
            model = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z[train],
                lambda, options...)
            @test model.settings.lambda_source == :fixed
            # The trapezoidal default matches refund's integration="trapezoidal";
            # :refund_simpson matches refund's default integration="simpson".
            @test all(==(case == "simpson" ? :refund_simpson : :quadrature), model.settings.integration)
            @test _pfr_rel(StatsAPI.coef(model)[1:2], _pfr_values("$(case)_parametric")) < 1e-8
            @test _pfr_rel(StatsAPI.fitted(model), _pfr_values("$(case)_fitted")) < 1e-8
            @test _pfr_rel([model.sigma2], _pfr_values("$(case)_sigma2")) < 1e-8
            @test _pfr_rel([model.pls.criterion_value], _pfr_values("$(case)_score")) < 1e-8
            @test _pfr_rel(model.edf_terms, _pfr_values("$(case)_edf")) < 1e-8
            @test _pfr_rel([StatsAPI.dof(model)], _pfr_values("$(case)_edf_total")) < 1e-8
            # Our optimum is at least as good as mgcv's selection.
            @test free.pls.criterion_value <= model.pls.criterion_value + 1e-10 * abs(model.pls.criterion_value)
            prediction = StatsAPI.predict(model, xnew; scalar_covariates=z[test], interval=:confidence)
            @test _pfr_rel(prediction.estimate, _pfr_values("$(case)_predict")) < 1e-8
            @test _pfr_rel(prediction.se, _pfr_values("$(case)_predict_se")) < 1e-8
            @test prediction.settings.covariance == :bayesian
            grids = ntuple(j -> _pfr_values("$(case)_beta_grid$j"), length(x))
            for j in eachindex(x)
                @test _pfr_rel(FunctionalDataAnalysis._pfr_default_grid(model.terms[j].grid), grids[j]) < 1e-12
            end
            bands = coefficient_bands(model; covariance=:bayesian)
            for j in eachindex(x)
                @test _pfr_rel(bands[j].grid, grids[j]) < 1e-12
                @test _pfr_rel(bands[j].estimate, _pfr_values("$(case)_beta$j")) < 1e-8
                @test _pfr_rel(bands[j].se, _pfr_values("$(case)_se_vp$j")) < 1e-8
            end
            if compare_vc
                # mgcv's Vc at its selected smoothing parameters.
                penalties = [t.penalty ./ t.penalty_scale for t in model.terms]
                corrected, info = FunctionalDataAnalysis._pfr_corrected_covariance(model.design,
                    model.response, penalties, model.pls.penalty_columns, model.pls.lambda,
                    model.pls.coef, model.sigma2, model.pls.null_space_dimension)
                @test _pfr_rel(info.hessian, _pfr_matrix("$(case)_hessian")) < 1e-7
                offset = 2
                for j in eachindex(x)
                    index = offset .+ (1:model.terms[j].k)
                    B = basis_matrix(model.terms[j].basis, grids[j])
                    se = sqrt.(vec(sum((B * corrected[index, index]) .* B; dims=2)))
                    @test _pfr_rel(se, _pfr_values("$(case)_se$j")) < 1e-8
                    offset += model.terms[j].k
                end
            end
        end
    end

    @testset "Analytic design, integration weights and fixed-lambda solution" begin
        rng = Random.MersenneTwister(20261026)
        grid = collect(range(0, 2; length=9)) .^ 1.2
        X = _pfr_curves(rng, 30, grid)
        z = randn(rng, 30, 2)
        y = randn(rng, 30)
        x = GridFunctionalData(X, grid)
        weights = [0.1, 0.3, 0.2, 0.25, 0.15, 0.2, 0.3, 0.25, 0.1]
        f = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z,
            integration=weights, lambda=[0.0], k=6)
        B = basis_matrix(f.terms[1].basis, grid)
        design = hcat(ones(30), z, (X .* transpose(weights)) * B)
        @test f.design ≈ design rtol=1e-13
        @test StatsAPI.coef(f) ≈ design \ y rtol=1e-9
        @test StatsAPI.dof(f) ≈ 9 rtol=1e-10
        @test StatsAPI.dof_residual(f) ≈ 21 rtol=1e-10
        @test f.sigma2 ≈ sum(abs2, y - design * (design \ y)) / 21 rtol=1e-9
        @test StatsAPI.r2(f) ≈ 1 - sum(abs2, StatsAPI.residuals(f)) / sum(abs2, y .- sum(y) / 30)
        @test f.settings.corrected_covariance == :unavailable
        @test_throws ArgumentError StatsAPI.vcov(f; type=:corrected)
        @test StatsAPI.vcov(f) ≈ f.pls.vcov_bayesian
        @test StatsAPI.vcov(f; type=:frequentist) ≈ f.pls.vcov_frequentist
        @test StatsAPI.stderror(f) ≈ sqrt.(diag(StatsAPI.vcov(f)))
        # refund's integration formulas, written out by hand.
        J = length(grid)
        simpson = (grid[J] - grid[1]) / J / 3 .* [1, 4, 2, 4, 2, 4, 2, 4, 1]
        @test FunctionalDataAnalysis._pfr_integration(:refund_simpson, grid) ≈ simpson
        d = diff(grid)
        @test FunctionalDataAnalysis._pfr_integration(:riemann, grid) ≈ vcat(sum(d) / 8, d)
        @test FunctionalDataAnalysis._pfr_integration(:quadrature, grid) == quadrature_weights(grid)
        # A supplied basis with its own derivative penalty.
        spline = BSplineBasis(4, (0.0, last(grid)), 7)
        g = StatsAPI.fit(PenalizedFunctionalRegression, x, y; basis=spline, penalty_order=2)
        @test g.terms[1].basis == spline
        @test g.terms[1].penalty ≈ penalty(spline, 2)
        Bs = basis_matrix(spline, grid)
        @test g.penalty_scale[1] ≈ opnorm(penalty(spline, 2), 1) / maximum(sum(abs, Bs; dims=2))^2
        @test only(coefficient_functions(g)).basis == spline
        # y is pure noise, so REML sends λ to its bound: Vc is not formed and
        # Vp is used, instead of factorizing a numerically singular XᵀX + S_λ.
        @test g.settings.lambda_status == [:effectively_infinite]
        @test g.settings.corrected_covariance == :unavailable_infinite_lambda
        @test g.corrected_covariance === nothing
        @test isempty(g.settings.smoothing_parameter_hessian)
        @test_throws ArgumentError StatsAPI.vcov(g; type=:corrected)
        @test only(coefficient_bands(g)).settings.covariance == :bayesian
        @test StatsAPI.predict(g, x; interval=:confidence).settings.covariance == :bayesian
        @test occursin("unavailable_infinite_lambda", sprint(show, g))
        # More basis functions (12) than grid points (9): the penalty identifies
        # the coefficients, as in mgcv; without it the design is rank deficient.
        wide = BSplineBasis(4, (0.0, last(grid)), 12)
        h = StatsAPI.fit(PenalizedFunctionalRegression, x, y; basis=wide, penalty_order=2)
        @test h.terms[1].k == 12
        @test rank(h.design) < size(h.design, 2)
        @test all(isfinite, StatsAPI.coef(h))
        @test h.settings.corrected_covariance == :unavailable_infinite_lambda
        @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, x, y; basis=wide,
            penalty_order=2, lambda=[0.0])
        # A failed Cholesky factorization yields no Vc rather than an error.
        @test FunctionalDataAnalysis._pfr_corrected_covariance([ones(5) zeros(5)], ones(5),
            [ones(1, 1)], [[2]], [0.0], [1.0, 0.0], 1.0, 1) === (nothing, nothing)
    end

    @testset "Recovery, bands, prediction and shared interface" begin
        grid = collect(range(0, 1; length=30)) .^ 1.3
        X1 = _pfr_deterministic_curves(120, grid, 1)
        g2 = 2 .* grid
        X2 = _pfr_deterministic_curves(120, g2, 3)
        z = sqrt(12) .* vec(_pfr_uniform(120, 1, 5))
        beta1 = sinpi.(2grid)
        beta2 = (g2 ./ 2) .^ 2
        w, w2 = quadrature_weights(grid), quadrature_weights(g2)
        y = 1 .+ 0.5z .+ X1 * (w .* beta1) .+ X2 * (w2 .* beta2) .+ 0.1 .* vec(_pfr_uniform(120, 1, 7))
        x1, x2 = GridFunctionalData(X1[1:100, :], grid), GridFunctionalData(X2[1:100, :], g2)
        f = @inferred StatsAPI.fit(PenalizedFunctionalRegression, (x1, x2), y[1:100];
            scalar_covariates=z[1:100])
        @test (@inferred StatsAPI.nobs(f)) == 100
        @test f.settings.criterion == :REML
        @test all(in((:estimated, :effectively_infinite)), f.settings.lambda_status)
        @test StatsAPI.coef(f)[1:2] ≈ [1, 0.5] atol=0.05
        beta = @inferred coefficient_functions(f)
        @test beta isa Tuple{BasisFunctionalData,BasisFunctionalData}
        @test maximum(abs, vec(evaluate(beta[1], grid)) - beta1) < 0.25
        @test maximum(abs, vec(evaluate(beta[2], g2)) - beta2) < 0.25
        # Corrected covariance is the refund default for bands and adds variance.
        bands = coefficient_bands(f; alpha=0.1)
        @test bands[1].settings.covariance == :corrected
        @test bands[1].critical_value ≈ 1.6448536269514722 rtol=1e-12
        plain = coefficient_bands(f; covariance=:bayesian, grid=(grid, g2))
        @test plain[1].grid == grid
        @test all(StatsAPI.stderror(f; type=:corrected) .>= StatsAPI.stderror(f) .- 1e-12)
        @test bands[1].upper ≈ bands[1].estimate .+ bands[1].critical_value .* bands[1].se
        @test band_data(bands[1]).values == permutedims(bands[1].estimate)
        @test size(f.settings.smoothing_parameter_hessian) == (3, 3)
        # Prediction for new subjects equals the weighted-sum formula.
        new = (GridFunctionalData(X1[101:120, :], grid), GridFunctionalData(X2[101:120, :], g2))
        estimate = @inferred StatsAPI.predict(f, new; scalar_covariates=z[101:120])
        b = StatsAPI.coef(f)
        manual = b[1] .+ b[2] .* z[101:120] .+
            vec(evaluate(beta[1], grid) * (w .* transpose(X1[101:120, :]))) .+
            vec(evaluate(beta[2], g2) * (w2 .* transpose(X2[101:120, :])))
        @test estimate ≈ manual rtol=1e-10
        @test StatsAPI.predict(f, (x1, x2); scalar_covariates=z[1:100]) ≈ StatsAPI.fitted(f) rtol=1e-12
        interval = StatsAPI.predict(f, new; scalar_covariates=z[101:120], interval=:prediction)
        confidence = StatsAPI.predict(f, new; scalar_covariates=z[101:120], interval=:confidence)
        @test interval.se .^ 2 ≈ confidence.se .^ 2 .+ f.sigma2
        @test interval.settings.distribution == :normal
        fb = fitted_bands(f; interval=:prediction, covariance=:corrected)
        @test fb.estimate ≈ StatsAPI.fitted(f)
        @test fb.settings.covariance == :corrected
        @test occursin("PenalizedFunctionalRegression(n=100, terms=2", sprint(show, f))
        @test occursin("lambda=", sprint(show, MIME("text/plain"), f))
        # FPC and penalized fits share downstream code.
        fpc = StatsAPI.fit(FPCRegression, (x1, x2), y[1:100]; scalar_covariates=z[1:100],
            fpca_options=(; method=:dense))
        summary(m) = (length(coefficient_functions(m)), length(StatsAPI.fitted(m)),
            length(StatsAPI.predict(m, new; scalar_covariates=z[101:120])),
            length(coefficient_bands(m)), StatsAPI.dof_residual(m) > 0)
        @test summary(f) == summary(fpc)
        # MultivariateFunctionalData input and one-term grids.
        g = StatsAPI.fit(PenalizedFunctionalRegression, MultivariateFunctionalData([x1, x2]), y[1:100];
            scalar_covariates=z[1:100])
        @test StatsAPI.coef(g) ≈ StatsAPI.coef(f)
        single = StatsAPI.fit(PenalizedFunctionalRegression, x1, y[1:100]; criterion=:GCV, k=6)
        @test single.terms[1].k == 6
        @test single.settings.corrected_covariance == :unavailable
        @test only(coefficient_bands(single; grid=grid)).settings.covariance == :bayesian
    end

    @testset "Presmoothing and sparse covariates" begin
        rng = Random.MersenneTwister(77)
        grid = collect(range(0, 1; length=21))
        # Cubic curves are reproduced exactly by B-spline presmoothing.
        coefficients = randn(rng, 40, 4)
        X = coefficients * transpose(hcat(ones(21), grid, grid .^ 2, grid .^ 3))
        y = X * (quadrature_weights(grid) .* cospi.(grid)) .+ 0.1 .* randn(rng, 40)
        x = GridFunctionalData(X, grid)
        f = StatsAPI.fit(PenalizedFunctionalRegression, x, y; presmooth=:bspline,
            presmooth_options=(; nbasis=8))
        @test f.terms[1].values ≈ X rtol=1e-10
        @test nbasis(f.terms[1].presmoother) == 8
        # Linear interpolation with constant ends (R approx(rule=2)).
        masked = Matrix{Union{Missing,Float64}}(X)
        masked[1, [1, 2, 10]] .= missing
        xm = GridFunctionalData(masked, grid)
        @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, xm, y)
        g = StatsAPI.fit(PenalizedFunctionalRegression, xm, y; presmooth=:interpolate)
        @test g.terms[1].values[1, [1, 2]] == [X[1, 3], X[1, 3]]
        @test g.terms[1].values[1, 10] ≈ (X[1, 9] + X[1, 11]) / 2
        @test g.terms[1].values[2:end, :] == X[2:end, :]
        # Sparse covariates use the package's PACE reconstruction.
        times = [sort(rand(rng, 21)[1:6]) for _ in 1:40]
        sparse = IrregularFunctionalData([collect(t) for t in times],
            [[dot(coefficients[i, :], [1, t, t^2, t^3]) for t in times[i]] for i in eachindex(times)])
        @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, sparse, y)
        s = StatsAPI.fit(PenalizedFunctionalRegression, sparse, y; presmooth=:fpca,
            presmooth_options=(; ncomponents=3))
        fpca = s.terms[1].presmoother
        @test fpca isa FPCA
        @test s.terms[1].grid == fpca.grid
        @test s.terms[1].values ≈ reconstruct(fpca).values
        @test s.terms[1].sampling == :sparse
        @test StatsAPI.predict(s, sparse) ≈ StatsAPI.fitted(s) rtol=1e-12
        held = sparse[1:5]
        @test StatsAPI.predict(s, held) ≈ StatsAPI.fitted(s)[1:5] rtol=1e-8
        @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, sparse, y; presmooth=:bspline)
        b = StatsAPI.fit(PenalizedFunctionalRegression, sparse, y; presmooth=:bspline,
            presmooth_options=(; nbasis=4), grid=grid)
        @test b.terms[1].values ≈ X rtol=1e-8
        # Basis covariates are evaluated exactly on the requested grid.
        basis = MonomialBasis((0.0, 1.0), 3)
        bx = BasisFunctionalData(coefficients, basis)
        @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, bx, y)
        h = StatsAPI.fit(PenalizedFunctionalRegression, bx, y; grid=grid)
        @test h.terms[1].values ≈ X rtol=1e-12
        @test StatsAPI.predict(h, bx) ≈ StatsAPI.fitted(h)
        @test StatsAPI.predict(h, BasisFunctionalData(coefficients[1:3, :], basis)) ≈ StatsAPI.fitted(h)[1:3]
    end

    @testset "Input validation" begin
        rng = Random.MersenneTwister(5)
        grid = collect(range(0, 1; length=15))
        X = _pfr_curves(rng, 25, grid)
        x = GridFunctionalData(X, grid)
        y = randn(rng, 25)
        fitp(args...; kw...) = StatsAPI.fit(PenalizedFunctionalRegression, args...; kw...)
        @test_throws DimensionMismatch fitp(x, y[1:24])
        @test_throws ArgumentError fitp(x, fill(1.0, 25))
        @test_throws ArgumentError fitp(x, [missing; y[2:end]])
        @test_throws ArgumentError fitp(x, [NaN; y[2:end]])
        @test_throws ArgumentError fitp(x, string.(y))
        @test_throws DimensionMismatch fitp(x, y; scalar_covariates=randn(24))
        @test_throws ArgumentError fitp(x, y; criterion=:ML)
        @test_throws DimensionMismatch fitp(x, y; lambda=[1.0, 2.0])
        @test_throws ArgumentError fitp(x, y; lambda=[-1.0])
        @test_throws ArgumentError fitp(x, y; integration=:simpson)
        @test_throws DimensionMismatch fitp(x, y; integration=ones(3))
        @test_throws ArgumentError fitp(x, y; integration=[NaN; ones(14)])
        @test_throws ArgumentError fitp(x, y; presmooth=:smooth)
        @test_throws ArgumentError fitp(x, y; presmooth=:bspline, presmooth_options=(; df=5))
        @test_throws ArgumentError fitp(x, y; presmooth=:bspline, presmooth_options=(; nbasis=3))
        @test_throws ArgumentError fitp(x, y; presmooth_options=(; nbasis=5))
        @test_throws ArgumentError fitp(x, y; presmooth=:interpolate, presmooth_options=(; nbasis=5))
        @test_throws ArgumentError fitp(x, y; grid=grid .+ 1)
        @test_throws ArgumentError fitp(x, y; k=2)
        @test_throws ArgumentError fitp(x, y; k=16)
        @test_throws ArgumentError fitp(x, y; k=true)
        @test_throws ArgumentError fitp(x, y; penalty_order=3)
        spline = BSplineBasis(4, (0.0, 1.0), 6)
        @test_throws ArgumentError fitp(x, y; basis=spline, k=5)
        @test_throws ArgumentError fitp(x, y; basis=spline, penalty_order=4)
        @test_throws ArgumentError fitp(x, y; basis=BSplineBasis(4, (0.0, 0.5), 6))
        @test_throws ArgumentError fitp(x, y; basis=:tp)
        @test_throws DimensionMismatch fitp((x, x), y; k=(5, 6, 7))
        # Integrals equal for every subject: confounded with the intercept.
        @test_throws ArgumentError fitp(GridFunctionalData(repeat(transpose(sinpi.(grid)), 25), grid), y)
        f = fitp(x, y)
        @test_throws DimensionMismatch StatsAPI.predict(f, (x, x))
        @test_throws DimensionMismatch StatsAPI.predict(f, x; scalar_covariates=ones(25))
        @test_throws ArgumentError StatsAPI.predict(f, GridFunctionalData(X[:, 1:10], grid[1:10]))
        @test_throws ArgumentError StatsAPI.predict(f, x; interval=:both)
        @test_throws ArgumentError StatsAPI.predict(f, to_irregular(x))
        @test_throws ArgumentError coefficient_bands(f; covariance=:sandwich)
        @test_throws ArgumentError coefficient_bands(f; alpha=1.5)
        @test_throws ArgumentError coefficient_bands(f; grid=[0.0, 2.0])
        @test_throws ArgumentError fitted_bands(f; interval=:none)
        @test_throws ArgumentError StatsAPI.vcov(f; type=:robust)
    end
end
