using Test
using FunctionalDataAnalysis
using LinearAlgebra
import StatsAPI
using DelimitedFiles

_m7f(name) = readdlm(joinpath(@__DIR__, "data", "m7f_$(name).csv"), ',', Any; comments=true)
_m7f_vector(table, quantity) = Float64.(table[table[:, 1] .== quantity, 4])
function _m7f_matrix(table, quantity)
    rows = table[table[:, 1] .== quantity, :]
    out = zeros(maximum(Int.(rows[:, 2])), maximum(Int.(rows[:, 3])))
    for row in eachrow(rows)
        out[Int(row[2]), Int(row[3])] = Float64(row[4])
    end
    out
end
_m7f_relative(actual, expected) = norm(actual - expected) / norm(expected)
# R stores surfaces with the response index varying fastest.
_m7f_surface(values) = vec(permutedims(values))

@testset "Penalized function-on-function regression (refund pffr with ff)" begin
    covariates = readdlm(joinpath(@__DIR__, "data", "m7f_input_covariates.csv"), ','; comments=true)
    responses = readdlm(joinpath(@__DIR__, "data", "m7f_input_responses.csv"), ','; comments=true)
    t = collect(range(0.0, 1.0; length=25))
    s = collect(range(0.0, 1.0; length=21))
    z = covariates[:, 1]
    x = GridFunctionalData(covariates[:, 2:end], t)
    y1 = GridFunctionalData(responses[:, 1:21], s)
    y2 = GridFunctionalData(responses[:, 22:42], s)
    Model = PenalizedFunctionOnFunctionRegression

    @testset "pffr(Y ~ ff(X)) at its REML lambda" begin
        golden = _m7f("pffr_ff")
        ref(q) = _m7f_vector(golden, q)
        lambda = ref("lambda")
        fits = Dict(c => StatsAPI.fit(Model, x, y1; lambda, covariance=c)
            for c in (:bayesian, :frequentist, :cluster))
        f = fits[:bayesian]
        tt, ss = ref("surface_grid"), ref("response_grid")
        bands = coefficient_bands(f; grid=tt, response_grid=ss)
        surface = only(bands.functional)
        @test surface.grid == tt && surface.response_grid == ss
        @test size(surface.estimate) == (length(tt), length(ss))
        @test _m7f_relative(_m7f_surface(surface.estimate), ref("beta")) < 1e-8
        for (c, q) in ((:bayesian, "se_bayesian"), (:frequentist, "se_frequentist"), (:cluster, "se_cluster"))
            band = only(coefficient_bands(fits[c]; grid=tt, response_grid=ss).functional)
            @test _m7f_relative(_m7f_surface(band.se), ref(q)) < 1e-8
            @test fits[c].settings.covariance == c
            @test band.settings.covariance == c
        end
        intercept = only(bands.scalar)
        @test _m7f_relative(intercept.estimate, ref("intercept")) < 1e-8
        @test _m7f_relative(intercept.se, ref("intercept_se_bayesian")) < 1e-8
        cluster_intercept = only(coefficient_bands(fits[:cluster]; response_grid=ss).scalar)
        @test _m7f_relative(cluster_intercept.se, ref("intercept_se_cluster")) < 1e-8
        @test _m7f_relative(f.edf, ref("edf")) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.sigma2 / only(ref("sigma2")) - 1) < 1e-8
        @test StatsAPI.dof_residual(f) ≈ 40 * 21 - StatsAPI.dof(f)
        fitted_curves = StatsAPI.fitted(f)
        @test fitted_curves.grid == s
        @test _m7f_relative(fitted_curves.values, transpose(_m7f_matrix(golden, "fitted"))) < 1e-8
        @test StatsAPI.residuals(f).values ≈ y1.values - fitted_curves.values
        newx = GridFunctionalData(permutedims(_m7f_matrix(golden, "new_covariates")), t)
        prediction = StatsAPI.predict(f, newx; interval=:confidence)
        @test prediction.grid == s
        @test _m7f_relative(prediction.estimate, transpose(_m7f_matrix(golden, "prediction"))) < 1e-8
        @test StatsAPI.predict(f, newx).values ≈ prediction.estimate
        @test StatsAPI.predict(f, newx; interval=:prediction).se .^ 2 ≈ prediction.se .^ 2 .+ f.settings.sigma2
        @test prediction.critical_value ≈ 1.959963984540054 rtol=1e-12
        # Training curves reproduce the fitted values.
        @test StatsAPI.predict(f, x).values ≈ fitted_curves.values
        # The REML score differs from mgcv's by a lambda-free constant (other
        # coordinates for the penalty null space), so differences agree.
        doubled = StatsAPI.fit(Model, x, y1; lambda=2 .* lambda)
        @test abs((doubled.settings.criterion_value - f.settings.criterion_value) /
            (only(ref("reml_double")) - only(ref("reml"))) - 1) < 1e-8
        # Criterion check: the REML optimum here is no worse than at R's lambda.
        selected = @inferred StatsAPI.fit(Model, x, y1)
        @test selected.settings.criterion_value <= f.settings.criterion_value + 1e-12 * abs(f.settings.criterion_value)
        @test selected.settings.lambda_status == fill(:estimated, 3)
        @test selected.settings.solver.convergence == :gradient
        @test selected.settings.lambda_selection == :REML
        @test selected.settings.covariance == :cluster
        @test selected.lambda ≈ lambda rtol=1e-6
        @test f.settings.lambda_selection == :fixed
        @test selected.settings.integration == (:quadrature,)
        @test selected.settings.k == ((5, 5),)
        @test only(selected.terms).reparameterized == (true, true)
        @test nbasis(only(selected.bases)) == 20
        functions = coefficient_functions(f; grid=tt, response_grid=ss)
        @test only(functions.functional).values ≈ surface.estimate
        @test vec(basis_matrix(only(functions.scalar).basis, ss) * vec(only(functions.scalar).coefs)) ≈ intercept.estimate
        defaults = coefficient_bands(f)
        @test only(defaults.functional).grid == t && only(defaults.functional).response_grid == s
        @test length(StatsAPI.coef(f)) == 20 + 25 == size(StatsAPI.vcov(f), 1)
        @test StatsAPI.stderror(f) ≈ sqrt.(diag(StatsAPI.vcov(f)))
        @test StatsAPI.nobs(f) == 40
        displayed = sprint(show, selected)
        for choice in ("REML", "cluster", "lambda", "edf", "pffr", "quadrature", "(5, 5)")
            @test occursin(choice, displayed)
        end
        @test sprint(show, MIME"text/plain"(), selected) == displayed
        # Linear interpolation at the grid points reproduces complete grid data.
        interpolated = StatsAPI.fit(Model, x, y1; lambda, presmooth=:interpolate)
        @test StatsAPI.coef(interpolated) ≈ StatsAPI.coef(f) rtol=1e-12
    end

    @testset "pffr(Y ~ z + ff(X)) with refund's Simpson weights" begin
        golden = _m7f("pffr_ff_scalar")
        ref(q) = _m7f_vector(golden, q)
        tt = _m7f_vector(_m7f("pffr_ff"), "surface_grid")
        ss = _m7f_vector(_m7f("pffr_ff"), "response_grid")
        lambda = ref("lambda")
        f = StatsAPI.fit(Model, x, y2; scalar_covariates=z, integration=:refund_simpson, lambda,
            covariance=:bayesian)
        fc = StatsAPI.fit(Model, x, y2; scalar_covariates=z, integration=:refund_simpson, lambda)
        bands = coefficient_bands(f; grid=tt, response_grid=ss)
        cluster = coefficient_bands(fc; grid=tt, response_grid=ss)
        @test _m7f_relative(_m7f_surface(only(bands.functional).estimate), ref("beta")) < 1e-8
        @test _m7f_relative(_m7f_surface(only(bands.functional).se), ref("se_bayesian")) < 1e-8
        @test _m7f_relative(_m7f_surface(only(cluster.functional).se), ref("se_cluster")) < 1e-8
        @test _m7f_relative(bands.scalar[1].estimate, ref("intercept")) < 1e-8
        @test _m7f_relative(bands.scalar[2].estimate, ref("scalar")) < 1e-8
        @test _m7f_relative(bands.scalar[2].se, ref("scalar_se_bayesian")) < 1e-8
        @test _m7f_relative(cluster.scalar[2].se, ref("scalar_se_cluster")) < 1e-8
        @test _m7f_relative(f.edf, ref("edf")) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.sigma2 / only(ref("sigma2")) - 1) < 1e-8
        @test _m7f_relative(StatsAPI.fitted(f).values, transpose(_m7f_matrix(golden, "fitted"))) < 1e-8
        newx = GridFunctionalData(permutedims(_m7f_matrix(_m7f("pffr_ff"), "new_covariates")), t)
        prediction = StatsAPI.predict(f, newx; scalar_covariates=ref("new_scalar"))
        @test _m7f_relative(prediction.values, transpose(_m7f_matrix(golden, "prediction"))) < 1e-8
        doubled = StatsAPI.fit(Model, x, y2; scalar_covariates=z, integration=:refund_simpson,
            lambda=2 .* lambda)
        @test abs((doubled.settings.criterion_value - f.settings.criterion_value) /
            (only(ref("reml_double")) - only(ref("reml"))) - 1) < 1e-8
        selected = @inferred StatsAPI.fit(Model, x, y2; scalar_covariates=z, integration=:refund_simpson)
        @test selected.settings.criterion_value <= f.settings.criterion_value + 1e-12 * abs(f.settings.criterion_value)
        @test selected.settings.solver.convergence == :gradient
        @test selected.lambda ≈ lambda rtol=1e-6
        @test selected.settings.design_columns == ["(Intercept)", "z1"]
        @test nbasis.(selected.bases) == [20, 5]
        @test selected.settings.integration == (:refund_simpson,)
        # A table gives the same fit, with the column name.
        table_fit = StatsAPI.fit(Model, x, y2; scalar_covariates=(; z), integration=:refund_simpson,
            lambda, covariance=:bayesian)
        @test StatsAPI.coef(table_fit) ≈ StatsAPI.coef(f) rtol=1e-12
        @test table_fit.settings.design_columns == ["(Intercept)", "z"]
        @test StatsAPI.predict(table_fit, newx; scalar_covariates=(; z=ref("new_scalar"))).values ≈ prediction.values
        @test_throws ArgumentError StatsAPI.predict(table_fit, newx; scalar_covariates=ref("new_scalar"))
        @test_throws ArgumentError StatsAPI.predict(f, newx)
        @test_throws DimensionMismatch StatsAPI.predict(f, newx; scalar_covariates=[1.0, 2.0])
        @test_throws DimensionMismatch StatsAPI.predict(f, (newx, newx); scalar_covariates=ref("new_scalar"))
        @test_throws ArgumentError StatsAPI.predict(f, newx; scalar_covariates=ref("new_scalar"), interval=:wide)
        @test_throws ArgumentError StatsAPI.predict(f, newx; scalar_covariates=ref("new_scalar"), grid=[0.5, 2.0])
    end

    @testset "irregular response (pffr ydata)" begin
        golden = _m7f("pffr_ff_sparse")
        ref(q) = _m7f_vector(golden, q)
        tt = _m7f_vector(_m7f("pffr_ff"), "surface_grid")
        ss = _m7f_vector(_m7f("pffr_ff"), "response_grid")
        keep = [(i + j) % 3 != 0 for i in 1:40, j in 1:21]
        times = reduce(vcat, [s[keep[i, :]] for i in 1:40])
        values = reduce(vcat, [y1.values[i, keep[i, :]] for i in 1:40])
        offsets = cumsum(vcat(1, vec(sum(keep; dims=2))))
        sparse = IrregularFunctionalData(times, values, offsets; domain=(0.0, 1.0))
        lambda = ref("lambda")
        f = StatsAPI.fit(Model, x, sparse; lambda, covariance=:bayesian)
        surface = only(coefficient_bands(f; grid=tt, response_grid=ss).functional)
        @test _m7f_relative(_m7f_surface(surface.estimate), ref("beta")) < 1e-8
        @test _m7f_relative(_m7f_surface(surface.se), ref("se_bayesian")) < 1e-8
        @test abs(f.settings.sigma2 / only(ref("sigma2")) - 1) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        fitted_curves = StatsAPI.fitted(f)
        @test fitted_curves isa IrregularFunctionalData
        @test _m7f_relative(fitted_curves.values, ref("fitted")) < 1e-8
        @test StatsAPI.residuals(f).values ≈ values - fitted_curves.values
        @test f.settings.nobservations == length(values)
        @test StatsAPI.predict(f, x).grid == sort(unique(times))
        @test only(coefficient_bands(f).functional).response_grid == collect(range(0.0, 1.0; length=40))
        selected = StatsAPI.fit(Model, x, sparse)
        @test selected.settings.criterion_value <= f.settings.criterion_value + 1e-12 * abs(f.settings.criterion_value)
        # The same points as a grid with missing values give the same fit.
        masked = Matrix{Union{Missing,Float64}}(y1.values)
        masked[.!keep] .= missing
        grid_fit = StatsAPI.fit(Model, x, GridFunctionalData(masked, s); lambda, covariance=:bayesian)
        @test StatsAPI.coef(grid_fit) ≈ StatsAPI.coef(f) rtol=1e-10
        # Grid values in curve order, as the irregular response stores them.
        @test permutedims(StatsAPI.fitted(grid_fit).values)[permutedims(keep)] ≈ fitted_curves.values rtol=1e-10
        @test StatsAPI.residuals(grid_fit).observed == keep
    end

    @testset "options and errors" begin
        lambda = _m7f_vector(_m7f("pffr_ff"), "lambda")
        small = StatsAPI.fit(Model, x, y1; k=(4, 6), lambda=[lambda[1], 1.0, 1.0])
        @test only(small.terms).k == (4, 6) && length(StatsAPI.coef(small)) == 20 + 24
        # A second covariate that is a nonlinear transform of the first, so the
        # two surfaces have distinct integrated design columns.
        x2 = GridFunctionalData(sin.(2 .* x.values) .+ transpose(t), t)
        two = StatsAPI.fit(Model, (x, x2), y1; k=[4, (5, 4)],
            lambda=[lambda[1], 1.0, 1.0, 1.0, 1.0])
        @test two.settings.k == ((4, 4), (5, 4))
        @test [term.k for term in two.terms] == [(4, 4), (5, 4)]
        @test length(StatsAPI.coef(two)) == 20 + 16 + 20
        @test length(coefficient_bands(two).functional) == 2
        gcv = StatsAPI.fit(Model, x, y1; criterion=:GCV)
        @test gcv.settings.criterion == :GCV && gcv.settings.lambda_selection == :GCV
        @test_throws ArgumentError StatsAPI.fit(Model, x, to_basis(y1, BSplineBasis(4, (0.0, 1.0), 8)))
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1.values)
        @test_throws DimensionMismatch StatsAPI.fit(Model, x[1:30], y1)
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; k=3)
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; k=(5, true))
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; k=5.0)
        @test_throws DimensionMismatch StatsAPI.fit(Model, x, y1; k=[5, 5])
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; criterion=:AIC)
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; covariance=:pooled)
        @test_throws DimensionMismatch StatsAPI.fit(Model, x, y1; lambda=[1.0, 1.0])
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; lambda=[1.0, -1.0, 1.0])
        @test_throws ArgumentError StatsAPI.fit(Model, x, y1; integration=:simpson)
        @test_throws DimensionMismatch StatsAPI.fit(Model, x, y1; scalar_covariates=z[1:39])
        # Integrals equal for every curve confound the surface with the intercept.
        flat = GridFunctionalData(repeat(transpose(sinpi.(t)), 40), t)
        @test_throws "confounded with the intercept" StatsAPI.fit(Model, flat, y1)
        # Integrals near zero with a low-rank covariate: refund would constrain
        # the surface (its null space overlaps the t penalty's), which throws.
        u = 1:40
        overlap = GridFunctionalData(cos.(u) .* transpose(sinpi.(2 .* t)) .+
            sin.(u) .* transpose(cospi.(2 .* t)) .+ 1e-9 .* cos.(3 .* u) .* transpose(ones(25)), t)
        @test_throws "not implemented" StatsAPI.fit(Model, overlap, y1)
        f = StatsAPI.fit(Model, x, y1; lambda)
        @test only(f.terms).diagnostics.effective_rank >= 4
        @test only(f.terms).diagnostics.null_overlap < 0.95
        @test_throws ArgumentError coefficient_bands(f; response_grid=[0.0, 2.0])
        @test_throws ArgumentError coefficient_bands(f; grid=[0.5, 0.2])
        @test_throws ArgumentError StatsAPI.predict(f, x; scalar_covariates=z)
        @test_throws ArgumentError StatsAPI.predict(f, GridFunctionalData(x.values[:, 1:20], t[1:20]))
    end
end
