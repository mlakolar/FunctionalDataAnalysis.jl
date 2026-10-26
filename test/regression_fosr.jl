using Test
using FunctionalDataAnalysis
using LinearAlgebra
import Random
import StatsAPI
using DelimitedFiles

_m7e(name) = readdlm(joinpath(@__DIR__, "data", "m7e_$(name).csv"), ',', Any; comments=true)
_m7e_vector(table, quantity) = Float64.(table[table[:, 1] .== quantity, 4])
function _m7e_matrix(table, quantity)
    rows = table[table[:, 1] .== quantity, :]
    out = zeros(maximum(Int.(rows[:, 2])), maximum(Int.(rows[:, 3])))
    for row in eachrow(rows)
        out[Int(row[2]), Int(row[3])] = Float64(row[4])
    end
    out
end
_m7e_relative(actual, expected) = norm(actual - expected) / norm(expected)
_m7e_columns(bands, field) = reduce(hcat, [getfield(band, field) for band in bands])

@testset "Penalized function-on-scalar regression (refund pffr and fosr)" begin
    input = readdlm(joinpath(@__DIR__, "data", "m7e_input.csv"), ','; comments=true)
    z1 = input[:, 1]
    level = ["a", "b", "c"][Int.(input[:, 2])]
    grid = collect(range(0.0, 1.0; length=31))
    y = GridFunctionalData(input[:, 3:end], grid)
    table = (; z1, g=level)
    basis = BSplineBasis(4, (0.0, 1.0), 10)

    @testset "pffr with scalar terms at its REML lambda" begin
        golden = _m7e("pffr")
        ref(q) = _m7e_vector(golden, q)
        mat(q) = _m7e_matrix(golden, q)
        lambda = ref("lambda")
        fits = Dict(c => StatsAPI.fit(FunctionOnScalarRegression, table, y; basis, lambda, covariance=c)
            for c in (:bayesian, :frequentist, :cluster))
        f = fits[:bayesian]
        bands = coefficient_bands(f)
        @test bands[1].grid == grid
        @test _m7e_relative(_m7e_columns(bands, :estimate), mat("beta")) < 1e-8
        for (c, q) in ((:bayesian, "se_bayesian"), (:frequentist, "se_frequentist"), (:cluster, "se_cluster"))
            @test _m7e_relative(_m7e_columns(coefficient_bands(fits[c]), :se), mat(q)) < 1e-8
            @test fits[c].settings.covariance == c
        end
        @test _m7e_relative(f.edf, ref("edf")) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.sigma2 / only(ref("sigma2")) - 1) < 1e-8
        @test _m7e_relative(StatsAPI.fitted(f).values, transpose(mat("fitted"))) < 1e-8
        @test StatsAPI.dof_residual(f) ≈ 40 * 31 - StatsAPI.dof(f)
        codes = mat("new_covariates")
        newtable = (; z1=codes[:, 1], g=["a", "b", "c"][Int.(codes[:, 2])])
        prediction = StatsAPI.predict(f, newtable; interval=:confidence)
        @test _m7e_relative(prediction.estimate, transpose(mat("prediction"))) < 1e-8
        @test StatsAPI.predict(f, newtable).coefs * transpose(basis_matrix(basis, grid)) ≈ prediction.estimate
        new_prediction = StatsAPI.predict(f, newtable; interval=:prediction)
        @test new_prediction.se .^ 2 ≈ prediction.se .^ 2 .+ f.settings.sigma2
        @test prediction.critical_value ≈ 1.959963984540054 rtol=1e-12
        # Criterion check: the REML optimum here is no worse than at R's lambda.
        selected = @inferred StatsAPI.fit(FunctionOnScalarRegression, table, y; basis)
        @test selected.settings.criterion_value <= f.settings.criterion_value
        @test selected.settings.lambda_status == fill(:estimated, 4)
        @test selected.settings.solver.convergence == :gradient
        @test selected.settings.lambda_selection == :REML
        @test selected.settings.covariance == :cluster
        @test selected.lambda ≈ lambda rtol=1e-4
        @test f.settings.lambda_selection == :fixed
        @test selected.settings.design_columns == ["(Intercept)", "z1", "gb", "gc"]
        # The same design as a matrix of treatment-coded columns.
        X = hcat(z1, level .== "b", level .== "c")
        matrix_fit = StatsAPI.fit(FunctionOnScalarRegression, X, y; basis, lambda, covariance=:bayesian)
        @test StatsAPI.coef(matrix_fit) ≈ StatsAPI.coef(f) rtol=1e-12
        @test StatsAPI.vcov(matrix_fit) ≈ StatsAPI.vcov(f) rtol=1e-12
        @test_throws ArgumentError StatsAPI.predict(f, X)
        @test_throws ArgumentError StatsAPI.predict(matrix_fit, newtable)
        @test_throws ArgumentError StatsAPI.predict(f, (; z1=[0.0], g=["d"]))
        @test_throws ArgumentError StatsAPI.predict(f, (; z1=[0.0]))
        @test_throws ArgumentError StatsAPI.predict(f, (; z1=["x"], g=["a"]))
        displayed = sprint(show, selected)
        for choice in ("REML", "cluster", "lambda", "edf", "pffr")
            @test occursin(choice, displayed)
        end
        @test sprint(show, MIME"text/plain"(), selected) == displayed
    end

    @testset "pffr with its default P-spline bases (basis=:pffr)" begin
        golden = _m7e("pffr_default")
        ref(q) = _m7e_vector(golden, q)
        mat(q) = _m7e_matrix(golden, q)
        lambda = ref("lambda")
        fits = Dict(c => StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=:pffr, lambda, covariance=c)
            for c in (:bayesian, :cluster))
        f = fits[:bayesian]
        @test f.bases[1] == PSplineBasis((0.0, 1.0), 20; degree=3, difference_order=1)
        @test all(==(PSplineBasis((0.0, 1.0), 5; degree=3, difference_order=1)), f.bases[2:end])
        @test f.settings.coefficient_basis == :pffr_default
        @test f.penalty_order == fill(1, 4)
        @test length(StatsAPI.coef(f)) == 35
        @test _m7e_relative(_m7e_columns(coefficient_bands(f), :estimate), mat("beta")) < 1e-8
        @test _m7e_relative(_m7e_columns(coefficient_bands(f), :se), mat("se_bayesian")) < 1e-8
        @test _m7e_relative(_m7e_columns(coefficient_bands(fits[:cluster]), :se), mat("se_cluster")) < 1e-8
        @test _m7e_relative(f.edf, ref("edf")) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.sigma2 / only(ref("sigma2")) - 1) < 1e-8
        @test _m7e_relative(StatsAPI.fitted(f).values, transpose(mat("fitted"))) < 1e-8
        codes = _m7e_matrix(_m7e("pffr"), "new_covariates")
        newtable = (; z1=codes[:, 1], g=["a", "b", "c"][Int.(codes[:, 2])])
        # Different coefficient bases: predictions are curves on a grid.
        curves = StatsAPI.predict(f, newtable)
        @test curves isa GridFunctionalData
        @test curves.grid == grid
        @test _m7e_relative(curves.values, transpose(mat("prediction"))) < 1e-8
        @test StatsAPI.predict(f, newtable; interval=:confidence).estimate ≈ curves.values
        @test StatsAPI.predict(f, newtable; grid=grid[1:2:end]).values ≈ curves.values[:, 1:2:end]
        functions = coefficient_functions(f)
        @test nbasis.(getfield.(functions, :basis)) == (20, 5, 5, 5)
        # Criterion check: the REML optimum here is no worse than at R's lambda
        # (mgcv reaches the same optimum, up to rounding of the score).
        selected = @inferred StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=:pffr)
        @test selected.settings.criterion_value <= f.settings.criterion_value * (1 + 1e-12)
        @test selected.settings.lambda_status == fill(:estimated, 4)
        @test selected.lambda ≈ lambda rtol=1e-4
        @test occursin("bases=", sprint(show, selected))
        # A tuple of bases (one per coefficient) gives the same fit.
        tuple_fit = StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=Tuple(f.bases), lambda,
            covariance=:bayesian)
        @test StatsAPI.coef(tuple_fit) ≈ StatsAPI.coef(f) rtol=1e-12
        @test tuple_fit.settings.coefficient_basis == :supplied
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, table, y;
            basis=Tuple(f.bases[1:3]))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, y;
            basis=(f.bases[1:3]..., PSplineBasis((0.0, 2.0), 5)))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=:mgcv)
        yb = BasisFunctionalData(randn(Random.Xoshiro(3), 40, 6), BSplineBasis(4, (0.0, 1.0), 6))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, yb; basis=:pffr)
        # mgcv takes the knot range from the observation times, not the domain.
        wide = IrregularFunctionalData([grid for _ in 1:40], [y.values[i, :] for i in 1:40];
            domain=(-1.0, 2.0))
        @test StatsAPI.fit(FunctionOnScalarRegression, table, wide; basis=:pffr, lambda).bases == f.bases
    end

    @testset "fosr on a grid at its GCV lambda" begin
        golden = _m7e("fosr_grid")
        ref(q) = _m7e_vector(golden, q)
        mat(q) = _m7e_matrix(golden, q)
        lambda = only(ref("lambda"))
        f = StatsAPI.fit(FunctionOnScalarRegression, table, y; basis, lambda, shared_lambda=true,
            criterion=:GCV, covariance=:pooled)
        bands = coefficient_bands(f)
        @test _m7e_relative(_m7e_columns(bands, :estimate), mat("beta")) < 1e-8
        @test _m7e_relative(_m7e_columns(bands, :se), mat("se_pooled")) < 1e-8
        @test _m7e_relative(StatsAPI.fitted(f).values, transpose(mat("fitted"))) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.criterion_value / only(ref("gcv")) - 1) < 1e-8
        @test f.lambda == fill(lambda, 4)
        @test f.settings.lambda_structure == :shared
        # The residual covariance is fosr's ((n-1)/n) cov(residual curves).
        residual = StatsAPI.residuals(f).values
        centered = residual .- sum(residual; dims=1) ./ 40
        @test f.residual_covariance ≈ transpose(centered) * centered ./ 40
        @test size(f.residual_covariance) == (31, 31)
        selected = StatsAPI.fit(FunctionOnScalarRegression, table, y; basis, shared_lambda=true,
            criterion=:GCV, covariance=:pooled)
        # mgcv reaches the same optimum here, up to rounding of the score.
        @test selected.settings.criterion_value <= f.settings.criterion_value * (1 + 1e-12)
        @test selected.lambda[1] ≈ lambda rtol=1e-4
        fixed = StatsAPI.fit(FunctionOnScalarRegression, table, y; basis, lambda=0.05, shared_lambda=true,
            covariance=:pooled)
        fixed_bands = coefficient_bands(fixed)
        @test _m7e_relative(_m7e_columns(fixed_bands, :estimate), mat("fixed_beta")) < 1e-8
        @test _m7e_relative(_m7e_columns(fixed_bands, :se), mat("fixed_se_pooled")) < 1e-8
    end

    @testset "fosr with a basis response at its GCV lambda" begin
        golden = _m7e("fosr_basis")
        ref(q) = _m7e_vector(golden, q)
        mat(q) = _m7e_matrix(golden, q)
        response_basis = BSplineBasis(4, (0.0, 1.0), 12)
        yb = BasisFunctionalData(mat("coefs"), response_basis)
        lambda = only(ref("lambda"))
        f = StatsAPI.fit(FunctionOnScalarRegression, table, yb; lambda, shared_lambda=true,
            criterion=:GCV, covariance=:pooled, method=:penalized)
        bands = coefficient_bands(f; grid=ref("grid"))
        @test _m7e_relative(_m7e_columns(bands, :estimate), mat("beta")) < 1e-8
        @test _m7e_relative(_m7e_columns(bands, :se), mat("se_pooled")) < 1e-8
        @test _m7e_relative(StatsAPI.fitted(f).coefs, mat("fitted_coefs")) < 1e-8
        @test abs(StatsAPI.dof(f) / only(ref("edf_total")) - 1) < 1e-8
        @test abs(f.settings.criterion_value / only(ref("gcv")) - 1) < 1e-8
        @test f.bases == fill(response_basis, 4)
        @test f.settings.loss == :L2_exact_gram
        @test f.settings.coefficient_basis == :response_basis
        @test size(f.residual_covariance) == (12, 12)
        @test length(coefficient_bands(f)[1].grid) == 121
        selected = StatsAPI.fit(FunctionOnScalarRegression, table, yb; shared_lambda=true,
            criterion=:GCV, covariance=:pooled)
        @test selected.settings.criterion_value <= f.settings.criterion_value * (1 + 1e-12)
        @test selected.settings.estimator == :penalized_least_squares
        # Default REML fit of a basis response, and the fixed-lambda fRegress path.
        @test StatsAPI.fit(FunctionOnScalarRegression, table, yb).settings.lambda_selection == :REML
        fregress = StatsAPI.fit(FunctionOnScalarRegression, table, yb; lambda=fill(lambda, 4))
        @test fregress.settings.estimator == :fRegress
        @test StatsAPI.coef(fregress) ≈ StatsAPI.coef(f) rtol=1e-10
        @test fregress.edf ≈ f.edf rtol=1e-10
    end

    @testset "Irregular and incomplete responses" begin
        times = [grid[1:2:end] for _ in 1:40]
        values = [input[i, 3:2:end] for i in 1:40]
        irregular = IrregularFunctionalData(times, values)
        dense = GridFunctionalData(input[:, 3:2:end], grid[1:2:end])
        f_irregular = StatsAPI.fit(FunctionOnScalarRegression, table, irregular; basis)
        f_dense = StatsAPI.fit(FunctionOnScalarRegression, table, dense; basis)
        @test StatsAPI.coef(f_irregular) ≈ StatsAPI.coef(f_dense) rtol=1e-8
        @test StatsAPI.vcov(f_irregular) ≈ StatsAPI.vcov(f_dense) rtol=1e-8
        @test StatsAPI.fitted(f_irregular) isa IrregularFunctionalData
        @test StatsAPI.fitted(f_irregular).values ≈ vec(transpose(StatsAPI.fitted(f_dense).values))
        @test StatsAPI.residuals(f_irregular).values ≈ irregular.values - StatsAPI.fitted(f_irregular).values
        @test size(f_irregular.residual_covariance) == (0, 0)
        @test f_irregular.settings.residual_covariance == :not_estimated
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, irregular; basis,
            covariance=:pooled)
        # Missing grid values are absent observations; the same points as an
        # irregular dataset give the same fit.
        values = Matrix{Union{Missing,Float64}}(input[:, 3:end])
        for i in 1:40
            values[i, mod1(3i, 31)] = missing
        end
        incomplete = GridFunctionalData(values, grid)
        f_incomplete = StatsAPI.fit(FunctionOnScalarRegression, table, incomplete; basis)
        f_points = StatsAPI.fit(FunctionOnScalarRegression, table, to_irregular(incomplete); basis)
        @test StatsAPI.coef(f_incomplete) ≈ StatsAPI.coef(f_points) rtol=1e-8
        residual = StatsAPI.residuals(f_incomplete)
        @test residual.observed == incomplete.observed
        @test StatsAPI.fitted(f_incomplete).observed === nothing
        @test size(f_incomplete.residual_covariance) == (0, 0)
        @test StatsAPI.fit(FunctionOnScalarRegression, table, y).bases == fill(BSplineBasis(4, (0.0, 1.0), 15), 4)
        # The last grid point missing for every curve: a supplied basis must
        # still cover the whole grid, and basis=:pffr builds its knots on it.
        values = Matrix{Union{Missing,Float64}}(input[:, 3:end])
        values[:, end] .= missing
        open_end = GridFunctionalData(values, grid)
        narrow = BSplineBasis(4, (0.0, grid[end-1]), 10)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, open_end; basis=narrow)
        error = try
            StatsAPI.fit(FunctionOnScalarRegression, table, open_end; basis=narrow)
        catch e
            e
        end
        @test occursin("response grid", error.msg)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=narrow)
        f_open = StatsAPI.fit(FunctionOnScalarRegression, table, open_end; basis=:pffr, lambda=ones(4))
        @test f_open.bases == StatsAPI.fit(FunctionOnScalarRegression, table, y; basis=:pffr, lambda=ones(4)).bases
        @test size(StatsAPI.fitted(f_open).values) == (40, 31)
        @test coefficient_bands(f_open)[1].grid == grid
    end

    @testset "Analytic penalized fits and validation" begin
        rng = Random.Xoshiro(11)
        fourier = FourierBasis((0.0, 2.0), 5)
        z = randn(rng, 12, 2)
        Y = randn(rng, 12, 5)
        yb = BasisFunctionalData(Y, fourier)
        X = hcat(ones(12), z)
        # An orthonormal basis with no penalty: ordinary least squares per coefficient.
        ols = StatsAPI.fit(FunctionOnScalarRegression, z, yb; lambda=0, method=:penalized)
        @test reshape(StatsAPI.coef(ols), 5, 3) ≈ transpose(X \ Y)
        @test ols.edf ≈ fill(5.0, 3)
        @test ols.penalty_order == fill(:harmonic_acceleration, 3)
        # Without an intercept the first factor keeps every level.
        groups = repeat(["u", "v", "w"], 4)
        noint = StatsAPI.fit(FunctionOnScalarRegression, (; g=groups, x=z[:, 1]), yb;
            intercept=false, lambda=0, method=:penalized)
        @test noint.settings.design_columns == ["gu", "gv", "gw", "x"]
        D = hcat(groups .== "u", groups .== "v", groups .== "w", z[:, 1])
        @test reshape(StatsAPI.coef(noint), 5, 4) ≈ transpose(D \ Y)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, (; g=fill("u", 12)), yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, (; g=[missing; groups[2:end]]), yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, (; g=Any[1, "u", 2, "v", 1, "u", 2, "v", 1, "u", 2, "v"]), yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, NamedTuple(), yb)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, (; x=z[1:11, 1]), yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, (; x=[NaN; z[2:end, 1]]), yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, :z, yb)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; covariance=:robust)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; residual_covariance=:residual)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; basis=fourier)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; method=:ols)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; method=:fRegress)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; lambda=1, covariance=:bayesian)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; criterion=:AIC)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionOnScalarRegression, z, yb; lambda=[1, 2],
            shared_lambda=true, method=:penalized)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, yb; lambda=-1, method=:penalized)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y[1:12]; basis=:bspline)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, y[1:12];
            basis=BSplineBasis(4, (0.0, 0.5), 6))
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, z)
        # Exactly fitted responses leave no error variance: REML and the
        # covariance at a fixed lambda both throw.
        exact = BasisFunctionalData(X * transpose(reshape(StatsAPI.coef(ols), 5, 3)), fourier)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, exact; lambda=0, method=:penalized)
        @test_throws ArgumentError StatsAPI.fit(FunctionOnScalarRegression, z, exact)
    end
end
