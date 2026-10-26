# Focused harness checks only; no regression fits or study runs.
include("run.jl")
using .RegressionStudy
using FunctionalDataAnalysis
using LinearAlgebra
using Test
import StatsAPI

# A fitted-model stand-in with a known coefficient band and predictions.
struct OracleFit
    grid::Vector{Float64}
    estimate::Vector{Float64}
    halfwidth::Float64
    predictions::Vector{Float64}
end

FunctionalDataAnalysis.coefficient_bands(f::OracleFit; grid, alpha) =
    ((; grid, estimate=f.estimate, lower=f.estimate .- f.halfwidth,
        upper=f.estimate .+ f.halfwidth, settings=(; covariance=:oracle)),)
StatsAPI.predict(f::OracleFit, x) = f.predictions
RegressionStudy.term_dof(f::OracleFit) = 4.5
RegressionStudy.fitted_lambda(f::OracleFit) = 2.0

struct StudyTestException <: Exception end

@testset "Regression study work selection and failures" begin
    config = (; replicates=4)
    @test work_selection(config, 1, nothing, nothing) == (1:4, (:fpc, :fpc_bic, :pfr))
    @test work_selection(config, 2, 3, :pfr) == (2:3, (:pfr,))
    for (lo, hi) in ((0, 1), (3, 2), (1, 5))
        @test_throws ArgumentError work_selection(config, lo, hi, :fpc)
    end
    @test_throws ArgumentError work_selection(config, 1, 1, :pace)
    prepared = (; train=nothing, y=nothing)
    valid = (; metrics=(; beta_relative_l2_error=0.1, prediction_relative_error=0.2,
        pointwise_coverage=1.0, whole_curve_coverage=1, band_mean_width=0.3,
        ncomponents=nothing, term_dof=3.0), lambda=1.0, band_covariance="corrected")
    success = run_attempt(:pfr, :dense, prepared, nothing;
        fit_function=(args...) -> :fit, metric_function=(args...) -> valid,
        tuning_function=fit -> "settings")
    @test success.status == "success"
    @test success.metrics == valid.metrics
    @test success.lambda == 1.0
    @test success.settings == "settings"
    # Every Exception is a retained, counted failure, including programming errors.
    for exception in (ArgumentError("bad fit"), DomainError(-1), BoundsError(),
        ErrorException("fit failed"), StudyTestException())
        fit_failure = run_attempt(:fpc, :sparse, prepared, nothing;
            fit_function=(args...) -> throw(exception))
        @test fit_failure.status == "failure"
        @test fit_failure.stage == "fit"
        @test fit_failure.failure === exception
        @test all(isnothing, fit_failure.metrics)
        @test fit_failure.measured.time >= 0
        for stage in (:metrics, :tuning)
            metric_failure = run_attempt(:fpc, :sparse, prepared, nothing;
                fit_function=(args...) -> :fit,
                metric_function=(args...) -> stage == :metrics ? throw(exception) : valid,
                tuning_function=fit -> throw(exception))
            @test metric_failure.status == "failure"
            @test metric_failure.stage == "metrics"
            @test all(isnothing, metric_failure.metrics)
            @test metric_failure.lambda === nothing
            @test isempty(metric_failure.settings)
        end
    end
    for exception in (InterruptException(), OutOfMemoryError())
        @test !RegressionStudy.fit_failure(exception)
        @test_throws typeof(exception) run_attempt(:fpc, :dense, prepared, nothing;
            fit_function=(args...) -> throw(exception))
    end
end

@testset "Regression study population and metrics" begin
    # Orthonormal Fourier coordinates are exact for the two eigenfunction designs.
    leading = RegressionStudy.truth(:leading)
    @test leading.projection ≈ [1, 1, 0, 0, 0, 0, 0, 0, 0] atol=1e-10
    @test leading.signal_variance ≈ 1 + 1/4 atol=1e-10
    @test leading.offset ≈ 1.5 - 1/(sqrt(2) * pi) atol=1e-8   # ∫(1+t)(1+√2 sin 2πt)
    @test leading.unidentified_norm < 1e-4
    trailing = RegressionStudy.truth(:trailing)
    @test trailing.projection ≈ [0, 0, 0, 0, 0, 0, 0, 0, 1] atol=1e-10
    @test trailing.signal_variance ≈ 1/81 atol=1e-12
    @test trailing.beta_norm ≈ 1 atol=1e-8
    smooth = RegressionStudy.truth(:smooth)
    # ∫exp(-(t-0.3)²/(2σ²)) over [0,1], σ = 0.1, via the error function.
    gaussian_mass = 0.1 * sqrt(pi / 2) * (FunctionalDataAnalysis.SpecialFunctions.erf(0.7 / (0.1 * sqrt(2))) +
        FunctionalDataAnalysis.SpecialFunctions.erf(0.3 / (0.1 * sqrt(2))))
    @test smooth.projection[1] ≈ gaussian_mass atol=1e-8
    @test smooth.beta_norm^2 ≈ sum(abs2, smooth.projection) + smooth.unidentified_norm^2 atol=1e-10
    @test 0 < smooth.unidentified_norm < 0.05 * smooth.beta_norm

    grid = collect(range(0.0, 1.0; length=101))
    target = leading.beta.(grid)
    test_mean = [0.5, -1.0, 2.0]
    prepared = (; grid, test=nothing, test_mean)
    exact = RegressionStudy.accuracy(OracleFit(grid, target, 0.5, test_mean), prepared, leading)
    @test exact.metrics.beta_relative_l2_error == 0
    @test exact.metrics.prediction_relative_error == 0
    @test exact.metrics.pointwise_coverage == 1
    @test exact.metrics.whole_curve_coverage == 1
    @test exact.metrics.band_mean_width ≈ 1
    @test exact.metrics.ncomponents === nothing
    @test exact.metrics.term_dof == 4.5
    @test exact.lambda == 2.0
    @test exact.band_covariance == "oracle"
    # Shift by a constant c: relative L2 error c/‖β‖, band misses everywhere.
    shifted = RegressionStudy.accuracy(OracleFit(grid, target .+ 1, 0.5, test_mean .+ 0.5),
        prepared, leading)
    weights = quadrature_weights(grid)
    @test shifted.metrics.beta_relative_l2_error ≈ 1 / sqrt(sum(weights .* abs2.(target)))
    @test shifted.metrics.prediction_relative_error ≈ 0.25 / leading.signal_variance
    @test shifted.metrics.pointwise_coverage == 0
    @test shifted.metrics.whole_curve_coverage == 0
    # Covering exactly the first half of the grid gives coverage 51/101.
    partial = copy(target)
    partial[52:end] .+= 1
    covered = RegressionStudy.accuracy(OracleFit(grid, partial, 0.5, test_mean), prepared, leading)
    @test covered.metrics.pointwise_coverage == 51 / 101
    @test covered.metrics.whole_curve_coverage == 0
    @test_throws ArgumentError RegressionStudy.accuracy(
        OracleFit(grid, target, 0.5, [NaN, 0.0, 0.0]), prepared, leading)

    mktemp() do path, io
        println(io, join(RegressionStudy.CONFIG_HEADER, ','))
        println(io, "1,sparse,smooth,40,0.5,20,2,20262026")
        flush(io)
        config = RegressionStudy.read_config(path, 1)
        @test config.beta == :smooth
        @test_throws ArgumentError RegressionStudy.read_config(path, 0)
        @test_throws ArgumentError RegressionStudy.read_config(path, 2)
        # Paired routes see the same sample; sparse test times stay in range.
        a = RegressionStudy.simulation(config, 2)
        b = RegressionStudy.simulation(config, 2)
        @test a.y == b.y && a.test_mean == b.test_mean && a.grid == b.grid
        @test extrema(a.test.times)[1] >= a.grid[1] && extrema(a.test.times)[2] <= a.grid[end]
        @test length(a.y) == 40 && length(a.test_mean) == 20
    end
end
