# Focused harness checks only; no FPCA fits or benchmark runs.
include("run.jl")
using .ScalingBenchmark
using FunctionalDataAnalysis
using LinearAlgebra
using Test

struct OracleFit
    values::Vector{Float64}
    basis::FPCBasis{Float64}
    reconstruction::Matrix{Float64}
    sigma2::Union{Nothing,Float64}
end

struct BenchmarkTestException <: Exception end

@testset "Scaling benchmark work selection and failures" begin
    config = (; design=:dense, replicates=4)
    @test work_selection(config, 1, nothing, nothing) == (1:4, (:dense, :face))
    @test work_selection(config, 2, 3, :face) == (2:3, (:face,))
    @test work_selection(config, 2, 3, nothing) == (2:3, (:dense, :face))
    @test work_selection((; design=:sparse, replicates=4), 4, 4, :pace) == (4:4, (:pace,))
    for (lo, hi) in ((0, 1), (3, 2), (1, 5))
        @test_throws ArgumentError work_selection(config, lo, hi, :dense)
    end
    @test_throws ArgumentError work_selection(config, 1, 1, :pace)
    prepared = (; simulated=(; data=nothing), grid=nothing)
    valid_metrics = (; eigenvalue_relative_error=0.0, eigenfunction_l2_error=0.0,
        sigma2_error=nothing, ncomponents=3, reconstruction_error=0.0)
    success = run_attempt(:dense, prepared, nothing, 0.1;
        fit_function=(args...) -> :fit, metric_function=(args...) -> valid_metrics,
        tuning_function=fit -> "settings")
    @test success.status == "success"
    @test success.metrics == valid_metrics
    @test success.settings == "settings"
    # These include programming errors and a user-defined exception, outside
    # the former numerical-error allowlist. Every scored attempt is retained.
    for exception in (ArgumentError("bad fit"), DomainError(-1), BoundsError(),
        TypeError(:fit, "input", Int, nothing), ErrorException("fit failed"), BenchmarkTestException())
        fit_failure = run_attempt(:dense, prepared, nothing, 0.1;
            fit_function=(args...) -> throw(exception))
        @test fit_failure.status == "failure"
        @test fit_failure.stage == "fit"
        @test fit_failure.failure === exception
        @test all(isnothing, fit_failure.metrics)
        @test fit_failure.measured.time >= 0
        @test fit_failure.measured.bytes >= 0
        for stage in (:metrics, :tuning)
            metric_failure = run_attempt(:dense, prepared, nothing, 0.1;
                fit_function=(args...) -> :fit,
                metric_function=(args...) -> stage == :metrics ? throw(exception) : valid_metrics,
                tuning_function=fit -> throw(exception))
            @test metric_failure.status == "failure"
            @test metric_failure.stage == "metrics"
            @test metric_failure.failure === exception
            @test all(isnothing, metric_failure.metrics)
            @test isempty(metric_failure.settings)
        end
    end
    for exception in (InterruptException(), OutOfMemoryError())
        @test !ScalingBenchmark.fit_failure(exception)
        @test_throws typeof(exception) run_attempt(:dense, prepared, nothing, 0.1;
            fit_function=(args...) -> throw(exception))
        @test_throws typeof(exception) run_attempt(:dense, prepared, nothing, 0.1;
            fit_function=(args...) -> :fit, metric_function=(args...) -> throw(exception))
    end
    @test !ScalingBenchmark.fit_failure(:nonexception)
end
FunctionalDataAnalysis.ncomponents(f::OracleFit) = length(f.values)
FunctionalDataAnalysis.eigenfunctions(f::OracleFit) = f.basis
LinearAlgebra.eigvals(f::OracleFit) = f.values
FunctionalDataAnalysis.noise_variance(f::OracleFit) = f.sigma2
FunctionalDataAnalysis.reconstruct(f::OracleFit; grid) = (; values=f.reconstruction)

@testset "Scaling benchmark analytic metrics" begin
    grid = collect(range(0.0, 1.0; length=101))
    truth = ScalingBenchmark.population(grid)
    @test truth.values ≈ [1.0, 0.5, 0.25] atol=1e-14
    @test transpose(truth.functions)*(truth.weights .* truth.functions) ≈ I atol=1e-14
    # Analytic Fourier covariance, checked without a grid-square matrix.
    for i in (1, 13, 60, 101), j in (1, 26, 80, 101)
        s, t = grid[i], grid[j]
        expected = 1+sinpi(2s)*sinpi(2t)+0.5cospi(2s)*cospi(2t)
        @test sum(truth.values .* truth.functions[i, :] .* truth.functions[j, :]) ≈ expected atol=1e-14
    end
    latent = [1+t for i in 1:2, t in grid]
    prepared = (; grid, simulated=(; scores=zeros(2, 3)))
    # All signs are reversed; the L2 error must still vanish.
    oracle = OracleFit(truth.values, FPCBasis(grid, -truth.functions), latent, 0.1)
    metrics = ScalingBenchmark.accuracy(oracle, prepared, truth, 0.1)
    @test metrics.eigenvalue_relative_error == 0
    @test metrics.eigenfunction_l2_error < 1e-14
    @test metrics.sigma2_error == 0
    @test metrics.reconstruction_error == 0
    truncated = OracleFit(truth.values[1:1], FPCBasis(grid, truth.functions[:, 1:1]), latent, nothing)
    metrics = ScalingBenchmark.accuracy(truncated, prepared, truth, 0.1)
    @test metrics.eigenvalue_relative_error ≈ sqrt(0.5^2+0.25^2)/sqrt(1+0.5^2+0.25^2)
    @test metrics.eigenfunction_l2_error ≈ 2/3
    @test metrics.sigma2_error === nothing
    mktemp() do path, io
        println(io, join(ScalingBenchmark.CONFIG_HEADER, ','))
        println(io, "1,dense,12,40,0.05,40,40,1,20261025")
        flush(io)
        @test ScalingBenchmark.read_config(path, 1).m == 40
        @test_throws ArgumentError ScalingBenchmark.read_config(path, 0)
        @test_throws ArgumentError ScalingBenchmark.read_config(path, 2)
    end
end
