using Test, LinearAlgebra, Random
using FunctionalDataAnalysis
import StatsAPI
import KernelSmoothing

include("support/bands_coverage.jl")

@testset "Gaussian sparse band coverage smoke" begin
    result = bands_coverage(; replicates=4, n=30, B=19, seed=20261025)
    # Exercise the seeded study pipeline; this sample cannot certify coverage.
    # The dedicated study retains the original statistical acceptance floors.
    for rate in (result.fitted_pointwise, result.fitted_simultaneous,
            result.mean_pointwise, result.mean_simultaneous)
        @test isfinite(rate) && 0 <= rate <= 1
    end
end
