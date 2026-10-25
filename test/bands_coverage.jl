using Test, LinearAlgebra, Random
using FunctionalDataAnalysis
import StatsAPI
import KernelSmoothing

# Recreate the larger study with:
# FDA_BANDS_FULL_COVERAGE=true julia --project=. -e 'include("test/bands_coverage.jl")'
include("support/bands_coverage.jl")

@testset "Gaussian sparse band coverage" begin
    full = get(ENV, "FDA_BANDS_FULL_COVERAGE", "false") == "true"
    result = full ? bands_coverage(; replicates=200, n=300, B=1000) : bands_coverage()
    @info "Band coverage (nominal 0.95; known population inputs for PACE)" result
    # Fitted observations share scores within curves. The chi-square ellipsoid
    # can be conservative over the five-point grid.
    @test abs(result.fitted_pointwise - 0.95) < 0.04
    @test 0.91 < result.fitted_simultaneous <= 1.0
    # The full study found mean coverage about .925 rather than nominal .95.
    # Use a one-sided regression floor of .90, not a claim of nominal coverage:
    # it allows that documented shortfall plus .025 Monte Carlo variation.
    # The dedicated CI study uses 400 independent data sets (MC SE .0109 at .95); the full study
    # uses 200 (SE .0154). Grid points are not independent replicates.
    # The dedicated CI study uses 399 bootstrap draws (full: 1000) to reduce tail-quantile noise.
    @test 0.90 <= result.mean_pointwise <= 1.0
    @test 0.90 <= result.mean_simultaneous <= 1.0
end
