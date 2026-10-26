using FunctionalDataAnalysis, LinearAlgebra, Random, Test
import StatsAPI

BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","support","prediction.jl"))
include(joinpath(@__DIR__,"..","support","goldens.jl"))
using .GoldenTestHelpers

@testset "Four-thread golden smoke (BLAS 1)" begin
    @test Threads.nthreads() == 4
    @test BLAS.get_num_threads() == 1
    @test get(ENV,"FDA_BANDS_FULL_COVERAGE","false") != "true"
    println("Julia_threads=",Threads.nthreads()," BLAS_threads=",BLAS.get_num_threads())
    # Reuse the existing aligned R comparisons and former-smoother tolerances.
    # No band/bootstrap study, dataset artifacts or ordinary full-suite include.
    for file in ("pace_golden.jl","multivariate_fpca_golden.jl","kernelsmoothing_agreement.jl")
        include(joinpath(@__DIR__,"..",file))
    end
end
