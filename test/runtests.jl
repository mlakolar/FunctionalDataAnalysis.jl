using Aqua
using FunctionalDataAnalysis
using LinearAlgebra
using Random
using Tables
using Test

@testset "FunctionalDataAnalysis" begin
    include("aqua.jl")
    include("quadrature.jl")
    include("bases.jl")
    include("data.jl")
    include("tables.jl")
    include("datasets.jl")
    include("simulation.jl")
    include("smoothing.jl")
    include("mean.jl")
    include("covariance.jl")
    include("fpcbasis.jl")
    include("fpca.jl")
    include("face.jl")
    include("face_sparse.jl")
    include("pace.jl")
    include("pace_golden.jl")
    include("pace_defaults.jl")
    include("pace_defaults_golden.jl")
    include("multivariate_fpca.jl")
    include("multivariate_fpca_golden.jl")
    include("mfpca.jl")
    include("cross_covariance.jl")
    include("cross_structure.jl")
    include("plotting.jl")
    include("derivatives.jl")
    include("bands.jl")
    include("bands_coverage.jl")
    include("kernelsmoothing_agreement.jl")
end
