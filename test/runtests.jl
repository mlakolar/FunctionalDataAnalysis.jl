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
    include("simulation.jl")
    include("smoothing.jl")
    include("mean.jl")
    include("covariance.jl")
    include("fpcbasis.jl")
    include("fpca.jl")
    include("pace.jl")
    include("pace_golden.jl")
    include("pace_defaults.jl")
    include("pace_defaults_golden.jl")
    include("multivariate_fpca.jl")
    include("multivariate_fpca_golden.jl")
    include("cross_covariance.jl")
end
