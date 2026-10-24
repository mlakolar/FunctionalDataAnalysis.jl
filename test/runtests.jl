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
end
