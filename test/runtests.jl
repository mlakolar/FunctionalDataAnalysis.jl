using Aqua
using FunctionalDataAnalysis
using LinearAlgebra
using Random
using Test

@testset "FunctionalDataAnalysis" begin
    include("aqua.jl")
    include("quadrature.jl")
    include("bases.jl")
end
