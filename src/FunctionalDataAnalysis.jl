module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate,
    sampling_design, quadrature_weights

include("interface.jl")
include("quadrature.jl")

end
