module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs
import BSplineKit

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate,
    sampling_design, quadrature_weights
export nbasis, basismatrix, gram, penalty, BSplineBasis, FourierBasis,
    MonomialBasis

include("interface.jl")
include("quadrature.jl")
include("bases.jl")

end
