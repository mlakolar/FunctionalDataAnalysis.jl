module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs
import BSplineKit
import Tables
using Random: AbstractRNG, rand, randn

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate,
    sampling_design, quadrature_weights
export nbasis, basismatrix, gram, penalty, BSplineBasis, FourierBasis,
    MonomialBasis
export GridFunctionalData, IrregularFunctionalData, BasisFunctionalData,
    MultivariateFunctionalData, PiecewiseLinear, NoInterpolation, nobs_per_curve,
    ncomponents, component, to_array, to_irregular, to_grid, to_basis
export ids, RegularSampling, SparseSampling, simulate_kl,
    fourier_eigenfunctions, legendre_eigenfunctions

include("interface.jl")
include("quadrature.jl")
include("bases.jl")
include("data.jl")
include("data_interface.jl")
include("evaluation.jl")
include("conversions.jl")
include("multivariate.jl")
include("tables.jl")
include("simulation.jl")

end
