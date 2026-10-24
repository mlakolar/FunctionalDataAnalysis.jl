module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs, fit, predict
import KernelSmoothing
import BSplineKit
import Tables
using Random: AbstractRNG, rand, randn, randperm

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate,
    sampling_design, quadrature_weights
export nbasis, basismatrix, gram, penalty, BSplineBasis, FourierBasis,
    MonomialBasis
export GridFunctionalData, IrregularFunctionalData, BasisFunctionalData,
    MultivariateFunctionalData, PiecewiseLinear, NoInterpolation, nobs_per_curve,
    ncomponents, component, to_array, to_irregular, to_grid, to_basis
export SmoothedMean, SmoothedCovariance, CurveCV
export FPCA, FPCBasis, FVE, AIC, BIC, eigenfunctions, eigvals, scores,
    score_covariances, fve, noise_variance, reconstruct, predict, pace_bandwidth_candidates
export MultivariateFPCA, score_array
export LocalPolynomial, PenalizedSpline, GCV, smooth_curves
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
include("smoothing.jl")
include("mean.jl")
include("smoothing2d.jl")
include("covariance.jl")
include("fpcbasis.jl")
include("fpca.jl")
include("pace_defaults.jl")
include("pace.jl")
include("multivariate_fpca.jl")

end
