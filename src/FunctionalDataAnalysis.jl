module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs, fit, predict
import KernelSmoothing
import BSplineKit
import Tables
using Artifacts: @artifact_str
import LazyArtifacts
using RecipesBase: @recipe, @series
import RecipesBase
using Random: AbstractRNG, rand, randn, randperm
import Random

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate,
    sampling_design, quadrature_weights
export nbasis, basismatrix, gram, penalty, BSplineBasis, FourierBasis,
    MonomialBasis, DerivativeBasis, derivative, FPCADerivative, MultivariateFPCADerivative
export GridFunctionalData, IrregularFunctionalData, BasisFunctionalData,
    MultivariateFunctionalData, PiecewiseLinear, NoInterpolation, nobs_per_curve,
    ncomponents, component, to_array, to_irregular, to_grid, to_basis
export SmoothedMean, SmoothedCovariance, CurveCV
export CrossCovariance, SmoothedCrossCovariance, cross_covariances
export FSVD, concurrent_correlation, dynamic_correlation
export FPCA, FPCBasis, FVE, AIC, BIC, eigenfunctions, eigvals, scores,
    score_covariances, fve, noise_variance, reconstruct, predict, pace_bandwidth_candidates
export MultivariateFPCA, MFPCA, score_array
export LocalPolynomial, PenalizedSpline, GCV, smooth_curves
export ids, RegularSampling, SparseSampling, simulate_kl,
    fourier_eigenfunctions, legendre_eigenfunctions
export curve_data, mean_data, modes_of_variation, scree_data, covariance_data,
    design_data, fitted_curve_data
export modesplot, screeplot, covplot, designplot, fitplot
export FittedBands, MeanBands, fitted_bands, mean_bands, band_data, bandplot
export datasets, load_dataset

include("interface.jl")
include("quadrature.jl")
include("bases.jl")
include("data.jl")
include("data_interface.jl")
include("evaluation.jl")
include("conversions.jl")
include("multivariate.jl")
include("tables.jl")
include("datasets.jl")
include("simulation.jl")
include("smoothing.jl")
include("mean.jl")
include("smoothing2d.jl")
include("covariance.jl")
include("fpcbasis.jl")
include("fpca.jl")
include("pace_defaults.jl")
include("pace.jl")
include("psplines.jl")
include("face.jl")
include("face_sparse.jl")
include("multivariate_fpca.jl")
include("mfpca.jl")
include("cross_covariance.jl")
include("cross_structure.jl")
include("derivatives.jl")
include("bands.jl")
include("plot_data.jl")
include("recipes.jl")

end
