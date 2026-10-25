module FunctionalDataAnalysis

using LinearAlgebra
import StatsAPI: nobs, fit, predict, reconstruct, reconstruct!, fitted, residuals, coef
import StatsAPI: vcov, dof, dof_residual, r2, stderror
import KernelSmoothing
import SpecialFunctions
import BSplineKit
import Tables
using Artifacts: @artifact_str
import LazyArtifacts
using RecipesBase: @recipe, @series
import RecipesBase
using Random: AbstractRNG, rand, randn, randperm
import Random

export AbstractFunctionalData, AbstractBasis, nobs, domain, evaluate, evaluate!,
    sampling_design, quadrature_weights
export nbasis, basis_matrix, gram, penalty, BSplineBasis, FourierBasis,
    MonomialBasis, DerivativeBasis, derivative, FPCADerivative, MultivariateFPCADerivative
export GridFunctionalData, IrregularFunctionalData, BasisFunctionalData,
    MultivariateFunctionalData, PiecewiseLinear, NoInterpolation, nobs_per_curve,
    ncomponents, component, to_array, to_irregular, to_grid, to_basis
export SmoothedMean, SmoothedCovariance, CurveCV
export CrossCovariance, SmoothedCrossCovariance, cross_covariances
export FSVD, svdvals, left_functions, right_functions, left_scores, right_scores,
    concurrent_correlation, dynamic_correlation
export FPCA, FPCBasis, FVE, AIC, BIC, eigenfunctions, eigvals, scores,
    score_covariances, fve, noise_variance, reconstruct, reconstruct!, predict, pace_bandwidth_candidates
export MultivariateFPCA, MFPCA, score_array
export ScorePrediction, project_scores, predict_scores, fitted, residuals, coef
export LambdaCV, FunctionalCCA, canonical_weights, canonical_variates,
    canonical_correlations, squared_canonical_correlations
export LocalPolynomial, PenalizedSpline, GCV, REML, smooth_curves
export PenalizedLeastSquares, vcov, dof, dof_residual
export ids, RegularSampling, SparseSampling, simulate_kl,
    fourier_eigenfunctions, legendre_eigenfunctions
export curve_data, mean_data, modes_of_variation, scree_data, covariance_data,
    design_data, fitted_curve_data
export modes_plot, scree_plot, cov_plot, design_plot, fit_plot
export FittedBands, MeanBands, fitted_bands, mean_bands, band_data, band_plot
export datasets, load_dataset
export FPCRegression, coefficient_functions, coefficient_bands, pointwise_r2
export ConcurrentRegression, FunctionOnScalarRegression

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
include("smoothed_fpca.jl")
include("pace_defaults.jl")
include("pace.jl")
include("psplines.jl")
include("penalized.jl")
include("face.jl")
include("face_sparse.jl")
include("multivariate_fpca.jl")
include("mfpca.jl")
include("cross_covariance.jl")
include("cross_structure.jl")
include("derivatives.jl")
include("prediction.jl")
include("buffers.jl")
include("bands.jl")
include("regression.jl")
include("concurrent_regression.jl")
include("function_on_scalar.jl")
include("plot_data.jl")
include("recipes.jl")
include("compat_naming.jl")

end
