"""
    ScorePrediction(scores, covariances)

Score prediction returned by [`predict_scores`](@ref). `scores` is a subject ×
component matrix. `covariances` is a component × component × subject array,
or `nothing` when this route supplies no conditional score covariance.
`nothing` does not assert zero uncertainty. No array conversion or iteration
implicitly discards uncertainty.

The constructor adopts its arrays. Predictions contain newly computed arrays
independent of the fit and input data; use `deepcopy` for another independent
result. The fields themselves expose the result's stored arrays.
"""
struct ScorePrediction{T<:AbstractFloat,C<:Union{Nothing,Array{T,3}}}
    scores::Matrix{T}
    covariances::C
end

_prediction_scores(s::AbstractMatrix) = s
_prediction_scores(s::NamedTuple) = s.scores
_score_prediction(s::Matrix) = ScorePrediction(s, nothing)
_score_prediction(s::NamedTuple) = ScorePrediction(s.scores, s.covariances)

"""
    project_scores(model, newdata)

Return score matrices using the fitted mean, basis, preprocessing and score
solver, without refitting. FPCA and joint MFPCA return `n × K` matrices;
nodewise `MultivariateFPCA` returns a vector of `n × K_j` matrices in node
order. Matrices are newly computed, independent of training scores and input
arrays. Modifying them does not change the model.

Dense FPCA requires complete data on its training grid; basis FPCA requires
its exact training basis. PACE and sparse FACE accept grid or irregular
observations inside the work grid. Common-grid FACE requires its training
grid and permits masks. Multivariate nodes follow these same restrictions.
For conditional score covariance use [`predict_scores`](@ref).
"""
project_scores(f::Union{FPCA,MFPCA}, fd) = _prediction_scores(_predict_result(f, fd))
project_scores(f::MultivariateFPCA, fd::MultivariateFunctionalData) =
    map(_prediction_scores, _predict_result(f, fd))

"""
    predict_scores(model, newdata)

Return a [`ScorePrediction`](@ref), or a vector of these in node order for
`MultivariateFPCA`. Its scores equal [`project_scores`](@ref). PACE and both
FACE routes supply their existing conditional score covariance arrays;
dense/basis FPCA and joint MFPCA supply `nothing`. Joint MFPCA does not assume
independent node errors or propagate marginal covariances into joint scores.

Conditional covariance excludes mean/covariance/noise estimation uncertainty;
common-grid FACE's completed training data also exclude imputation uncertainty.
The data restrictions are those of `project_scores`. Result arrays are newly
computed and independent of the fit and input data.
"""
predict_scores(f::Union{FPCA,MFPCA}, fd) = _score_prediction(_predict_result(f, fd))
predict_scores(f::MultivariateFPCA, fd::MultivariateFunctionalData) =
    map(_score_prediction, _predict_result(f, fd))

function _deprecated_score_prediction()
    Base.depwarn("predict for FPCA, MultivariateFPCA and MFPCA is deprecated; use project_scores for score matrices or predict_scores for ScorePrediction results. Legacy return values are unchanged.", :predict)
    nothing
end

"""
    fitted(model::Union{FPCA,MultivariateFPCA,MFPCA})

Return the latent training curves as `reconstruct(model)`, with its existing
functional-data type, work grids and IDs. The result owns its storage and
contains no added observation noise or parameter uncertainty. No smoother is
refitted. Exact-basis FPCA is evaluated exactly on its stored display grid.
This extends `StatsAPI.fitted`; FSVD, CCA and derivative summaries have no
fitted-observation method.
"""
fitted(f::Union{FPCA,MultivariateFPCA,MFPCA}) = reconstruct(f)

function _observation_residuals(fd::GridFunctionalData, mu, b, projected)
    if fd.observed === nothing
        latent = projected * transpose(basis_matrix(b, fd.grid)) .+ _fpca_mean_values(mu, fd.grid)
        return GridFunctionalData(fd.values .- latent, fd.grid; ids=fd.ids, metadata=fd.metadata)
    end
    # Unobserved grid columns may lie outside the fitted work grid. Evaluate
    # only actual observations, including single-point curves.
    T = promote_type(eltype(fd), eltype(mu), eltype(projected))
    values = Matrix{Union{Missing,T}}(missing, size(fd.values))
    for i in 1:nobs(fd)
        selected = findall(@view fd.observed[i,:])
        times = fd.grid[selected]
        latent = basis_matrix(b, times) * projected[i,:] .+ vec(_fpca_mean_values(mu, times))
        values[i,selected] = fd.values[i,selected] .- latent
    end
    GridFunctionalData(values, fd.grid; ids=fd.ids, metadata=fd.metadata)
end

function _observation_residuals(fd::IrregularFunctionalData, mu, b, projected)
    T = promote_type(eltype(fd), eltype(mu), eltype(projected))
    values = Vector{T}(undef, length(fd.values))
    for i in 1:nobs(fd)
        times, observed = fd[i]
        latent = basis_matrix(b, times) * projected[i,:] .+ vec(_fpca_mean_values(mu, times))
        values[fd.offsets[i]:fd.offsets[i+1]-1] = observed .- latent
    end
    IrregularFunctionalData(fd.times, values, fd.offsets; domain=domain(fd), ids=fd.ids)
end

function _observation_residuals(fd::BasisFunctionalData, mu::BasisFunctionalData,
        b::_BasisEigenfunctions, projected)
    latent = projected * transpose(b.coefs) .+ mu.coefs
    BasisFunctionalData(fd.coefs .- latent, fd.basis; ids=fd.ids, metadata=fd.metadata)
end

_training_node(f::MultivariateFPCA, j) =
    f.basis === :marginal ? f.fits[j].training_data : component(f.training_data, j)

"""
    residuals(model::Union{FPCA,MultivariateFPCA,MFPCA})

Subtract the fitted latent training reconstruction from original stored
observations, before any presmoothing. Return functional data with copied IDs
and the original observation layout: grids and masks, irregular times and
offsets, or exact basis coefficients. Multivariate results retain node order.
Grid/PACE/FACE functions use the same piecewise-linear reconstruction as
[`reconstruct`](@ref), evaluated at actual observation times; exact-basis
residuals retain the input basis. Missing points remain unobserved.

Results own their storage; modifying them does not change the fit. These
residuals include observation noise and truncation/smoothing error and are
not independent noise estimates. This extends `StatsAPI.residuals`; no
residuals are invented for FSVD, CCA or derivative summaries.
"""
residuals(f::FPCA) = _observation_residuals(f.training_data, f.mean, f.basis, f.scores)

function residuals(f::MultivariateFPCA{T}) where {T}
    nodes = AbstractFunctionalData{T}[
        _observation_residuals(_training_node(f, j), f.means[j], f.bases[j], f.scores[j])
        for j in eachindex(f.scores)]
    MultivariateFunctionalData{T}(nodes)
end

function residuals(f::MFPCA{T}) where {T}
    nodes = AbstractFunctionalData{T}[
        _observation_residuals(_training_node(f.univariate, j),
            f.univariate.means[j], f.eigenfunctions[j], f.scores)
        for j in eachindex(f.eigenfunctions)]
    MultivariateFunctionalData{T}(nodes)
end

"""
    coef(fd::BasisFunctionalData)

Return the stored coefficient matrix, with curves in rows and input basis
functions in columns. Mutating it changes the data; use `copy` for independent
storage. This extends `StatsAPI.coef`. Decomposition scores, canonical weights
and multivariate rotations are not exposed as ambiguous model coefficients.
"""
coef(fd::BasisFunctionalData) = fd.coefs
