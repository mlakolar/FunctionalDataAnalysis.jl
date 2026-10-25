"""
    MFPCA
    fit(MFPCA, mfd::MultivariateFunctionalData; ncomponents,
        univariate=(;), weights=ones(ncomponents(mfd)), eigen_tol=1e-10)

Happ and Greven (2018, Proposition 5; doi:10.1080/01621459.2016.1273115)
multivariate FPCA. First call marginal `fit(FPCA, mfd; univariate...)`, then
diagonalize `Ac'Ac/(n-1)`, where `A` concatenates the univariate scores with
node `j` multiplied by `sqrt(weights[j])` and `Ac` subtracts column means.
This matches `cov(allScores)` in R MFPCA 1.3-11's exact eigenanalysis.
For unit eigenvector `c` and eigenvalue `nu`, set `Q=A/sqrt(n-1)` and
`h=1/norm(Q*c)`. Scores are `A*c*sqrt(nu)*h`; node `j`'s function is
`phi_j * (Q'Q*c)_j * h/sqrt(weights[j]*nu)`, following R's uncentered
normalization. With mean-zero scores these reduce to `A*c` and
`phi_j*c_j/sqrt(weights[j])`, with weighted orthonormal functions.
For nonzero score means the R-normalized functions need not be orthonormal.
Stored node quadrature weights are used on different grids and domains.

`ncomponents` is required and must be a positive integer, as for `M` in R
MFPCA 1.3-11. It cannot exceed the retained positive score rank; requests
are never silently truncated. `weights` must be finite and strictly positive.
The default all ones follows R MFPCA (represented in the data's floating
type). The paper's alternative is inverse integrated node variance,
`1 / integral(Var(X_j))`; it is never selected automatically.

`univariate` must be a named tuple of keywords for marginal FPCA. Each may
be shared or a vector of node-specific values, following [`MultivariateFPCA`](@ref).
Its empty default reuses the existing dense/PACE routing and all [`FPCA`](@ref)
defaults, including `FVE(0.99)`. This intentionally differs from R's required
`uniExpansions` and its spline-based `uFPCA`/PACE implementation: no smoother
is duplicated. PACE conditional scores can have nonzero sample means: only
the covariance eigenproblem centers them, while normalization and score
projection use uncentered scores, exactly as in R's default exact route.
Unlike marginal PACE eigenvalues, the multivariate eigenvalues describe the
sample covariance of these estimated scores.

For common-grid margins, `univariate=(; method=:face, face=(; ...))` uses
FACE with the defaults and source audit in [`FPCA`](@ref). Its shrinkage
scores pass through this same joint eigenproblem and prediction interface.

The existing univariate eigenstep is reused with `eigen_tol=1e-10`, the same
package numerical cutoff relative to the largest absolute eigenvalue. Its
largest-absolute-entry sign rule is applied to `c` (first entry breaks ties).
Repeated eigenvalues identify only an eigenspace, not a unique orientation.
The fit stores `eigvals`, `eigenfunctions` (one [`FPCBasis`](@ref) per node),
`scores`, `coefficients`, the marginal fit in `univariate`, copied node weights,
IDs, sign rule, and all requested/resolved settings, including R normalization
factors and score scales. `show` displays choices.
Exact eigenanalysis is used; approximate eigenanalysis and bootstrap are not
implemented. Reconstruction is explicit through [`reconstruct`](@ref), matching
R's `fit=FALSE` default. See the manual's Comparison with MFPCA section.
"""
struct MFPCA{T<:AbstractFloat,U<:AbstractFloat,O}
    eigvals::Vector{T}
    eigenfunctions::Vector{FPCBasis{T}}
    scores::Matrix{T}
    coefficients::Matrix{T}
    univariate::MultivariateFPCA{U}
    weights::Vector{T}
    ids::Union{Nothing,Vector{Any}}
    sign_rule::NamedTuple
    settings::O
end

function fit(::Type{MFPCA}, mfd::MultivariateFunctionalData;
        ncomponents, univariate=(;),
        weights=ones(eltype(mfd), FunctionalDataAnalysis.ncomponents(mfd)),
        eigen_tol=1e-10)
    ncomponents isa Integer && !(ncomponents isa Bool) && ncomponents >= 1 ||
        throw(ArgumentError("MFPCA ncomponents must be a positive integer"))
    univariate isa NamedTuple || throw(ArgumentError("MFPCA univariate must be a named tuple of marginal FPCA keywords"))
    haskey(univariate, :basis) && throw(ArgumentError("MFPCA univariate cannot set basis; the first step is marginal FPCA"))
    p = FunctionalDataAnalysis.ncomponents(mfd)
    weights isa AbstractVector{<:Real} || throw(ArgumentError("MFPCA weights must be a real vector"))
    Base.require_one_based_indexing(weights)
    length(weights) == p || throw(DimensionMismatch("MFPCA weights must have one entry per node ($p)"))
    all(w -> isfinite(w) && w > 0, weights) || throw(ArgumentError("MFPCA weights must be finite and strictly positive"))
    _fpca_tolerance(eigen_tol, "MFPCA eigen_tol")
    n = nobs(mfd)
    n >= 2 || throw(ArgumentError("MFPCA requires at least two subjects"))
    marginal = _multivariate_marginal(mfd, univariate)
    _mfpca(marginal, weights, ncomponents, eigen_tol, univariate)
end

function _mfpca(marginal::MultivariateFPCA{U}, weights::AbstractVector{W},
        count, eigen_tol, options::NamedTuple) where {U,W<:Real}
    T = promote_type(U, _floattype(W))
    w = Vector{T}(weights)
    all(x -> isfinite(x) && x > 0, w) || throw(ArgumentError("MFPCA weights must remain finite and positive after conversion"))
    counts = ncomponents(marginal)
    offsets = cumsum(vcat(0, counts))
    total = last(offsets)
    count <= total || throw(ArgumentError("MFPCA ncomponents exceeds the total univariate component count ($total)"))
    root = sqrt.(w)
    weighted_scores = Matrix{T}(undef, nobs(marginal), total)
    for j in eachindex(counts)
        weighted_scores[:, offsets[j]+1:offsets[j+1]] = marginal.scores[j] .* root[j]
    end
    all(isfinite, weighted_scores) || throw(ArgumentError("MFPCA weighted scores must be finite; rescale observations or weights"))
    # Divide before the cross product to avoid unnecessary intermediate overflow.
    scaled = weighted_scores ./ sqrt(T(nobs(marginal)-1))
    centered = scaled .- sum(scaled; dims=1) ./ nobs(marginal)
    covariance = transpose(centered) * centered
    system = _fpca_eigen(covariance, ones(T, total), zeros(T, total);
        max_k=total, eigen_tol, sign_tol=0)
    selection = _fpca_selection(count, system.fve, total)
    c = system.eigenfunctions[:, 1:Int(count)]
    values = system.eigenvalues[1:Int(count)]
    # R MFPCA 1.3-11 calcMFPCA centers only for the exact eigenproblem;
    # normFactors, scores and tmpWeights all use the uncentered matrix Q.
    scaled_projection = scaled * c
    normalization_factors = [inv(norm(column)) for column in eachcol(scaled_projection)]
    score_scales = sqrt.(values) .* normalization_factors
    function_coefficients = (transpose(scaled) * scaled_projection) .*
        transpose(normalization_factors ./ sqrt.(values))
    all(isfinite, normalization_factors) && all(isfinite, function_coefficients) ||
        throw(ArgumentError("MFPCA normalization must be finite; rescale observations or weights"))
    projected = (weighted_scores * c) .* transpose(score_scales)
    all(isfinite, projected) || throw(ArgumentError("MFPCA scores must be finite; rescale observations or weights"))
    bases = FPCBasis{T}[]
    for j in eachindex(counts)
        b = marginal.bases[j]
        functions = Matrix{T}(b.values) * function_coefficients[offsets[j]+1:offsets[j+1], :] ./ root[j]
        push!(bases, FPCBasis(Vector{T}(b.grid), functions; weights=Vector{T}(b.weights)))
    end
    sign_rule = (; method=:largest_absolute_coefficient, ties=:first_entry)
    settings = (; ncomponents=Int(count), univariate=deepcopy(options),
        node_settings=NamedTuple[deepcopy(f.settings) for f in marginal.fits],
        univariate_counts=counts, offsets, weights=copy(w), eigen_tol,
        covariance_denominator=nobs(marginal)-1, score_centering=:covariance_only,
        normalization=:r_mfpca_1_3_11, normalization_factors, score_scales,
        eigenanalysis=:exact, bootstrap=false, fit=false, sign_rule,
        selection, all_eigenvalues=system.eigenvalues, dropped=system.dropped)
    MFPCA(values, bases, projected, c, marginal,
        w, ids(marginal), sign_rule, settings)
end

"""
    ncomponents(f::MFPCA)

Return the selected number of joint multivariate components.
"""
ncomponents(f::MFPCA) = size(f.scores, 2)
nobs(f::MFPCA) = size(f.scores, 1)
ids(f::MFPCA) = f.ids === nothing ? nothing : deepcopy(f.ids)

"""
    component(f::MFPCA, j)

Return node `j`'s stored univariate FPCA fit from the marginal first step.
"""
component(f::MFPCA, j::Integer) = component(f.univariate, j)

"""
    eigvals(f::MFPCA)

Return the selected multivariate eigenvalues in decreasing order.
"""
LinearAlgebra.eigvals(f::MFPCA) = f.eigvals

"""
    eigenfunctions(f::MFPCA)

Return one stored `FPCBasis` per node, with a shared multivariate column index.
With mean-zero univariate scores, orthonormality holds for the weighted sum
of node inner products, rather than for each node separately. With nonzero
score means, R's normalization need not give orthonormal functions; see
[`MFPCA`](@ref). Use `deepcopy` for independent arrays.
"""
eigenfunctions(f::MFPCA) = f.eigenfunctions

"""
    scores(f::MFPCA)

Return the stored subject × multivariate-component training score matrix.
Use `copy` for independent storage.
"""
scores(f::MFPCA) = f.scores

_mfpca_prediction_scores(s::AbstractMatrix) = s
_mfpca_prediction_scores(s::NamedTuple) = s.scores

"""
    predict(f::MFPCA, newmfd::MultivariateFunctionalData)

Predict univariate scores per node using the stored marginal fit, then apply
the fitted node weights, multivariate coefficients and R score scales to the
uncentered scores (no new-data mean subtraction). Return a subject ×
component matrix. PACE uses its existing conditional-expectation solver;
conditional uncertainties are not propagated across nodes. Node counts must
match, and each node must satisfy its univariate prediction requirements.
"""
function predict(f::MFPCA{T}, mfd::MultivariateFunctionalData{S}) where {T,S}
    predicted = predict(f.univariate, mfd)
    R = promote_type(T, S)
    weighted_scores = Matrix{R}(undef, nobs(mfd), size(f.coefficients, 1))
    offsets = f.settings.offsets
    for j in eachindex(predicted)
        weighted_scores[:, offsets[j]+1:offsets[j+1]] =
            _mfpca_prediction_scores(predicted[j]) .* sqrt(f.weights[j])
    end
    projected = (weighted_scores * f.coefficients) .* transpose(f.settings.score_scales)
    all(isfinite, projected) || throw(ArgumentError("MFPCA prediction scores must be finite; rescale observations or weights"))
    projected
end

"""
    reconstruct(f::MFPCA, scores=scores(f); grid=nothing)

Return `MultivariateFunctionalData` with node `j` equal to its training mean
plus `scores * eigenfunctions(f)[j]'`. `grid=nothing` uses each training grid;
a real vector is shared, or a vector of grids applies nodewise. Grids must
be finite, strictly increasing and inside the corresponding training domains.
Scores must be finite with one column per multivariate component. Training
IDs are retained only for the stored training score object.
"""
function reconstruct(f::MFPCA{T}, projected::AbstractMatrix{S}=scores(f);
        grid=nothing) where {T,S<:Real}
    Base.require_one_based_indexing(projected)
    size(projected, 1) >= 1 || throw(ArgumentError("MFPCA reconstruction requires at least one score row"))
    size(projected, 2) == ncomponents(f) || throw(DimensionMismatch("MFPCA reconstruction score columns must equal ncomponents"))
    all(isfinite, projected) || throw(ArgumentError("MFPCA reconstruction scores must be finite"))
    grids = _multivariate_grids(f.univariate, grid)
    R = promote_type(T, _floattype(S), _multivariate_grid_type(T, grid))
    nodes = AbstractFunctionalData{R}[]
    for j in eachindex(f.eigenfunctions)
        points = grids[j]
        quadrature_weights(points)
        functions = basis_matrix(f.eigenfunctions[j], points)
        mu = evaluate(f.univariate.means[j], points, PiecewiseLinear())
        values = projected * transpose(functions) .+ mu
        push!(nodes, GridFunctionalData(Matrix{R}(values), Vector{R}(points);
            ids=projected === f.scores ? f.ids : nothing))
    end
    MultivariateFunctionalData{R}(nodes)
end

function Base.show(io::IO, f::MFPCA)
    print(io, "MFPCA(n=", nobs(f), ", nodes=", length(f.weights),
        ", ncomponents=", ncomponents(f), ", weights=", f.weights,
        ", sign_rule=", f.sign_rule, ", settings=", f.settings,
        ", univariate=", f.univariate, ")")
end
Base.show(io::IO, ::MIME"text/plain", f::MFPCA) = show(io, f)
