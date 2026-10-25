"""
    DerivativeBasis(basis, order=1)

The functions obtained by differentiating `basis` `order` times. The basis
is copied; its domain and number of functions are retained. Evaluation delegates
to `basismatrix(basis, t; derivative=order+derivative)`. Gram and penalty matrices
are exact whenever the underlying basis supplies exact derivative penalties
(B-splines through BSplineKit Galerkin, Fourier and monomial by closed forms).
Zero functions and a singular Gram matrix are valid. Orders are nonnegative
integers; nested derivative orders add. Unlike fda 6.3.0 `deriv.fd`, this does
not project the derivative back into the original basis span.
"""
struct DerivativeBasis{B<:AbstractBasis} <: AbstractBasis
    basis::B
    order::Int
    function DerivativeBasis(basis::B, order=1) where {B<:AbstractBasis}
        nu = _derivative_order(order)
        new{B}(deepcopy(basis), nu)
    end
end

DerivativeBasis(basis::DerivativeBasis, order=1) =
    DerivativeBasis(basis.basis, _derivative_sum(basis.order, order))

Base.eltype(::Type{DerivativeBasis{B}}) where {B} = eltype(B)
nbasis(b::DerivativeBasis) = nbasis(b.basis)
domain(b::DerivativeBasis) = domain(b.basis)
basismatrix(b::DerivativeBasis, t::AbstractVector{<:Real}; derivative=0) =
    basismatrix(b.basis, t; derivative=_derivative_sum(b.order, derivative))
penalty(b::DerivativeBasis, order) = penalty(b.basis, _derivative_sum(b.order, order))
Base.:(==)(a::DerivativeBasis, b::DerivativeBasis) = a.order == b.order && a.basis == b.basis
Base.isequal(a::DerivativeBasis, b::DerivativeBasis) =
    isequal(a.order, b.order) && isequal(a.basis, b.basis)
Base.hash(b::DerivativeBasis, h::UInt) = hash(b.basis, hash(b.order, hash(:DerivativeBasis, h)))
Base.show(io::IO, b::DerivativeBasis) = print(io, "DerivativeBasis(", b.basis, ", order=", b.order, ")")
Base.show(io::IO, ::MIME"text/plain", b::DerivativeBasis) = show(io, b)

"""
    derivative(fd::BasisFunctionalData, order=1)
    derivative(f::FPCA, order=1; method=:FPC, bandwidth=nothing,
        mean_bandwidth=nothing, covariance_bandwidth=nothing,
        kernel=KernelSmoothing.GaussianKernel(), degree=nothing,
        ncomponents=nothing, eigen_tol=1e-10, sign_tol=1e-12)
    derivative(f::MultivariateFPCA, order=1; kwargs...)

Differentiate basis curves exactly, retaining copied coefficients, ids and
metadata in a [`DerivativeBasis`](@ref). Raw grid and irregular data throw:
use `smooth_curves(fd, LocalPolynomial(; derivative=order, bandwidth); grid)`
or smooth/project to basis data first. No raw finite differences are used.

For FPCA, return [`FPCADerivative`](@ref). Defaults `method=:FPC`, `order=1`,
and Gaussian kernel follow fdapace 0.6.0 `SetDerOptions`. With no bandwidth
supplied, both bandwidths equal `order*c*domain_width`, where `c=0.10` if
`noise_variance/sum(eigvals(f)) >= 0.01`, otherwise `c=0.05` (including dense
fits with no noise estimate). Supply a shared `bandwidth`, or both
`mean_bandwidth` and `covariance_bandwidth`, never a mixture.

`:FPC` locally differentiates the stored mean and selected eigenfunctions,
using degree equal to `order` by default, as in fdapace `FPCAder`. Original
scores and their conditional covariances are retained; its eigenvalues describe
original curves, not derivative variance. Orders 1 and 2 are supported.

`:DPC` supports order 1. It uses degree 2 by default, and differentiates the
fitted covariance in both arguments with the existing local polynomial
smoother. It reuses weighted FPCA eigenanalysis and selects `FVE(0.9999)` by
default (fdapace's derivative threshold); an integer or `FVE` overrides it.
Dense scores project locally differentiated, centered training curves. PACE
and FACE scores use conditional expectation with the original full projected
covariance and noise, and the cross-covariance of curves with derivative scores.
The stored mean is differentiated, rather than re-estimated from observations.
This gives consistent derivative scores/uncertainties on nonuniform grids;
fdapace instead uses covariance finite differences, rectangular quadrature,
and an unsmoothed spectrum for uncertainties. See the fdapace audit for the
precise reference comparison. `eigen_tol` and `sign_tol` have the same package
numerical defaults as [`FPCA`](@ref) and only apply to DPC. `degree` overrides
the method's polynomial degree, with `degree >= order`.

`FPC1` is not implemented (it requires a separate bivariate derivative smoother).
All settings are stored and displayed. Multivariate fits return a
[`MultivariateFPCADerivative`](@ref) in node order; keywords may be shared or
vectors of node settings, as for marginal FPCA. For pooled fits, FPC retains
the shared basis and original node scores; DPC uses each node's own sample
covariance and copied original observations.
"""
function derivative(fd::BasisFunctionalData, order=1)
    BasisFunctionalData(fd.coefs, DerivativeBasis(fd.basis, order);
        ids=fd.ids, metadata=fd.metadata)
end

function derivative(::Union{GridFunctionalData,IrregularFunctionalData}, order=1)
    _derivative_order(order)
    throw(ArgumentError("derivative of raw observations is undefined; use smooth_curves with LocalPolynomial(derivative=order, bandwidth=...) or to_basis first"))
end

"""
    FPCADerivative

Derivative fit returned by [`derivative`](@ref). Stores method, order, work
grid/weights, mean derivative (`mean`), derivative functions (`basis`, an
`FPCBasis`), eigenvalues, scores, conditional score covariances, copied subject
ids, both bandwidths, kernel, and all settings. For FPC, eigenvalues and scores
refer to the original components; derivative functions need not be orthogonal.
For DPC, they describe derivative components. `settings` records the covariance
source, differentiation rule, selection, dropped spectrum and sign rule.
Use `reconstruct`, `curve_data` and `mean_data` without resmoothing. Uncertainty
is conditional on the fitted mean/covariance/noise, excluding their estimation.
"""
struct FPCADerivative{T<:AbstractFloat,K,S,C}
    method::Symbol
    order::Int
    grid::Vector{T}
    weights::Vector{T}
    mean::GridFunctionalData{T}
    basis::FPCBasis{T}
    eigenvalues::Vector{T}
    scores::Matrix{T}
    score_covariances::C
    ids::Union{Nothing,Vector{Any}}
    mean_bandwidth::T
    covariance_bandwidth::T
    kernel::K
    settings::S
end

nobs(d::FPCADerivative) = size(d.scores, 1)
ncomponents(d::FPCADerivative) = size(d.scores, 2)
ids(d::FPCADerivative) = deepcopy(d.ids)
domain(d::FPCADerivative) = (first(d.grid), last(d.grid))
eigenfunctions(d::FPCADerivative) = d.basis
LinearAlgebra.eigvals(d::FPCADerivative) = d.eigenvalues
scores(d::FPCADerivative) = d.scores
score_covariances(d::FPCADerivative) = d.score_covariances

function _derivative_bandwidths(grid, values, sigma2, order, bandwidth, mean_bandwidth, covariance_bandwidth)
    if bandwidth !== nothing
        mean_bandwidth === nothing && covariance_bandwidth === nothing ||
            throw(ArgumentError("supply bandwidth or both mean_bandwidth and covariance_bandwidth, not both forms"))
        return bandwidth, bandwidth, :shared
    end
    if mean_bandwidth === nothing && covariance_bandwidth === nothing
        noisy = sigma2 !== nothing && sigma2 / sum(values) >= 0.01
        h = order * (noisy ? 0.10 : 0.05) * (last(grid) - first(grid))
        return h, h, noisy ? :fdapace_noise_ratio_010 : :fdapace_noise_ratio_005
    end
    mean_bandwidth !== nothing && covariance_bandwidth !== nothing ||
        throw(ArgumentError("mean_bandwidth and covariance_bandwidth must both be supplied"))
    mean_bandwidth, covariance_bandwidth, :separate
end

function _differentiate_columns(grid, values, method)
    out = similar(values, promote_type(eltype(values), eltype(grid), typeof(method.bandwidth)))
    # Every column is observed on the same grid, so one smoother matrix serves all.
    L = KernelSmoothing.smoothermatrix(grid, grid; bandwidth=method.bandwidth,
        degree=method.degree, derivative=method.derivative, kernel=method.kernel)
    for (point, row) in zip(grid, eachrow(L))
        all(isfinite, row) || throw(ArgumentError(
            "derivative function 1 has an undefined estimate at point $point; use a larger bandwidth"))
    end
    mul!(out, L, values)
end

function _derivative_result(method, order, grid, weights, mu, functions, values,
        projected, covariances, labels, mean_method, covariance_method, settings)
    T = promote_type(eltype(grid), eltype(mu), eltype(functions), eltype(projected))
    x, w = Vector{T}(grid), Vector{T}(weights)
    FPCADerivative(method, order, x, w,
        GridFunctionalData(reshape(Vector{T}(mu), 1, :), x),
        FPCBasis(x, Matrix{T}(functions); weights=w), Vector{T}(values),
        Matrix{T}(projected), deepcopy(covariances), deepcopy(labels),
        T(mean_method.bandwidth), T(covariance_method.bandwidth),
        deepcopy(mean_method.kernel), deepcopy(settings))
end

_derivative_covariance(f, ::Nothing) = f.covariance
_derivative_covariance(f, ::Real) = f.settings.full_projected_covariance

function _derivative_dpc_scores(f, method, functions, values, cov10, source, sigma2::Real)
    zeta = transpose(cov10) * (f.weights .* functions)
    predicted = _pace_ce(f.training_data, f.grid, vec(f.mean.values),
        zeta ./ transpose(values), values, source, sigma2)
    predicted.scores, predicted.covariances
end

function _derivative_dpc_scores(f, method, functions, values, cov10, source, ::Nothing)
    observations, _ = _fpca_preprocess(f.training_data, f.presmooth)
    centered = observations .- f.mean.values
    derivatives = _differentiate_columns(f.grid, Matrix(transpose(centered)), method)
    transpose(derivatives) * (f.weights .* functions), nothing
end

Base.@constprop :aggressive derivative(f::FPCA, order=1; kwargs...) =
    _fpca_derivative(f, order; kwargs...)

Base.@constprop :aggressive function _fpca_derivative(f, order; method=:FPC,
        bandwidth=nothing, mean_bandwidth=nothing, covariance_bandwidth=nothing,
        kernel=KernelSmoothing.GaussianKernel(), degree=nothing,
        ncomponents=nothing, eigen_tol=_FPCA_EIGEN_TOL, sign_tol=_FPCA_SIGN_TOL)
    nu = _derivative_order(order)
    method in (:FPC, :DPC) || throw(ArgumentError("derivative method must be :FPC or :DPC; FPC1 is not implemented"))
    nu in (1, 2) || throw(ArgumentError("FPCA derivative order must be 1 or 2"))
    method === :DPC && nu != 1 && throw(ArgumentError("DPC supports derivative order 1 only"))
    method === :FPC && (ncomponents !== nothing || eigen_tol != _FPCA_EIGEN_TOL || sign_tol != _FPCA_SIGN_TOL) &&
        throw(ArgumentError("ncomponents, eigen_tol and sign_tol apply only to DPC; FPC retains the original components"))
    hmu, hcov, bandwidth_rule = _derivative_bandwidths(f.grid, f.eigenvalues,
        f.noise_variance, nu, bandwidth, mean_bandwidth, covariance_bandwidth)
    p = degree === nothing ? (method === :FPC ? nu : nu + 1) : degree
    mm = LocalPolynomial(; bandwidth=hmu, derivative=nu, degree=p, kernel)
    cm = LocalPolynomial(; bandwidth=hcov, derivative=nu, degree=p, kernel)
    mu = vec(_differentiate_columns(f.grid, transpose(f.mean.values), mm))
    base_settings = (; source_method=f.method, degree=mm.degree,
        bandwidth_rule, mean_source=:stored, weighting=:equal_grid_points,
        minobs=0, binning=:none, folds=0, reference="fdapace 0.6.0",
        mean_interpolation=:piecewise_linear, eigenfunction_interpolation=:piecewise_linear,
        original_settings=deepcopy(f.settings))
    if method === :FPC
        functions = _differentiate_columns(f.grid, f.basis.values, cm)
        settings = merge(base_settings, (; covariance_derivative=:not_applicable,
            spectrum=:original, score_method=:original, selection=deepcopy(f.selection)))
        return _derivative_result(method, nu, f.grid, f.weights, mu, functions,
            f.eigenvalues, f.scores, f.score_covariances, f.ids, mm, cm, settings)
    end
    _fpca_tolerance(eigen_tol, "eigen_tol")
    _fpca_tolerance(sign_tol, "sign_tol")
    # D G D' is the covariance of the same linear derivative operator applied
    # to curves. No new smoother, eigenstep or conditional-expectation solver.
    source = _derivative_covariance(f, f.noise_variance)
    cov10 = _differentiate_columns(f.grid, source, cm)
    cov11 = _differentiate_columns(f.grid, Matrix(transpose(cov10)), cm)
    cov11 = (cov11 + transpose(cov11)) / 2
    system = _fpca_eigen(cov11, f.weights, mu; eigen_tol, sign_tol)
    rule = ncomponents === nothing ? FVE(0.9999) : ncomponents
    selection = _fpca_selection(rule, system.fve, length(f.grid))
    k = selection.chosen
    values, functions = system.eigenvalues[1:k], system.eigenfunctions[:, 1:k]
    projected, covariances = _derivative_dpc_scores(f, cm, functions, values,
        cov10, source, f.noise_variance)
    all(isfinite, projected) || throw(ArgumentError("DPC scores must be finite; rescale observations"))
    settings = merge(base_settings, (; covariance_derivative=:local_polynomial_both_arguments,
        covariance_source=f.method === :dense ? :sample : :full_projected,
        derivative_covariance=cov11, selection, all_eigenvalues=system.eigenvalues,
        fve=system.fve, dropped=system.dropped, eigen_tol, sign_tol,
        score_method=f.method === :dense ? :derivative_projection : :conditional_expectation,
        quadrature=:stored, spectrum=:derivative))
    _derivative_result(method, nu, f.grid, f.weights, mu, functions, values,
        projected, covariances, f.ids, mm, cm, settings)
end

"""
    reconstruct(d::FPCADerivative, scores=scores(d); grid=d.grid)

Return derivative curves as `GridFunctionalData`, interpolating the stored
mean derivative and derivative functions linearly on a finite increasing grid
inside `domain(d)`. No differentiation is repeated. Training scores preserve
copied ids; other score matrices have no inferred ids.
"""
function reconstruct(d::FPCADerivative, projected::AbstractMatrix{<:Real}=scores(d);
        grid::AbstractVector{<:Real}=d.grid)
    Base.require_one_based_indexing(projected, grid)
    size(projected, 1) > 0 || throw(ArgumentError("derivative reconstruction needs at least one score row"))
    size(projected, 2) == ncomponents(d) || throw(DimensionMismatch("derivative score columns must equal ncomponents"))
    all(isfinite, projected) || throw(ArgumentError("derivative reconstruction scores must be finite"))
    quadrature_weights(grid)
    functions = basismatrix(d.basis, grid)
    mu = evaluate(d.mean, grid, PiecewiseLinear())
    GridFunctionalData(projected * transpose(functions) .+ mu, grid;
        ids=projected === d.scores ? d.ids : nothing,
        metadata=(; method=d.method, derivative=d.order, settings=deepcopy(d.settings),
            mean_bandwidth=d.mean_bandwidth, covariance_bandwidth=d.covariance_bandwidth,
            kernel=deepcopy(d.kernel)))
end

function Base.show(io::IO, d::FPCADerivative)
    print(io, "FPCADerivative(method=", d.method, ", order=", d.order,
        ", n=", nobs(d), ", ncomponents=", ncomponents(d),
        ", mean_bandwidth=", d.mean_bandwidth, ", covariance_bandwidth=", d.covariance_bandwidth,
        ", kernel=", d.kernel, ", settings=", _show_summary(d.settings), ")")
end
Base.show(io::IO, ::MIME"text/plain", d::FPCADerivative) = show(io, d)

"""
    MultivariateFPCADerivative

Nodewise derivative results from [`derivative`](@ref) of `MultivariateFPCA`.
`component(d, j)` retrieves an `FPCADerivative`. `reconstruct(d; grid=nothing)`
returns multivariate derivative data, using each node's grid by default, a
shared real grid, or one grid per node. `curve_data` and `mean_data` return
node payloads for the plotting recipes.
"""
struct MultivariateFPCADerivative
    nodes::Vector{FPCADerivative}
end
nobs(d::MultivariateFPCADerivative) = nobs(first(d.nodes))
ncomponents(d::MultivariateFPCADerivative) = ncomponents.(d.nodes)
ids(d::MultivariateFPCADerivative) = ids(first(d.nodes))
function component(d::MultivariateFPCADerivative, j::Integer)
    1 <= j <= length(d.nodes) || throw(ArgumentError("derivative node index is outside 1:$(length(d.nodes))"))
    d.nodes[j]
end
scores(d::MultivariateFPCADerivative) = scores.(d.nodes)
eigenfunctions(d::MultivariateFPCADerivative) = eigenfunctions.(d.nodes)
LinearAlgebra.eigvals(d::MultivariateFPCADerivative) = eigvals.(d.nodes)
score_covariances(d::MultivariateFPCADerivative) = score_covariances.(d.nodes)

function _pooled_derivative_node(f, j, order; method=:FPC, kwargs...)
    b = f.bases[j]
    training = component(f.training_data::MultivariateFunctionalData, j)::GridFunctionalData
    covariance = if method === :DPC
        centered = training.values .- f.means[j].values
        transpose(centered) * centered / (nobs(f) - 1)
    else
        nothing
    end
    # A field-based input lets pooled nodes reuse every univariate derivative
    # step without asserting that their shared basis diagonalizes node covariance.
    input = (; method=:dense, grid=b.grid, weights=b.weights, mean=f.means[j],
        basis=b, covariance, noise_variance=nothing, eigenvalues=f.eigenvalues[j],
        scores=f.scores[j], score_covariances=nothing, ids=f.ids,
        selection=f.pooled.selection, presmooth=nothing, training_data=training,
        settings=(; source_basis=:pooled, node=j, original_settings=deepcopy(f.settings)))
    _fpca_derivative(input, order; method, kwargs...)
end

function derivative(f::MultivariateFPCA{T}, order=1; kwargs...) where {T}
    p = length(f.scores)
    options = (; kwargs...)
    split = map((name, value) -> _multivariate_keyword(value, name, p), keys(options), values(options))
    nodes = FPCADerivative[]
    for j in 1:p
        node_options = NamedTuple{keys(options)}(map((v, s) -> s ? v[j] : v, values(options), split))
        node = f.basis === :marginal ? derivative(f.fits[j], order; node_options...) :
            _pooled_derivative_node(f, j, order; node_options...)
        push!(nodes, node)
    end
    MultivariateFPCADerivative(nodes)
end

function _multivariate_grids(d::MultivariateFPCADerivative, grid)
    p = length(d.nodes)
    grid === nothing && return [node.grid for node in d.nodes]
    grid isa AbstractVector{<:Real} && return fill(grid, p)
    grid isa AbstractVector || throw(ArgumentError("derivative grids must be one real vector or a vector of grids"))
    Base.require_one_based_indexing(grid)
    length(grid) == p || throw(DimensionMismatch("derivative grids must have one entry per node"))
    all(g -> g isa AbstractVector{<:Real}, grid) || throw(ArgumentError("each derivative grid must be a real vector"))
    grid
end

function reconstruct(d::MultivariateFPCADerivative; grid=nothing)
    grids = _multivariate_grids(d, grid)
    MultivariateFunctionalData([reconstruct(node; grid=g) for (node, g) in zip(d.nodes, grids)])
end
Base.show(io::IO, d::MultivariateFPCADerivative) = print(io, "MultivariateFPCADerivative(", d.nodes, ")")
Base.show(io::IO, ::MIME"text/plain", d::MultivariateFPCADerivative) = show(io, d)
