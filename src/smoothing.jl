"""
    LocalPolynomial(; bandwidth, derivative=0, degree=derivative+1,
        kernel=KernelSmoothing.EpanechnikovKernel())

Per-curve local polynomial settings. Supply a finite positive scalar bandwidth;
no automatic selector or nearest-neighbour bandwidth floor is used. `derivative`
is a nonnegative integer and `degree >= derivative`. The default degree is
ν + 1 (odd degree minus derivative order; Fan and Gijbels, 1996), as in
KernelSmoothing.jl. Estimates are derivatives, not finite differences. Kernels
are `KernelSmoothing.AbstractKernel` instances.
Use [`smooth_curves`](@ref), or `StatsAPI.fit(LocalPolynomial, fd; grid, bandwidth, ...)`.
Here `degree` is polynomial degree; [`BSplineBasis`](@ref) instead takes
spline `order = degree + 1`. Neither is the derivative order being estimated.
"""
struct LocalPolynomial{T<:AbstractFloat,K<:KernelSmoothing.AbstractKernel}
    bandwidth::T
    degree::Int
    derivative::Int
    kernel::K
    function LocalPolynomial(; bandwidth, derivative=0, degree=nothing,
            kernel=KernelSmoothing.EpanechnikovKernel())
        bandwidth isa Real && isfinite(bandwidth) && bandwidth > 0 ||
            throw(ArgumentError("bandwidth must be a finite positive number"))
        nu = _derivative_order(derivative)
        degree = degree === nothing ? _derivative_sum(nu, 1) : degree
        _integer_in(degree, 0) ||
            throw(ArgumentError("degree must be a nonnegative integer representable as Int"))
        degree >= nu || throw(ArgumentError("degree must be at least derivative order"))
        kernel isa KernelSmoothing.AbstractKernel ||
            throw(ArgumentError("kernel must be a KernelSmoothing.AbstractKernel"))
        h = float(bandwidth)
        isfinite(h) || throw(ArgumentError("bandwidth must remain finite after conversion"))
        new{typeof(h),typeof(kernel)}(h, Int(degree), nu, deepcopy(kernel))
    end
end

"""
    GCV(lambdas)

Select one common spline penalty from a copied, nonempty vector of distinct,
finite nonnegative candidates. Minimize the sum over curves of
`n * RSS / (n - tr(H))^2`, where `H` is the unweighted penalized least-squares
smoother matrix at that curve's observed times. Singular fits and zero residual
degrees of freedom receive `Inf`; no curves are dropped. Exact ties use the
first supplied candidate. All scores are retained in the smoothed data's
`metadata.selection` named tuple (`lambdas`, `scores`, `lambda`).
"""
struct GCV{T<:AbstractFloat}
    lambdas::Vector{T}
    function GCV(lambdas::AbstractVector{<:Real})
        Base.require_one_based_indexing(lambdas)
        isempty(lambdas) && throw(ArgumentError("GCV lambdas must be nonempty"))
        all(x -> isfinite(x) && x >= 0, lambdas) ||
            throw(ArgumentError("GCV lambdas must be finite and nonnegative"))
        length(unique(lambdas)) == length(lambdas) ||
            throw(ArgumentError("GCV lambdas must be distinct"))
        T = _floattype(eltype(lambdas))
        values = Vector{T}(lambdas)
        all(isfinite, values) && length(unique(values)) == length(values) ||
            throw(ArgumentError("GCV lambdas must remain finite and distinct after conversion"))
        new{T}(values)
    end
end

"""
    PenalizedSpline(basis; lambda, order=2)

Per-curve unweighted penalized least squares through [`to_basis`](@ref).
`lambda` is a finite nonnegative number or [`GCV`](@ref) selecting a common
penalty. `order` is the nonnegative derivative order in the integrated squared
roughness penalty. The basis and settings are copied.
Use [`smooth_curves`](@ref), or `StatsAPI.fit(PenalizedSpline, fd; basis, lambda, order=2)`.
"""
struct PenalizedSpline{B<:AbstractBasis,L}
    basis::B
    lambda::L
    order::Int
    function PenalizedSpline(basis::AbstractBasis; lambda, order=2)
        lambda isa GCV || (lambda isa Real && isfinite(lambda) && lambda >= 0) ||
            throw(ArgumentError("lambda must be finite and nonnegative, or GCV(lambdas)"))
        new{typeof(basis),typeof(lambda)}(deepcopy(basis), deepcopy(lambda),
            _derivative_order(order))
    end
end

"""
    smooth_curves(fd, method::LocalPolynomial; grid)
    smooth_curves(fd, method::PenalizedSpline)

Smooth each observed curve of grid or irregular data, retaining copied subject
ids. Local polynomials return `GridFunctionalData` on the explicit finite,
strictly increasing work grid (at least two points). Points outside each
curve's observed range are missing; no extrapolation occurs. Empty curves and
nonfinite in-range estimates throw an `ArgumentError` identifying the curve
and point and suggesting a larger bandwidth.

Penalized splines return `BasisFunctionalData` through `to_basis`. Both results
store their method and all settings in `metadata`, shown by `show`. Spline GCV
also stores `metadata.selection`; fixed lambda uses `selection=nothing`.
All fits use equal observation weights, no binning, and no folds. GCV is a
within-curve prediction criterion, not a cluster-robust inference procedure.
Complete data on their original grid use one matrix multiplication for all
curves. Other designs cache at most eight distinct observed grids per call,
evicting the oldest on a cache miss. Smoother matrices retain KernelSmoothing's
Float64 arithmetic, including for Float32 data; output is converted to the
promoted data/work-grid type after smoothing.
"""
function smooth_curves end

function _curve_smoothing_metadata(method, x)
    (; method=:local_polynomial, bandwidth=method.bandwidth,
        degree=method.degree, derivative=method.derivative,
        kernel=method.kernel, weighting=:equal_observations,
        grid=copy(x), minobs=0, binning=:none, folds=0)
end

const _CURVE_SMOOTHER_CACHE_LIMIT = 8

function _curve_smoother_matrix(t, points, method)
    # Keep the backend's Float64 precision policy explicit. convert does not
    # copy a Matrix{Float64}, and preserves the previous cache conversion.
    convert(Matrix{Float64}, KernelSmoothing.smoother_matrix(t, points;
        bandwidth=method.bandwidth, degree=method.degree,
        derivative=method.derivative, kernel=method.kernel))
end

function _curve_smoother_entry(t, x, method)
    query_indices = findall(z -> first(t) <= z <= last(t), x)
    isempty(query_indices) && return query_indices, zeros(Float64, 0, length(t))
    query_indices, _curve_smoother_matrix(t, x[query_indices], method)
end

function _cached_curve_smoother!(cache, insertion_order, t, x, method)
    if !haskey(cache, t) && length(cache) == _CURVE_SMOOTHER_CACHE_LIMIT
        delete!(cache, popfirst!(insertion_order))
    end
    # Arguments are read-only captures; no binding is shared with an outer
    # assignment to the returned indices (the former Core.Box path).
    get!(cache, t) do
        entry = _curve_smoother_entry(t, x, method)
        push!(insertion_order, t)
        entry
    end
end

function _check_curve_estimate(value, ::Type{T}, curve, point) where {T}
    isfinite(value) && isfinite(T(value)) || throw(ArgumentError(
        "curve $curve has an undefined local polynomial estimate at point $point; use a larger bandwidth"))
    nothing
end

function _smooth_complete_grid(fd::GridFunctionalData, method, x::Vector{T}, weights, metadata) where {T}
    matrix = _curve_smoother_matrix(fd.grid, x, method)
    # Promote responses as the old Float64-matrix/vector products did. This
    # enables BLAS for Float32/Float64 without narrowing higher precision data.
    R = promote_type(eltype(fd), Float64)
    values = convert(Matrix{R}, fd.values) * transpose(matrix)
    if !all(value -> isfinite(value) && isfinite(T(value)), values)
        # Locate failures in the same curve/point order as the fallback path.
        for i in axes(values, 1), j in axes(values, 2)
            _check_curve_estimate(values[i, j], T, i, x[j])
        end
    end
    # All arrays are newly owned and already validated; avoid a missing-value
    # staging matrix and a second constructor pass over the numerical output.
    GridFunctionalData{T}(Val(:validated), convert(Matrix{T}, values), x, weights,
        nothing, _validate_ids(fd.ids, nobs(fd)), deepcopy(metadata))
end

function smooth_curves(fd::Union{GridFunctionalData,IrregularFunctionalData},
        method::LocalPolynomial; grid::AbstractVector{<:Real})
    x = _evaluation_points(fd, grid)
    weights = quadrature_weights(x)
    metadata = _curve_smoothing_metadata(method, x)
    if fd isa GridFunctionalData && fd.observed === nothing && x == fd.grid
        return _smooth_complete_grid(fd, method, x, weights, metadata)
    end
    T = eltype(x)
    out = Matrix{Union{Missing,T}}(missing, nobs(fd), length(x))
    smoothers = Dict{Vector{eltype(fd)},Tuple{Vector{Int},Matrix{Float64}}}()
    insertion_order = Vector{eltype(fd)}[]
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values to smooth"))
        indices, matrix = _cached_curve_smoother!(smoothers, insertion_order, t, x, method)
        isempty(indices) && continue
        estimates = matrix * y
        for (j, value) in zip(indices, estimates)
            _check_curve_estimate(value, T, i, x[j])
            out[i, j] = value
        end
    end
    GridFunctionalData(out, x; ids=fd.ids, metadata)
end

# Score an already prepared curve. Invalid candidates retain GCV's Inf policy;
# direct spline fits still diagnose singular systems in to_basis.
function _spline_gcv_score(X, y, XtX, P, Xty, lambda)
    T = eltype(X)
    n, k = size(X)
    if lambda == 0
        factor = qr(X, ColumnNorm())
        diagonal = abs.(diag(factor.R))
        tol = max(size(X)...) * eps(T) * maximum(diagonal)
        (n < k || count(>(tol), diagonal) < k) && return T(Inf)
        c = factor \ y
        df = T(k)
    else
        factor = cholesky(Symmetric(XtX + T(lambda) * P); check=false)
        isposdef(factor) || return T(Inf)
        c = factor \ Xty
        df = tr(factor \ XtX)
    end
    residual_df = n - df
    residual_df <= sqrt(eps(T)) * n ? T(Inf) :
        n * sum(abs2, y - X * c) / residual_df^2
end

function _gcv_selection(fd, basis, order, candidates)
    T = promote_type(eltype(fd), eltype(basis), eltype(candidates))
    scores = zeros(T, length(candidates))
    P = Matrix{T}(penalty(basis, order))
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values for GCV"))
        X = Matrix{T}(basis_matrix(basis, t))
        yy = Vector{T}(y)
        XtX = transpose(X) * X
        Xty = transpose(X) * yy
        for (j, lambda) in enumerate(candidates)
            isfinite(scores[j]) || continue
            scores[j] += _spline_gcv_score(X, yy, XtX, P, Xty, lambda)
        end
    end
    j = argmin(scores)
    isfinite(scores[j]) || throw(ArgumentError(
        "GCV has no admissible lambda: use more observations, fewer basis functions, or a positive penalty"))
    (; lambdas=copy(candidates), scores, lambda=candidates[j])
end

function smooth_curves(fd::Union{GridFunctionalData,IrregularFunctionalData},
        method::PenalizedSpline)
    selection = method.lambda isa GCV ?
        _gcv_selection(fd, method.basis, method.order, method.lambda.lambdas) : nothing
    lambda = selection === nothing ? method.lambda : selection.lambda
    fitted = to_basis(fd, method.basis; lambda, order=method.order)
    metadata = (; method=:penalized_spline, basis=method.basis, lambda,
        order=method.order, weighting=:equal_observations, grid=:observed_times,
        binning=:none, folds=0, selection)
    BasisFunctionalData(fitted.coefs, deepcopy(method.basis); ids=fd.ids, metadata)
end

fit(::Type{LocalPolynomial}, fd::Union{GridFunctionalData,IrregularFunctionalData};
        grid, kwargs...) = smooth_curves(fd, LocalPolynomial(; kwargs...); grid)
fit(::Type{PenalizedSpline}, fd::Union{GridFunctionalData,IrregularFunctionalData};
        basis, kwargs...) = smooth_curves(fd, PenalizedSpline(basis; kwargs...))

function Base.show(io::IO, method::LocalPolynomial)
    print(io, "LocalPolynomial(bandwidth=", method.bandwidth, ", degree=", method.degree,
        ", derivative=", method.derivative, ", kernel=", method.kernel, ")")
end
function Base.show(io::IO, method::GCV)
    print(io, "GCV(", method.lambdas, ")")
end
function Base.show(io::IO, method::PenalizedSpline)
    print(io, "PenalizedSpline(", method.basis, "; lambda=", method.lambda,
        ", order=", method.order, ")")
end
