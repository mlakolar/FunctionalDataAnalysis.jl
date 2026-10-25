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
        degree isa Integer && 0 <= degree <= typemax(Int) ||
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
"""
function smooth_curves(fd::Union{GridFunctionalData,IrregularFunctionalData},
        method::LocalPolynomial; grid::AbstractVector{<:Real})
    x = _evaluation_points(fd, grid)
    quadrature_weights(x)
    T = eltype(x)
    out = Matrix{Union{Missing,T}}(missing, nobs(fd), length(x))
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values to smooth"))
        indices = findall(z -> first(t) <= z <= last(t), x)
        isempty(indices) && continue
        model = KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, t, y;
            degree=method.degree, derivative=method.derivative,
            kernel=method.kernel, bandwidth=method.bandwidth,
            points=x[indices], weights=ones(length(t)), minobs=0)
        estimates = KernelSmoothing.coef(model)
        for (j, value) in zip(indices, estimates)
            isfinite(value) && isfinite(T(value)) || throw(ArgumentError(
                "curve $i has an undefined local polynomial estimate at point $(x[j]); use a larger bandwidth"))
            out[i, j] = value
        end
    end
    metadata = (; method=:local_polynomial, bandwidth=method.bandwidth,
        degree=method.degree, derivative=method.derivative,
        kernel=method.kernel, weighting=:equal_observations,
        grid=copy(x), minobs=0, binning=:none, folds=0)
    GridFunctionalData(out, x; ids=fd.ids, metadata)
end

function _gcv_selection(fd, basis, order, candidates)
    T = promote_type(eltype(fd), eltype(basis), eltype(candidates))
    scores = zeros(T, length(candidates))
    P = Matrix{T}(penalty(basis, order))
    k = nbasis(basis)
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values for GCV"))
        X = Matrix{T}(basismatrix(basis, t))
        yy = Vector{T}(y)
        n = length(y)
        XtX = transpose(X) * X
        for (j, lambda) in enumerate(candidates)
            isfinite(scores[j]) || continue
            if lambda == 0
                factor = qr(X, ColumnNorm())
                diagonal = abs.(diag(factor.R))
                tol = max(size(X)...) * eps(T) * maximum(diagonal)
                if n < k || count(>(tol), diagonal) < k
                    scores[j] = T(Inf)
                    continue
                end
                c = factor \ yy
                df = T(k)
            else
                factor = cholesky(Symmetric(XtX + T(lambda) * P); check=false)
                if !isposdef(factor)
                    scores[j] = T(Inf)
                    continue
                end
                c = factor \ (transpose(X) * yy)
                df = tr(factor \ XtX)
            end
            residual_df = n - df
            if residual_df <= sqrt(eps(T)) * n
                scores[j] = T(Inf)
            else
                scores[j] += n * sum(abs2, yy - X * c) / residual_df^2
            end
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
