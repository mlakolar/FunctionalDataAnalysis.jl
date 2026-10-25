"""
    to_irregular(fd)
    to_irregular(fd::BasisFunctionalData; grid)

Copy grid data to curve-specific observations, dropping only masked points.
At least one observation must remain in every curve. Basis data require an
explicit `grid`, where the functions are evaluated exactly. The original
domain is retained. Irregular data are copied without changing observations.
"""
function to_irregular(fd::GridFunctionalData{T}) where {T}
    IrregularFunctionalData(Vector{T}[fd[i][1] for i in 1:nobs(fd)],
        Vector{T}[fd[i][2] for i in 1:nobs(fd)]; domain=domain(fd), ids=fd.ids)
end
to_irregular(fd::IrregularFunctionalData) =
    IrregularFunctionalData(fd.times, fd.values, fd.offsets; domain=domain(fd), ids=fd.ids)
function to_irregular(fd::BasisFunctionalData; grid)
    sampled = to_grid(fd, grid)
    T = eltype(sampled)
    IrregularFunctionalData(Vector{T}[sampled[i][1] for i in 1:nobs(fd)],
        Vector{T}[sampled[i][2] for i in 1:nobs(fd)]; domain=domain(fd), ids=fd.ids)
end

"""
    to_grid(fd, grid, evaluator)
    to_grid(fd::BasisFunctionalData, grid)

Copy raw data to a supplied finite increasing grid, using an explicit
`PiecewiseLinear()` or `NoInterpolation()` evaluator. Piecewise-linear interpolation marks
target points outside each curve's observed range as missing, never `NaN`.
Exact conversion marks absent observations missing and requires every source
time to appear in the target grid using `==`, with no tolerance. Thus a
grid → irregular → grid round trip preserves the missing-value mask.

Basis data evaluate exactly on the supplied grid and need no evaluator.
"""
function to_grid(fd::Union{GridFunctionalData,IrregularFunctionalData},
        grid::AbstractVector{<:Real}, ::NoInterpolation)
    x = _evaluation_points(fd, grid)
    _validate_strict_grid(x)
    T = eltype(x)
    out = Matrix{Union{Missing,T}}(missing, nobs(fd), length(x))
    for i in 1:nobs(fd)
        times, values = fd[i]
        for k in eachindex(times)
            j = searchsortedfirst(x, times[k])
            j <= length(x) && x[j] == times[k] ||
                throw(ArgumentError("NoInterpolation target grid must contain every source time exactly"))
            out[i, j] = values[k]
        end
    end
    GridFunctionalData(out, x; ids=fd.ids)
end

function to_grid(fd::Union{GridFunctionalData,IrregularFunctionalData},
        grid::AbstractVector{<:Real}, evaluator::PiecewiseLinear)
    x = _evaluation_points(fd, grid)
    _validate_strict_grid(x)
    values = evaluate(fd, x, evaluator)
    T = eltype(values)
    out = Matrix{Union{Missing,T}}(values)
    out[isnan.(values)] .= missing
    GridFunctionalData(out, x; ids=fd.ids)
end

function to_grid(fd::BasisFunctionalData, grid::AbstractVector{<:Real})
    x = _evaluation_points(fd, grid)
    _validate_strict_grid(x)
    GridFunctionalData(evaluate(fd, x), x; ids=fd.ids)
end

"""
    to_basis(fd, basis; lambda=0, order=2)

Fit each raw curve by unweighted penalized least squares:
`sum((y - X*c).^2) + lambda * c' * penalty(basis, order) * c`.
Use the actual observed times, omitting only masked values. With zero
penalty, every curve needs at least `nbasis(basis)` observations and a
full-rank design; pivoted QR solves the fit. Positive finite `lambda` uses
Cholesky and throws if the penalized system is singular. No inverse is
formed. Basis data must first be sampled explicitly with `to_grid`.
"""
function to_basis(fd::Union{GridFunctionalData,IrregularFunctionalData},
        basis::AbstractBasis; lambda=0, order=2)
    lambda isa Real && isfinite(lambda) && lambda >= 0 ||
        throw(ArgumentError("lambda must be finite and nonnegative"))
    m = _derivative_order(order)
    T = promote_type(eltype(fd), _basis_eltype(basis))
    k = nbasis(basis)
    coefs = Matrix{T}(undef, nobs(fd), k)
    P = Matrix{T}(penalty(basis, m))
    for i in 1:nobs(fd)
        times, values = fd[i]
        length(times) >= k || lambda > 0 ||
            throw(ArgumentError("curve $i has fewer observations than basis functions; lambda must be positive"))
        isempty(times) && throw(ArgumentError("curve $i has no observed values to fit"))
        X = Matrix{T}(basis_matrix(basis, times))
        y = Vector{T}(values)
        if lambda == 0
            factor = qr(X, ColumnNorm())
            diagonal = abs.(diag(factor.R))
            tolerance = max(size(X)...) * eps(T) * maximum(diagonal)
            count(>(tolerance), diagonal) == k ||
                throw(ArgumentError("basis design is rank deficient for curve $i"))
            coefs[i, :] = factor \ y
        else
            A = transpose(X) * X + T(lambda) * P
            factor = cholesky(Symmetric(A); check=false)
            isposdef(factor) ||
                throw(ArgumentError("penalized basis fit is singular for curve $i"))
            coefs[i, :] = factor \ (transpose(X) * y)
        end
    end
    BasisFunctionalData(coefs, basis; ids=fd.ids)
end

to_basis(::BasisFunctionalData, ::AbstractBasis; kwargs...) =
    throw(ArgumentError("sample basis data explicitly with to_grid before fitting a basis"))

function Base.vcat(a::GridFunctionalData, b::GridFunctionalData)
    a.grid == b.grid || throw(ArgumentError("grid data must share the same grid for vcat"))
    a.metadata == b.metadata || throw(ArgumentError("grid data must share equal metadata for vcat"))
    GridFunctionalData(vcat(_grid_matrix(a), _grid_matrix(b)), a.grid;
        ids=_concatenated_ids(a, b), metadata=a.metadata)
end
function Base.vcat(a::IrregularFunctionalData, b::IrregularFunctionalData)
    domain(a) == domain(b) || throw(ArgumentError("irregular data must share the same domain for vcat"))
    offsets = vcat(a.offsets, b.offsets[2:end] .+ length(a.times))
    IrregularFunctionalData(vcat(a.times, b.times), vcat(a.values, b.values), offsets;
        domain=domain(a), ids=_concatenated_ids(a, b))
end
function Base.vcat(a::BasisFunctionalData, b::BasisFunctionalData)
    a.basis == b.basis || throw(ArgumentError("basis data must share the same basis for vcat"))
    a.metadata == b.metadata || throw(ArgumentError("basis data must share equal metadata for vcat"))
    BasisFunctionalData(vcat(a.coefs, b.coefs), a.basis;
        ids=_concatenated_ids(a, b), metadata=a.metadata)
end
