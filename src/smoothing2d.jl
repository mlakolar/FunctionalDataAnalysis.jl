# Product-kernel local linear regression. Coordinates are scaled by bandwidth
# so the three normal-equation columns have comparable units. Only the
# intercept is needed. No nearest-neighbour bandwidth adjustment is applied.
_bandwidth_axes(h::Real) = (h, h)
_bandwidth_axes(h::Tuple{<:Real,<:Real}) = h

_covariance_row(u, v, ::Val{:linear}) = (one(u), u, v)
# fdapace 0.6.0 RrotatedMullwlsk.cpp: no linear across-diagonal term
# and no cross term. Scaling columns by bandwidth preserves the intercept.
_covariance_row(u, v, ::Val{:rotated}) = (one(u), u^2, v)

_local_linear2d(x, y, z, weights, s, t, h, kernel) =
    _local_covariance2d(x, y, z, weights, s, t, h, kernel, Val(:linear))

function _local_covariance2d(x, y, z, weights, s, t, h, kernel, basis)
    h1, h2 = _bandwidth_axes(h)
    T = promote_type(eltype(x), eltype(y), eltype(z), eltype(weights),
        typeof(h1), typeof(h2), typeof(s), typeof(t))
    A = zeros(T, 3, 3)
    b = zeros(T, 3)
    for i in eachindex(z)
        u, v = (x[i] - s)/h1, (y[i] - t)/h2
        w = weights[i] * kernel(u) * kernel(v)
        w == 0 && continue
        row = _covariance_row(u, v, basis)
        for j in 1:3
            b[j] += w * row[j] * z[i]
            for k in 1:3
                A[j, k] += w * row[j] * row[k]
            end
        end
    end
    label = basis isa Val{:rotated} ? "rotated diagonal" : "local linear"
    failure() = ArgumentError("covariance has a singular or undefined $label fit at point ($s, $t); use a larger bandwidth")
    all(isfinite, A) && all(isfinite, b) || throw(failure())
    factor = cholesky(Symmetric(A); check=false)
    isposdef(factor) || throw(failure())
    minimum(abs2, diag(factor.U)) > 100eps(T) * maximum(abs, A) || throw(failure())
    value = (factor \ b)[1]
    isfinite(value) || throw(failure())
    value
end

function _rotated_covariance_diagonal(pool, grid, bandwidth, kernel)
    T = promote_type(eltype(pool.x), eltype(grid), typeof(bandwidth))
    root = sqrt(T(2))
    across = (pool.x - pool.y)/root
    along = (pool.x + pool.y)/root
    out = Vector{T}(undef, length(grid))
    for (i, t) in enumerate(grid)
        out[i] = _local_covariance2d(across, along, pool.z, pool.weights,
            zero(T), root*t, bandwidth, kernel, Val(:rotated))
    end
    out
end
