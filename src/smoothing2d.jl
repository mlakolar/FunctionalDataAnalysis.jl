# Product-kernel local linear covariance surface on the grid g1 × g2, from
# KernelSmoothing's bivariate smoother. Unidentified fits throw with their point.
function _local_linear_surface(pool, g1, g2, h, kernel)
    surface = KernelSmoothing.local_polynomial((pool.x, pool.y), pool.z, (g1, g2);
        bandwidth=h, degree=1, kernel, weights=pool.weights)
    for j in axes(surface, 2), i in axes(surface, 1)
        isfinite(surface[i, j]) || throw(ArgumentError(
            "covariance has a singular or undefined local linear fit at point ($(g1[i]), $(g2[j])); use a larger bandwidth"))
    end
    surface
end

# The rotated diagonal fit of fdapace 0.6.0 RrotatedMullwlsk.cpp: columns
# (1, u^2, v), with no linear across-diagonal term and no cross term. Coordinates
# are scaled by bandwidth so the columns have comparable units; only the
# intercept is needed. No nearest-neighbour bandwidth adjustment is applied.
function _rotated_covariance_fit(x, y, z, weights, s, t, h, kernel)
    T = promote_type(eltype(x), eltype(y), eltype(z), eltype(weights),
        typeof(h), typeof(s), typeof(t))
    A = zeros(T, 3, 3)
    b = zeros(T, 3)
    for i in eachindex(z)
        u, v = (x[i] - s)/h, (y[i] - t)/h
        w = weights[i] * kernel(u) * kernel(v)
        w == 0 && continue
        row = (one(u), u^2, v)
        for j in 1:3
            b[j] += w * row[j] * z[i]
            for k in 1:3
                A[j, k] += w * row[j] * row[k]
            end
        end
    end
    failure() = ArgumentError("covariance has a singular or undefined rotated diagonal fit at point ($s, $t); use a larger bandwidth")
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
        out[i] = _rotated_covariance_fit(across, along, pool.z, pool.weights,
            zero(T), root*t, bandwidth, kernel)
    end
    out
end
