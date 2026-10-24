# Product-kernel local linear regression. Coordinates are scaled by bandwidth
# so the three normal-equation columns have comparable units. Only the
# intercept is needed. No nearest-neighbour bandwidth adjustment is applied.
function _local_linear2d(x, y, z, weights, s, t, h, kernel)
    T = promote_type(eltype(x), eltype(z), typeof(h), typeof(s), typeof(t))
    A = zeros(T, 3, 3)
    b = zeros(T, 3)
    for i in eachindex(z)
        u, v = (x[i] - s)/h, (y[i] - t)/h
        w = weights[i] * kernel(u) * kernel(v)
        w == 0 && continue
        row = (one(T), u, v)
        for j in 1:3
            b[j] += w * row[j] * z[i]
            for k in 1:3
                A[j, k] += w * row[j] * row[k]
            end
        end
    end
    failure() = ArgumentError("covariance has a singular or undefined local linear fit at point ($s, $t); use a larger bandwidth")
    all(isfinite, A) && all(isfinite, b) || throw(failure())
    factor = cholesky(Symmetric(A); check=false)
    isposdef(factor) || throw(failure())
    minimum(abs2, diag(factor.U)) > 100eps(T) * maximum(abs, A) || throw(failure())
    value = (factor \ b)[1]
    isfinite(value) || throw(failure())
    value
end
