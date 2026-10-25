"""
    fit(FSVD, fd1, fd2; ncomponents=FVE(0.9999), maxK=nothing,
        bandwidth=nothing, kernel=KernelSmoothing.GaussianKernel(),
        fpca_options1=(;), fpca_options2=(;))

Functional singular component analysis of paired grid or irregular curves
(Yang, Müller and Stadtmüller, 2011). Subject counts and available IDs must
agree in order. Both marginal FPCA fits, cross-covariance, singular values,
functions, scores, cumulative squared cross-covariance fractions, and settings
are stored. Inputs are copied.

Defaults follow fdapace 0.6.0 `FSVD` / `SetSVDOptions`: FVE 0.9999,
`maxK=min(20,n-2)`, Gaussian smoothing, scores enabled, and independent
cross-node measurement errors (equal-time products retained). Fixed component
counts disable FVE selection. At least three subjects are required. Marginal
FPCA options are named tuples passed to [`FPCA`](@ref); empty tuples use its
source-audited defaults, including 51 work-grid points for PACE. Unless both
nodes are complete grids, FSVD forces `method=:pace` for both marginals,
including a complete-grid partner, so each has a measurement-error variance
for conditional scores. Explicit `method=:dense` is rejected on this route.
The requested and effective marginal options are recorded in `settings`.

Complete grid pairs reuse [`CrossCovariance`](@ref), rescaled to the reference
FSVD divisor `n-2` (distinct from the standalone sample covariance's `n-1`).
Otherwise [`SmoothedCrossCovariance`](@ref) uses the marginal means and work
grids. `bandwidth` is required on this route: a scalar, pair, or explicit
[`CurveCV`](@ref). This inherits the existing whole-subject CV contract in
place of fdapace's pair-level GCV. `kernel` applies to cross smoothing only;
marginal kernels are controlled by their FPCA options.

As in package FPCA, the SVD uses stored quadrature weights on both axes,
including nonuniform grids. This deliberately differs from fdapace's ordinary
matrix SVD and first-spacing singular-value scaling. Functions are mutually
orthonormal on each axis. Each left function's largest absolute entry is
positive; the right function and both scores receive the same sign. Relative
singular values at most `1e-10` of the largest are discarded before selection.
Dense scores are quadrature integrals. Sparse scores condition jointly on both
processes, using the existing PACE solver, PSD projection of the stacked
marginal/cross covariance, and the marginal noise variances; no rho is added.
"""
struct FSVD{T<:AbstractFloat,T1<:AbstractFloat,T2<:AbstractFloat,C,S,O}
    singular_values::Vector{T}
    left_functions::Matrix{T}
    right_functions::Matrix{T}
    left_scores::Matrix{T}
    right_scores::Matrix{T}
    fve::Vector{T}
    cross_covariance::C
    fpca1::FPCA{T1}
    fpca2::FPCA{T2}
    selection::S
    settings::O
end

function _fsvd_marginal_covariance(f::FPCA)
    f.method == :pace ? f.settings.full_projected_covariance : f.covariance
end

function _fsvd_joint_scores(fd1, fd2, f1, f2, c, left, right, values)
    g1, g2 = f1.grid, f2.grid
    m1, m2 = length(g1), length(g2)
    a, b = _fsvd_marginal_covariance(f1), _fsvd_marginal_covariance(f2)
    # The reference patches the unweighted stacked surface to be PSD.
    # Reuse the FPCA eigen step with unit weights for that same projection.
    stacked = [f1.covariance c.surface; transpose(c.surface) f2.covariance]
    stacked = (stacked + transpose(stacked))/2
    T = eltype(stacked)
    system = _fpca_eigen(stacked, ones(T, m1+m2), zeros(T, m1+m2);
        eigen_tol=0, sign_tol=0)
    projected = (system.eigenfunctions .* transpose(system.eigenvalues)) *
        transpose(system.eigenfunctions)
    # Integrate marginal score/observation covariances using the stored weights.
    p = a * (left .* f1.weights)
    q = b * (right .* f2.weights)
    left_scores = Matrix{T}(undef, nobs(fd1), length(values))
    right_scores = similar(left_scores)
    for i in 1:nobs(fd1)
        t1, y1 = fd1[i]
        t2, y2 = fd2[i]
        # Interpolation matrices retain the full stacked covariance and noise
        # grid before interpolation, as in FSVD's ConvertSupport route.
        h1 = _pace_interpolated_basis(g1, Matrix{T}(I, m1, m1), t1)
        h2 = _pace_interpolated_basis(g2, Matrix{T}(I, m2, m2), t2)
        h = [h1 zeros(T, length(t1), m2); zeros(T, length(t2), m1) h2]
        regularized = copy(projected)
        for j in 1:m1
            regularized[j,j] += something(f1.noise_variance, zero(T))
        end
        for j in 1:m2
            regularized[m1+j,m1+j] += something(f2.noise_variance, zero(T))
        end
        local_covariance = h * regularized * transpose(h)
        factor = _pace_cholesky(local_covariance, i)
        residual = vcat(y1 - vec(h1 * transpose(f1.mean.values)),
            y2 - vec(h2 * transpose(f2.mean.values)))
        solution = factor \ residual
        numerator1 = vcat(h1*p, (h2*right) .* transpose(values))
        numerator2 = vcat((h1*left) .* transpose(values), h2*q)
        left_scores[i,:] = transpose(numerator1)*solution
        right_scores[i,:] = transpose(numerator2)*solution
    end
    all(isfinite, left_scores) && all(isfinite, right_scores) ||
        throw(ArgumentError("FSVD conditional scores must be finite; rescale observations"))
    left_scores, right_scores
end

Base.@constprop :aggressive function fit(::Type{FSVD}, fd1::Union{GridFunctionalData,IrregularFunctionalData},
        fd2::Union{GridFunctionalData,IrregularFunctionalData};
        ncomponents=FVE(0.9999), maxK=nothing, bandwidth=nothing,
        kernel=KernelSmoothing.GaussianKernel(), fpca_options1=(;), fpca_options2=(;))
    _cross_subjects(fd1, fd2)
    n = nobs(fd1)
    n >= 3 || throw(ArgumentError("FSVD requires at least three subjects for its n-2 divisor and component cap"))
    fpca_options1 isa NamedTuple && fpca_options2 isa NamedTuple ||
        throw(ArgumentError("FSVD fpca_options1 and fpca_options2 must be named tuples"))
    ncomponents isa Bool && throw(ArgumentError("FSVD ncomponents must be a positive integer or FVE, not Bool"))
    dense = fd1 isa GridFunctionalData && fd2 isa GridFunctionalData &&
        fd1.observed === nothing && fd2.observed === nothing
    dense && bandwidth !== nothing && throw(ArgumentError(
        "FSVD bandwidth is unused for two complete grid nodes"))
    if dense
        all(o -> get(o, :presmooth, nothing) === nothing, (fpca_options1, fpca_options2)) ||
            throw(ArgumentError("FSVD dense marginals cannot use presmooth; smooth the input data explicitly"))
    end
    !dense && bandwidth === nothing && throw(ArgumentError(
        "FSVD bandwidth is required for sparse or incomplete nodes"))
    kernel isa KernelSmoothing.AbstractKernel || throw(ArgumentError(
        "FSVD kernel must be a KernelSmoothing.AbstractKernel"))
    if !dense
        for (options, name) in ((fpca_options1, "fpca_options1"), (fpca_options2, "fpca_options2"))
            get(options, :method, :auto) in (:auto, :pace) || throw(ArgumentError(
                "FSVD with sparse or incomplete nodes requires PACE marginals; use $name=(;method=:pace, ...)"))
        end
    end
    effective_options1 = merge(fpca_options1,
        (;method=dense ? get(fpca_options1, :method, :auto) : :pace))
    effective_options2 = merge(fpca_options2,
        (;method=dense ? get(fpca_options2, :method, :auto) : :pace))
    f1 = fit(FPCA, fd1; effective_options1...)
    f2 = fit(FPCA, fd2; effective_options2...)
    # A common within-node grid is needed by the existing surface and CE layer.
    for (fd, f) in ((fd1, f1), (fd2, f2))
        _pace_observations(fd, f.grid)
    end
    if dense
        f1.method == :dense && f2.method == :dense || throw(ArgumentError(
            "complete-grid FSVD requires dense marginal FPCA fits"))
        c0 = fit(CrossCovariance, fd1, fd2)
        T = eltype(c0.surface)
        c = CrossCovariance(c0.grid1, c0.grid2, c0.surface .* (T(n-1)/T(n-2)),
            c0.weights1, c0.weights2, c0.mean1, c0.mean2, n-2, c0.ids)
    else
        c = fit(SmoothedCrossCovariance, fd1, fd2, f1.mean, f2.mean;
            grid1=f1.grid, grid2=f2.grid, bandwidth, kernel)
    end
    maxK === nothing && (maxK = min(20, n-2))
    maxK isa Integer && !(maxK isa Bool) && 1 <= maxK <= typemax(Int) ||
        throw(ArgumentError("FSVD maxK must be a positive integer representable as Int"))
    root1, root2 = sqrt.(c.weights1), sqrt.(c.weights2)
    weighted = c.surface .* (root1 * transpose(root2))
    all(isfinite, weighted) || throw(ArgumentError("FSVD weighted cross-covariance must be finite"))
    d = svd(weighted)
    retained = min(Int(maxK), count(>(first(d.S)*_FPCA_EIGEN_TOL), d.S))
    retained > 0 || throw(ArgumentError("FSVD cross-covariance has no positive singular values above tolerance"))
    values = d.S[1:retained]
    fractions = cumsum((values ./ first(values)).^2) ./ sum((values ./ first(values)).^2)
    fractions[end] = one(eltype(fractions))
    selection = _fpca_selection(ncomponents, fractions, Int(maxK))
    k = selection.chosen
    values = values[1:k]
    left, right = d.U[:,1:k] ./ root1, d.V[:,1:k] ./ root2
    for j in 1:k
        if left[argmax(abs.(left[:,j])),j] < 0
            left[:,j] .*= -1
            right[:,j] .*= -1
        end
    end
    if dense
        ls = (fd1.values .- f1.mean.values) * (left .* c.weights1)
        rs = (fd2.values .- f2.mean.values) * (right .* c.weights2)
    else
        ls, rs = _fsvd_joint_scores(fd1, fd2, f1, f2, c, left, right, values)
    end
    all(isfinite, values) && all(isfinite, ls) && all(isfinite, rs) ||
        throw(ArgumentError("FSVD singular values and scores must be finite; rescale observations"))
    settings = (; quadrature=:stored, sign_rule=:left_largest_absolute_entry,
        singular_tol=_FPCA_EIGEN_TOL, scores=dense ? :integration : :joint_conditional_expectation,
        regularization=:sigma2, exclude_same_time=false, no_scores=false,
        fpca_options1=deepcopy(fpca_options1), fpca_options2=deepcopy(fpca_options2),
        effective_fpca_options1=deepcopy(effective_options1),
        effective_fpca_options2=deepcopy(effective_options2))
    FSVD(values, left, right, ls, rs, fractions, c, f1, f2, selection, settings)
end

function Base.show(io::IO, f::FSVD)
    print(io, "FSVD(singular_values=", f.singular_values,
        ", fve=", f.fve[1:length(f.singular_values)], ", selection=", f.selection,
        ", left_functions=", _show_summary(f.left_functions),
        ", right_functions=", _show_summary(f.right_functions),
        ", left_scores=", _show_summary(f.left_scores),
        ", right_scores=", _show_summary(f.right_scores),
        ", settings=", f.settings,
        ", cross_covariance=", f.cross_covariance, ", fpca1=", f.fpca1, ", fpca2=", f.fpca2, ")")
end

"""
    concurrent_correlation(fd1, fd2; bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), grid=nothing)

Concurrent correlation of simultaneously observed paired processes, following
fdapace 0.6.0 `FCCor`. Observation times must agree within each subject;
unmatched observations throw. `bandwidth` is required: one positive number or
five in mean1, mean2, variance1, variance2, cross-product order. The source's
actual default kernel is Epanechnikov, despite its help text saying Gaussian.
`grid=nothing` uses sorted unique observation times. Five local linear smooths
use KernelSmoothing with observation weights one. Means are estimated at the
pooled observed times before residual products are smoothed on the output
grid. This assumes no measurement error. Nonpositive variance estimates yield
NaN and a warning; affected points are recorded separately for each process.
The copied result retains all grids, five bandwidths, kernel, degree, means,
variances, covariance, correlation, and affected masks. No clipping is applied.
"""
function concurrent_correlation(fd1::Union{GridFunctionalData,IrregularFunctionalData},
        fd2::Union{GridFunctionalData,IrregularFunctionalData}; bandwidth,
        kernel=KernelSmoothing.EpanechnikovKernel(), grid=nothing)
    _cross_subjects(fd1, fd2)
    t1, y1, w, _ = _mean_pool(fd1, :obs)
    t2, y2, _, _ = _mean_pool(fd2, :obs)
    all(i -> fd1[i][1] == fd2[i][1], 1:nobs(fd1)) || throw(ArgumentError(
        "concurrent correlation requires identical observation times within every paired subject"))
    t1 == t2 || throw(ArgumentError("concurrent correlation observation times must agree"))
    bandwidth isa Real || bandwidth isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("concurrent correlation bandwidth must be a number or a vector of length one or five"))
    hs = bandwidth isa Real ? [bandwidth] : collect(bandwidth)
    length(hs) == 1 && (hs = fill(only(hs), 5))
    length(hs) == 5 || throw(ArgumentError("concurrent correlation bandwidth must have length one or five"))
    methods = map(h -> LocalPolynomial(; bandwidth=h, kernel, degree=1), hs)
    tall = sort(unique(t1))
    points = _pace_input_grid(grid === nothing ? tall : grid, "concurrent correlation")
    first(tall) <= first(points) && last(points) <= last(tall) || throw(ArgumentError(
        "concurrent correlation grid must lie inside the observed time range"))
    mu1 = _mean_predict(t1, y1, w, tall, methods[1])
    mu2 = _mean_predict(t1, y2, w, tall, methods[2])
    index = searchsortedfirst.(Ref(tall), t1)
    r1, r2 = y1 - mu1[index], y2 - mu2[index]
    v1 = _mean_predict(t1, r1.^2, w, points, methods[3]; label="concurrent variance1")
    v2 = _mean_predict(t1, r2.^2, w, points, methods[4]; label="concurrent variance2")
    cov = _mean_predict(t1, r1.*r2, w, points, methods[5]; label="concurrent covariance")
    bad1, bad2 = v1 .<= 0, v2 .<= 0
    if any(bad1) || any(bad2)
        @warn "NaN produced because a concurrent variance estimate is nonpositive"
        v1[bad1] .= NaN
        v2[bad2] .= NaN
    end
    (; grid=copy(points), correlation=cov ./ sqrt.(v1.*v2),
        bandwidth=map(m -> m.bandwidth, methods), kernel=deepcopy(kernel), degree=1,
        mean_grid=tall, mean1=mu1, mean2=mu2, variance1=v1, variance2=v2,
        covariance=cov, affected1=bad1, affected2=bad2)
end

"""
    dynamic_correlation(fd1::GridFunctionalData, fd2::GridFunctionalData)

Dynamical correlation of paired complete curves on the same grid, following
fdapace 0.6.0 `DynCorr` (Dubin and Müller, 2005). First subtract each curve's
time average, then the cross-sectional mean of those centered curves. Divide
each residual curve by its time RMS, integrate the paired standardized curves,
and average the individual correlations over subjects. All integrals use the
stored quadrature weights divided by the domain length. Both individual
correlations and their mean are returned, with copied standardized curves,
weights and grid. The source has no tuning defaults. Missing data throw;
unlike R, this routine does not silently interpolate them. A zero residual
RMS throws an informative error instead of returning an undefined correlation.
"""
function dynamic_correlation(fd1::GridFunctionalData, fd2::GridFunctionalData)
    _cross_subjects(fd1, fd2)
    fd1.observed === nothing && fd2.observed === nothing || throw(ArgumentError(
        "dynamic correlation requires complete curves; convert explicitly before fitting"))
    fd1.grid == fd2.grid && fd1.weights == fd2.weights || throw(ArgumentError(
        "dynamic correlation requires identical grids and quadrature weights"))
    nobs(fd1) >= 2 || throw(ArgumentError("dynamic correlation requires at least two subjects"))
    w = fd1.weights ./ (last(fd1.grid)-first(fd1.grid))
    function standardize(y)
        centered = y .- y*w
        residual = centered .- sum(centered; dims=1)./size(y,1)
        scale = sqrt.((residual.^2)*w)
        all(s -> isfinite(s) && s > 0, scale) || throw(ArgumentError(
            "dynamic correlation residual time RMS must be finite and positive for every subject"))
        z = residual ./ scale
        all(isfinite, z) || throw(ArgumentError("dynamic correlation standardized curves must be finite"))
        z
    end
    z1, z2 = standardize(fd1.values), standardize(fd2.values)
    individual = (z1 .* z2)*w
    (; correlation=sum(individual)/length(individual), individual,
        grid=copy(fd1.grid), weights=copy(fd1.weights), standardized1=z1, standardized2=z2,
        centering=:curve_then_population, normalization=:time_rms)
end
