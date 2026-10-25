# Gaussian penalized least squares with several quadratic penalties. Smoothing
# parameters are chosen by REML or GCV on log(lambda) with Newton steps, after
# mgcv 1.9-3 (Wood 2011, JRSSB 73:3-36; Wood 2017, Generalized Additive Models).

"""
    PenalizedLeastSquares

Result of `StatsAPI.fit(PenalizedLeastSquares, X, y; penalties, ...)`, which
minimizes

    ‖W^{1/2}(y - Xβ)‖² + Σⱼ λⱼ βᵀSⱼβ

over `β` for known positive observation weights `W`. Each penalty `Sⱼ` is
symmetric positive semidefinite, may be rank deficient, and acts on a block of
columns. Several penalties may share columns (tensor products); columns in no
penalty are unpenalized.

    fit(PenalizedLeastSquares, X, y; penalties, weights=nothing,
        criterion=:REML, lambda=nothing, lower=nothing, upper=nothing,
        rank_tol=eps(T)^(3/4), tol=1e-8, maxiter=200)

`penalties` is a vector whose entries are `S => columns` pairs (a square matrix
and the distinct indices of the columns of `X` it acts on), or full
`size(X, 2)`-square matrices. `weights` defaults to ones.

`criterion=:REML` (the default, as in refund's `pfr`/`pffr`) or `:GCV` (as fda
uses) selects mgcv's `method="REML"` or `method="GCV.Cp"` with unknown scale:

- REML: `V = Dₚ/(2φ) + (n-Mₚ)/2 log(2πφ) + log|XᵀWX + S_λ|/2 - log|S_λ|₊/2
  - Σ log wᵢ/2`, with `Dₚ` the penalized weighted residual sum of squares,
  `Mₚ` the dimension of the null space of `S_λ = Σ λⱼSⱼ` and `φ` profiled out at
  `Dₚ/(n-Mₚ)`.
- GCV: `V = n D/(n - τ)²`, with `D` the weighted residual sum of squares and
  `τ = tr(A)` the trace of the influence matrix.

Smoothing parameters are fixed by passing `lambda` (finite nonnegative values),
or optimized on `log λ` within `lower` and `upper` (defaults `0` and `Inf`; a
parameter with equal bounds is fixed). Newton's method uses exact first and
second derivatives, eigenvalue-floored Hessians, steps of at most 5 on the log
scale and up to 30 step halvings (mgcv `newton` defaults). It starts from
mgcv's `initial.sp` heuristic applied to the weighted design, clamped to the
bounds. It stops (`settings.convergence`) at `:gradient` when every free
gradient entry is at most `tol` times `1 + |V|` (REML) or `|V|` (GCV) and no
parameter without curvature can still decrease `V`, at `:no_decrease` when no
halved step decreases `V`, or at `:maxiter`. `tol=1e-8` is tighter than mgcv's
`conv.tol=1e-6`, so that the optimum is at least as good as mgcv's;
`maxiter=200` and `rank_tol` are package choices. Determinants and solves use
Wood's (2011, appendix B) similarity transform for every group of overlapping
penalties followed by diagonally scaled Cholesky factors, which keeps them
accurate when smoothing parameters differ by many orders of magnitude.

A direction the data do not identify drives its `λⱼ` towards infinity (or, for
GCV, zero). That parameter then moves in steps of 5 on the log scale while `V`
still decreases in working precision, up to `e^60` times its initial value,
and `lambda_status` records
`:effectively_infinite` (or `:effectively_zero`). Other statuses are
`:estimated`, `:fixed`, `:lower_bound` and `:upper_bound`. `rank_tol` is the
relative eigenvalue threshold that determines the null space of the penalties.
Responses fitted exactly (penalized residual sum of squares at rounding level)
leave no scale to estimate, and selection throws an `ArgumentError`; supply
`lambda` instead. With every `λⱼ` fixed the criterion is only reported: where
it is undefined (REML with `n ≤ Mₚ` or `Dₚ = 0`, GCV with `n - τ ≈ 0`),
`criterion_value` is `NaN` and `settings.criterion_defined` is `false`, while
`β`, the EDF and the fitted values are still computed. `sigma2` is `NaN` when
`n - τ ≈ 0` (and then so are both covariances), `reml_scale` when `n ≤ Mₚ`.
Designs that do not identify the penalties' null space throw.

Fields:

- `coef`: `β`; `fitted`, `residuals` (response scale), `weights`;
- `lambda`, `lambda_status`, `criterion`, `criterion_value`;
- `edf` (per coefficient, the diagonal of `F = (XᵀWX + S_λ)⁻¹XᵀWX`, mgcv
  `edf`), `edf_total = tr(F)`, the trace of the influence matrix (mgcv
  `sum(edf)`), and `edf_penalty`, with entry `j` the sum of `edf` over the
  columns penalty `j` acts on. Penalties on the same block (a tensor product)
  each report that block's EDF once, undivided, so `edf_penalty` need not sum
  to `edf_total`; the EDF of the distinct blocks plus the unpenalized columns
  does;
- `sigma2`: `D/(n - τ)`, the scale mgcv reports as `sig2` for both criteria;
  `reml_scale`: `Dₚ/(n - Mₚ)`;
- `vcov_bayesian`: `(XᵀWX + S_λ)⁻¹ sigma2` (mgcv `Vp`); `vcov_frequentist`:
  `F (XᵀWX + S_λ)⁻¹ sigma2` (mgcv `Ve`);
- `null_space_dimension` (`Mₚ`), copied `penalties` and `penalty_columns`;
- `settings`: weights source, bounds, initial values, tolerances, iterations,
  convergence, whether the criterion is defined (`criterion_defined`), final
  gradient and Hessian on the log scale.

`coef`, `vcov` (`type=:bayesian` or `:frequentist`), `fitted`, `residuals`,
`nobs`, `dof` (`edf_total + 1`, counting the scale), `dof_residual`
(`n - edf_total`) and `predict(fit, Xnew)` follow StatsAPI.
"""
struct PenalizedLeastSquares{T<:AbstractFloat,S<:NamedTuple}
    coef::Vector{T}
    lambda::Vector{T}
    lambda_status::Vector{Symbol}
    criterion::Symbol
    criterion_value::T
    edf::Vector{T}
    edf_total::T
    edf_penalty::Vector{T}
    sigma2::T
    reml_scale::T
    vcov_bayesian::Matrix{T}
    vcov_frequentist::Matrix{T}
    fitted::Vector{T}
    residuals::Vector{T}
    weights::Vector{T}
    nobs::Int
    null_space_dimension::Int
    penalties::Vector{Matrix{T}}
    penalty_columns::Vector{Vector{Int}}
    settings::S
end

# Penalties that share columns, in group coordinates.
struct _PenaltyGroup{T}
    columns::Vector{Int}
    members::Vector{Int}
    matrices::Vector{Matrix{T}}
    scales::Vector{T}
end

# One independent Gaussian block: RᵀR = XᵀWX and Rᵀf = XᵀWy.
struct _PLSBlock{T}
    R::Matrix{T}
    f::Vector{T}
end

# Blocks share coefficients' layout, penalties, smoothing parameters and scale;
# each has its own coefficients. A regression has one block.
struct _PLSProblem{T}
    blocks::Vector{_PLSBlock{T}}
    nobs::Int
    rss0::T
    logweights::T
    p::Int
    penalties::Vector{Matrix{T}}
    columns::Vector{Vector{Int}}
    groups::Vector{_PenaltyGroup{T}}
    unpenalized::Vector{Int}
    rank_tol::T
end

const _PLS_MAX_STEP = 5.0
const _PLS_MAX_HALVINGS = 30
const _PLS_LOG_RANGE = 60.0

_pls_root(parent, i) = parent[i] == i ? i : _pls_root(parent, parent[i])

function _penalty_groups(penalties::Vector{Matrix{T}}, columns, p) where {T}
    parent = collect(eachindex(penalties))
    owner = zeros(Int, p)
    for (j, cols) in enumerate(columns), c in cols
        if owner[c] == 0
            owner[c] = j
        else
            a, b = _pls_root(parent, owner[c]), _pls_root(parent, j)
            a == b || (parent[max(a, b)] = min(a, b))
        end
    end
    roots = unique(_pls_root(parent, j) for j in eachindex(penalties))
    groups = map(roots) do root
        members = [j for j in eachindex(penalties) if _pls_root(parent, j) == root]
        cols = sort!(unique!(reduce(vcat, columns[members])))
        matrices = map(members) do j
            out = zeros(T, length(cols), length(cols))
            position = indexin(columns[j], cols)
            out[position, position] .= penalties[j]
            out
        end
        _PenaltyGroup{T}(cols, members, matrices, T[norm(m) for m in matrices])
    end
    groups, findall(iszero, owner)
end

# Wood (2011, appendix B): order the group's coordinates as nested ranges of
# penalties of decreasing magnitude, then the null space of all active
# penalties. Ranks use penalties scaled by their own norms, so they do not
# depend on lambda. Return the orthogonal basis, the range dimension, and for
# each member the number of leading coordinates outside which it vanishes.
function _stable_group_basis(group::_PenaltyGroup{T}, lambda, rank_tol) where {T}
    m = length(group.columns)
    Q = Matrix{T}(I, m, m)
    basis = Matrix{T}(undef, m, m)
    active = [i for i in eachindex(group.members) if lambda[group.members[i]] > 0]
    extent = zeros(Int, length(group.members))
    rank::Int = 0
    dominance = eps(T)^(1/3)
    while !isempty(active) && size(Q, 2) > 0
        projected = [Matrix(Symmetric(transpose(Q) * group.matrices[i] * Q)) ./ group.scales[i]
            for i in active]
        sizes = T[norm(a) for a in projected]
        alive = sizes .> rank_tol
        extent[active[.!alive]] .= rank
        any(alive) || (active = Int[]; break)
        active, projected, sizes = active[alive], projected[alive], sizes[alive]
        magnitude = T[lambda[group.members[i]] * group.scales[i] * s for (i, s) in zip(active, sizes)]
        dominant = magnitude .>= dominance * maximum(magnitude)
        combined = sum(projected[dominant])
        decomposition = eigen(Symmetric(combined))
        r = count(>(rank_tol), decomposition.values)
        r == 0 && break
        k = size(Q, 2)
        basis[:, (rank+1):(rank+r)] .= Q * decomposition.vectors[:, (k-r+1):k]
        Q = Q * decomposition.vectors[:, 1:(k-r)]
        rank += r
        extent[active[dominant]] .= rank
        active = active[.!dominant]
    end
    extent[active] .= rank
    basis[:, (rank+1):m] .= Q
    basis, rank, extent
end

# Diagonally scaled Cholesky factor: accurate for graded positive definite
# matrices whose scale varies by orders of magnitude between coordinates.
function _scaled_cholesky(M::AbstractMatrix{T}) where {T}
    d = diag(M)
    all(x -> isfinite(x) && x > 0, d) || return nothing
    s = inv.(sqrt.(d))
    factor = cholesky(Symmetric(s .* M .* transpose(s)); check=false)
    issuccess(factor) || return nothing
    (; factor, s, logdet=logdet(factor) - 2 * sum(log, s))
end

_scaled_solve(F, b) = F.s .* (F.factor \ (F.s .* b))
_scaled_inverse(F) = F.s .* inv(F.factor) .* transpose(F.s)

# tr(A*B) without forming the product.
_trace_product(A, B) = sum(A .* transpose(B))

function _pls_basis(problem::_PLSProblem{T}, lambda) where {T}
    p = problem.p
    basis = zeros(T, p, p)
    ranges = UnitRange{Int}[]
    support = fill(1:0, length(problem.penalties))
    column, rank = 0, 0
    for group in problem.groups
        local_basis, r, extent = _stable_group_basis(group, lambda, problem.rank_tol)
        m = length(group.columns)
        basis[group.columns, (column+1):(column+m)] .= local_basis
        push!(ranges, (column+1):(column+r))
        for (i, j) in enumerate(group.members)
            support[j] = (column+1):(column+extent[i])
        end
        column += m
        rank += r
    end
    for c in problem.unpenalized
        column += 1
        basis[c, column] = one(T)
    end
    basis, ranges, rank, support
end

# Criterion, and optionally its gradient and Hessian on log(lambda), at lambda.
# Return nothing when a factorization fails (an inadmissible lambda), or when
# the criterion is undefined (n <= Mp or Dp = 0 for REML, n - tau ~ 0 for GCV)
# unless `undefined_ok`, in which case the value and derivatives are NaN.
function _pls_evaluate(problem::_PLSProblem{T}, lambda::Vector{T}, criterion::Symbol;
        derivatives::Bool=true, solution::Bool=false, undefined_ok::Bool=false) where {T}
    p, M = problem.p, length(lambda)
    basis, ranges, rank, support = _pls_basis(problem, lambda)
    # Each transformed penalty is exactly zero outside its leading coordinates;
    # dropping the rounding there keeps huge lambdas out of smaller blocks.
    A = map(1:M) do j
        out = zeros(T, p, p)
        lambda[j] == 0 && return out
        rows = view(basis, problem.columns[j], support[j])
        out[support[j], support[j]] .= Symmetric(lambda[j] .* (transpose(rows) * problem.penalties[j] * rows))
        out
    end
    Slambda = sum(A)
    logdet_penalty = zero(T)
    gradient_penalty = zeros(T, M)
    hessian_penalty = zeros(T, M, M)
    for (group, range) in zip(problem.groups, ranges)
        isempty(range) && continue
        F = _scaled_cholesky(Slambda[range, range])
        F === nothing && return nothing
        logdet_penalty += F.logdet
        derivatives || continue
        inverse = _scaled_inverse(F)
        B = [inverse * A[j][range, range] for j in group.members]
        for (a, j) in enumerate(group.members)
            gradient_penalty[j] = tr(B[a])
            for (b, k) in enumerate(group.members)
                hessian_penalty[j, k] = -_trace_product(B[a], B[b])
            end
            hessian_penalty[j, j] += gradient_penalty[j]
        end
    end
    nblocks = length(problem.blocks)
    null_dimension = p - rank
    logdet_hessian, rss, pen, tau = zero(T), problem.rss0, zero(T), zero(T)
    grad_logdet, hess_logdet = zeros(T, M), zeros(T, M, M)
    grad_pen, hess_pen = zeros(T, M), zeros(T, M, M)
    grad_rss, hess_rss = zeros(T, M), zeros(T, M, M)
    grad_tau, hess_tau = zeros(T, M), zeros(T, M, M)
    coefficients = Vector{T}[]
    inverses = Matrix{T}[]
    influences = Matrix{T}[]
    for block in problem.blocks
        RT = block.R * basis
        XtX = Matrix(Symmetric(transpose(RT) * RT))
        F = _scaled_cholesky(XtX + Slambda)
        F === nothing && return nothing
        logdet_hessian += F.logdet
        beta = _scaled_solve(F, transpose(RT) * block.f)
        rss += sum(abs2, block.f - RT * beta)
        Sbeta = Slambda * beta
        pen += dot(beta, Sbeta)
        inverse = _scaled_inverse(F)
        influence = inverse * XtX
        tau += tr(influence)
        if solution
            push!(coefficients, basis * beta)
            push!(inverses, inverse)
            push!(influences, influence)
        end
        derivatives || continue
        E = [inverse * A[j] for j in 1:M]
        v = [E[j] * beta for j in 1:M]
        Abeta = [A[j] * beta for j in 1:M]
        for j in 1:M
            grad_logdet[j] += tr(E[j])
            grad_pen[j] += dot(beta, Abeta[j])
            for k in 1:M
                hess_logdet[j, k] -= _trace_product(E[j], E[k])
                hess_pen[j, k] -= 2 * dot(Abeta[j], v[k])
            end
            hess_logdet[j, j] += tr(E[j])
            hess_pen[j, j] += dot(beta, Abeta[j])
        end
        criterion === :GCV || continue
        EF = [E[j] * influence for j in 1:M]
        for j in 1:M
            grad_tau[j] -= tr(EF[j])
            grad_rss[j] += 2 * dot(Sbeta, v[j])
            for k in 1:M
                hess_tau[j, k] += _trace_product(E[k], EF[j]) + _trace_product(E[j], EF[k])
                hess_rss[j, k] += 2 * dot(v[k], XtX * v[j]) -
                    2 * dot(Sbeta, E[k] * v[j] + E[j] * v[k])
            end
            hess_tau[j, j] -= tr(EF[j])
            hess_rss[j, j] += 2 * dot(Sbeta, v[j])
        end
    end
    n = problem.nobs
    residual_df = n - tau
    nu = n - nblocks * null_dimension
    Dp = rss + pen
    defined = criterion === :REML ? nu > 0 && Dp > 0 : residual_df > sqrt(eps(T)) * n
    if !defined
        undefined_ok || return nothing
        value, gradient, hessian = T(NaN), fill(T(NaN), M), fill(T(NaN), M, M)
    elseif criterion === :REML
        value = nu / 2 * (1 + log(2 * T(pi) * Dp / nu)) + logdet_hessian / 2 -
            nblocks * logdet_penalty / 2 - problem.logweights / 2
        gradient = nu / 2 .* grad_pen ./ Dp .+ grad_logdet ./ 2 .- nblocks .* gradient_penalty ./ 2
        hessian = nu / 2 .* (hess_pen ./ Dp .- grad_pen .* transpose(grad_pen) ./ Dp^2) .+
            hess_logdet ./ 2 .- nblocks .* hessian_penalty ./ 2
    else
        delta = residual_df
        value = n * rss / delta^2
        gradient = n .* grad_rss ./ delta^2 .+ 2n * rss .* grad_tau ./ delta^3
        hessian = n .* hess_rss ./ delta^2 .+
            2n .* (grad_rss .* transpose(grad_tau) .+ grad_tau .* transpose(grad_rss)) ./ delta^3 .+
            2n * rss .* hess_tau ./ delta^3 .+ 6n * rss .* grad_tau .* transpose(grad_tau) ./ delta^4
    end
    if defined && !isfinite(value)
        undefined_ok || return nothing
        defined = false
        value, gradient, hessian = T(NaN), fill(T(NaN), M), fill(T(NaN), M, M)
    end
    (; value, defined, gradient, hessian=Matrix(Symmetric((hessian .+ transpose(hessian)) ./ 2)),
        rss, pen, tau, null_dimension, basis, coefficients, inverses, influences)
end

# mgcv 1.9-3 initial.sp (expensive=FALSE), on the weighted design.
function _pls_initial_lambda(problem::_PLSProblem{T}) where {T}
    xx = zeros(T, problem.p)
    for block in problem.blocks
        xx .+= vec(sum(abs2, block.R; dims=1))
    end
    M = length(problem.penalties)
    initial = zeros(T, M)
    ss = zeros(T, problem.p)
    penalized = falses(problem.p)
    for (j, S) in enumerate(problem.penalties)
        threshold = eps(T)^(4/5) * maximum(abs, S)
        keep = (vec(sum(abs, S; dims=2)) ./ size(S, 1) .> threshold) .&
            (vec(sum(abs, S; dims=1)) ./ size(S, 2) .> threshold) .& (abs.(diag(S)) .> threshold)
        cols = problem.columns[j]
        size_x = sum(xx[cols[keep]]) / count(keep)
        size_s = sum(diag(S)[keep]) / count(keep)
        initial[j] = size_x / size_s
        # A zero design block has no mgcv scale; start that parameter at one.
        isfinite(initial[j]) && initial[j] > 0 || (initial[j] = one(T))
        penalized[cols] .|= keep
        ss[cols] .+= initial[j] .* diag(S)
    end
    use = (ss .> 0) .& penalized .& (xx .> 0)
    any(use) || return initial
    ratio(scale) = sum(xx[use] ./ (xx[use] .+ scale .* ss[use])) / count(use)
    scale = one(T)
    while ratio(scale) > T(0.4) && isfinite(scale)
        scale *= 10
    end
    while ratio(scale) < T(0.4) && scale > 0
        scale /= 10
    end
    initial .* scale
end

# Positive definite Newton direction: absolute eigenvalues with a relative floor.
function _pls_newton_direction(H, g)
    decomposition = eigen(Symmetric(H))
    values = abs.(decomposition.values)
    floor_value = max(maximum(values; init=zero(eltype(values))) * sqrt(eps(eltype(values))),
        floatmin(eltype(values)))
    values = max.(values, floor_value)
    -(decomposition.vectors * ((transpose(decomposition.vectors) * g) ./ values))
end

function _pls_criterion_scale(criterion, value)
    criterion === :REML ? 1 + abs(value) : abs(value)
end

function _pls_optimize(problem::_PLSProblem{T}, criterion, rho0, lo, hi, free; tol, maxiter) where {T}
    rho = copy(rho0)
    current = _pls_evaluate(problem, exp.(rho), criterion)
    current === nothing && throw(ArgumentError(
        "penalized fit is singular at the initial smoothing parameters; check that the data identify the penalty null space"))
    iterations, convergence = 0, :gradient
    frozen = falses(length(rho))
    while true
        g = current.gradient
        threshold = tol * _pls_criterion_scale(criterion, current.value)
        active = [j for j in free if !frozen[j] &&
            !(rho[j] <= lo[j] && g[j] > 0) && !(rho[j] >= hi[j] && g[j] < 0)]
        converged = isempty(active) || maximum(abs, g[active]) <= threshold
        # Parameters without curvature head to zero or infinity. Keep moving
        # them while the criterion still decreases in working precision.
        flat = converged ? [j for j in active if abs(current.hessian[j, j]) <= threshold] : Int[]
        if converged && isempty(flat)
            convergence = :gradient
            break
        end
        if iterations >= maxiter
            convergence = :maxiter
            break
        end
        iterations += 1
        if converged
            trial_rho = copy(rho)
            trial_rho[flat] .-= _PLS_MAX_STEP .* sign.(g[flat])
            trial_rho = clamp.(trial_rho, lo, hi)
            trial = _pls_evaluate(problem, exp.(trial_rho), criterion)
            if trial !== nothing && trial.value < current.value
                rho, current = trial_rho, trial
            else
                frozen[flat] .= true
            end
            continue
        end
        step = zeros(T, length(rho))
        step[active] = _pls_newton_direction(current.hessian[active, active], g[active])
        biggest = maximum(abs, step)
        biggest > _PLS_MAX_STEP && (step .*= _PLS_MAX_STEP / biggest)
        accepted = false
        for _ in 0:_PLS_MAX_HALVINGS
            trial_rho = clamp.(rho .+ step, lo, hi)
            trial = _pls_evaluate(problem, exp.(trial_rho), criterion)
            if trial !== nothing && (trial.value < current.value || (trial.value <= current.value &&
                    maximum(abs, trial.gradient[active]) < maximum(abs, g[active])))
                rho, current, accepted = trial_rho, trial, true
                break
            end
            step ./= 2
        end
        if !accepted
            convergence = :no_decrease
            break
        end
    end
    rho, current, iterations, convergence
end

function _pls_penalty_input(entry, p, ::Type{T}) where {T}
    S, columns = if entry isa Pair
        first(entry), last(entry)
    elseif entry isa AbstractMatrix
        entry, 1:p
    else
        throw(ArgumentError("each penalty must be a matrix or a matrix => columns pair"))
    end
    S isa AbstractMatrix{<:Real} || throw(ArgumentError("penalty matrices must be real matrices"))
    columns isa AbstractVector{<:Integer} || throw(ArgumentError("penalty columns must be a vector of integers"))
    Base.require_one_based_indexing(S, columns)
    cols = Vector{Int}(columns)
    isempty(cols) && throw(ArgumentError("penalty columns must be nonempty"))
    allunique(cols) || throw(ArgumentError("penalty columns must be distinct"))
    all(c -> 1 <= c <= p, cols) || throw(ArgumentError("penalty columns must index columns of X (1:$p)"))
    size(S) == (length(cols), length(cols)) || throw(DimensionMismatch(
        "penalty matrix is $(size(S)) but acts on $(length(cols)) columns"))
    all(isfinite, S) || throw(ArgumentError("penalty matrices must be finite"))
    matrix = Matrix{T}(S)
    scale = maximum(abs, matrix)
    scale > 0 || throw(ArgumentError("penalty matrices must not be zero"))
    maximum(abs, matrix - transpose(matrix)) <= sqrt(eps(T)) * scale ||
        throw(ArgumentError("penalty matrices must be symmetric"))
    matrix = (matrix + transpose(matrix)) / 2
    minimum(eigvals(Symmetric(matrix))) >= -sqrt(eps(T)) * scale ||
        throw(ArgumentError("penalty matrices must be positive semidefinite"))
    matrix, cols
end

function _pls_bounds(lambda, lower, upper, M, ::Type{T}) where {T}
    if lambda !== nothing
        lower === nothing && upper === nothing ||
            throw(ArgumentError("supply either fixed lambda or lower/upper bounds, not both"))
        lambda isa AbstractVector{<:Real} && length(lambda) == M ||
            throw(DimensionMismatch("lambda must be a real vector with one value per penalty ($M)"))
        all(x -> isfinite(x) && x >= 0, lambda) ||
            throw(ArgumentError("fixed lambda must be finite and nonnegative"))
        return Vector{T}(lambda), Vector{T}(lambda)
    end
    lo = lower === nothing ? zeros(T, M) : lower
    hi = upper === nothing ? fill(T(Inf), M) : upper
    for (name, bound) in (("lower", lo), ("upper", hi))
        bound isa AbstractVector{<:Real} && length(bound) == M ||
            throw(DimensionMismatch("$name must be a real vector with one value per penalty ($M)"))
    end
    all(x -> isfinite(x) && x >= 0, lo) || throw(ArgumentError("lower bounds must be finite and nonnegative"))
    all(x -> !isnan(x) && x > 0, hi) || throw(ArgumentError("upper bounds must be positive"))
    all(lo .<= hi) || throw(ArgumentError("lower bounds must not exceed upper bounds"))
    all(i -> lo[i] < hi[i] || isfinite(hi[i]), 1:M) ||
        throw(ArgumentError("equal bounds must be finite"))
    Vector{T}(lo), Vector{T}(hi)
end

function _pls_criterion(criterion)
    criterion in (:REML, :GCV) || throw(ArgumentError("criterion must be :REML or :GCV"))
    criterion
end

function _pls_rank_tol(rank_tol, ::Type{T}) where {T}
    value = rank_tol === nothing ? eps(T)^(3/4) : rank_tol
    value isa Real && isfinite(value) && 0 < value < 1 ||
        throw(ArgumentError("rank_tol must be in (0, 1)"))
    T(value)
end

# Group ranks of the active penalties, independent of lambda's size.
function _pls_reference_rank(problem::_PLSProblem{T}, active) where {T}
    rank = 0
    for group in problem.groups
        members = [i for i in eachindex(group.members) if active[group.members[i]]]
        isempty(members) && continue
        combined = sum(group.matrices[i] ./ group.scales[i] for i in members)
        rank += count(>(problem.rank_tol), eigvals(Symmetric(combined)))
    end
    rank
end

function _pls_problem(blocks::Vector{_PLSBlock{T}}, nobs, rss0, logweights, p, penalties,
        columns, rank_tol) where {T}
    groups, unpenalized = _penalty_groups(penalties, columns, p)
    _PLSProblem{T}(blocks, nobs, rss0, logweights, p, penalties, columns, groups, unpenalized, rank_tol)
end

# Block sufficient statistics from a pivot-free Householder QR of W^{1/2}X.
function _pls_block(X::Matrix{T}, y::Vector{T}) where {T}
    factor = qr(X)
    q = factor.Q' * y
    k = min(size(X)...)
    R = Matrix(factor.R)[1:k, :]
    _PLSBlock{T}(R, q[1:k]), sum(abs2, view(q, (k+1):length(q)))
end

# Optimize (or fix) smoothing parameters. Return lambda, statuses, the final
# evaluation (with solution) and optimizer settings.
function _pls_select(problem::_PLSProblem{T}, criterion, lo, hi; tol, maxiter) where {T}
    M = length(problem.penalties)
    rank = _pls_reference_rank(problem, hi .> 0)
    initial = clamp.(_pls_initial_lambda(problem), lo, hi)
    fixed = lo .== hi
    free = findall(!, fixed)
    rho0 = log.(initial)
    span = T(_PLS_LOG_RANGE)
    log_lo = T[fixed[j] ? rho0[j] : max(log(lo[j]), rho0[j] - span) for j in 1:M]
    log_hi = T[fixed[j] ? rho0[j] : min(log(hi[j]), rho0[j] + span) for j in 1:M]
    nblocks = length(problem.blocks)
    check_rank(evaluation) = evaluation === nothing ||
        problem.p - evaluation.null_dimension == rank || throw(ArgumentError(
            "penalty null space depends on lambda at rank_tol=$(problem.rank_tol); rescale the penalties or change rank_tol"))
    rho, iterations, convergence = rho0, 0, :fixed
    if !isempty(free)
        rho, current, iterations, convergence = _pls_optimize(problem, criterion, rho0,
            log_lo, log_hi, free; tol, maxiter)
        check_rank(current)
    end
    at_lower = [!fixed[j] && rho[j] <= log_lo[j] && lo[j] > 0 && log(lo[j]) >= log_lo[j] for j in 1:M]
    at_upper = [!fixed[j] && rho[j] >= log_hi[j] && isfinite(hi[j]) && log(hi[j]) <= log_hi[j] for j in 1:M]
    # A user bound is reported exactly, not as exp(log(bound)).
    lambda = [fixed[j] || at_lower[j] ? lo[j] : at_upper[j] ? hi[j] : exp(rho[j]) for j in 1:M]
    # With every lambda fixed the criterion is only reported, so an undefined
    # value (NaN) does not prevent the fit.
    final = _pls_evaluate(problem, lambda, criterion; solution=true, undefined_ok=isempty(free))
    final === nothing && throw(ArgumentError(
        "penalized fit is singular at the selected smoothing parameters; check that the data identify the penalty null space"))
    check_rank(final)
    # Residuals at rounding level leave REML and GCV without a noise scale.
    total = problem.rss0 + sum(block -> sum(abs2, block.f), problem.blocks)
    isempty(free) || final.rss + final.pen > eps(T)^(3/2) * total || throw(ArgumentError(
        "the responses are fitted exactly, so $criterion cannot choose smoothing parameters; supply lambda"))
    threshold = tol * _pls_criterion_scale(criterion, final.value)
    status = map(1:M) do j
        fixed[j] && return :fixed
        at_lower[j] && return :lower_bound
        at_upper[j] && return :upper_bound
        # Otherwise a reached limit is the internal e^±60 range.
        rho[j] <= log_lo[j] && return :effectively_zero
        rho[j] >= log_hi[j] && return :effectively_infinite
        flat = abs(final.gradient[j]) <= threshold && abs(final.hessian[j, j]) <= threshold
        flat ? (rho[j] > rho0[j] ? :effectively_infinite : :effectively_zero) : :estimated
    end
    settings = (; initial_lambda=initial, lower=copy(lo), upper=copy(hi), tol=T(tol),
        maxiter=Int(maxiter), rank_tol=problem.rank_tol, iterations, convergence,
        criterion_defined=final.defined,
        gradient=copy(final.gradient), hessian=copy(final.hessian),
        max_log_step=_PLS_MAX_STEP, max_halvings=_PLS_MAX_HALVINGS, log_range=_PLS_LOG_RANGE)
    lambda, status, final, settings, nblocks
end

function fit(::Type{PenalizedLeastSquares}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        penalties, weights=nothing, criterion::Symbol=:REML, lambda=nothing,
        lower=nothing, upper=nothing, rank_tol=nothing, tol::Real=1e-8, maxiter::Integer=200)
    Base.require_one_based_indexing(X, y)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("y has length $(length(y)) but X has $n rows"))
    n > 0 && p > 0 || throw(ArgumentError("X must have at least one row and one column"))
    weights === nothing || weights isa AbstractVector{<:Real} ||
        throw(ArgumentError("weights must be nothing or a real vector"))
    W = weights === nothing ? eltype(y) : eltype(weights)
    T = _floattype(promote_type(eltype(X), eltype(y), W))
    penalties isa AbstractVector || throw(ArgumentError("penalties must be a vector"))
    isempty(penalties) && throw(ArgumentError("supply at least one penalty"))
    criterion = _pls_criterion(criterion)
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    all(isfinite, X) || throw(ArgumentError("X must be finite"))
    all(isfinite, y) || throw(ArgumentError("y must be finite"))
    w = if weights === nothing
        ones(T, n)
    else
        Base.require_one_based_indexing(weights)
        length(weights) == n || throw(DimensionMismatch("weights have length $(length(weights)) but X has $n rows"))
        all(x -> isfinite(x) && x > 0, weights) || throw(ArgumentError("weights must be finite and positive"))
        Vector{T}(weights)
    end
    parsed = [_pls_penalty_input(entry, p, T) for entry in penalties]
    matrices = Matrix{T}[first(x) for x in parsed]
    columns = Vector{Int}[last(x) for x in parsed]
    lo, hi = _pls_bounds(lambda, lower, upper, length(matrices), T)
    Xc, yc = Matrix{T}(X), Vector{T}(y)
    root = sqrt.(w)
    block, rss0 = _pls_block(root .* Xc, root .* yc)
    problem = _pls_problem([block], n, rss0, sum(log, w), p, matrices, columns,
        _pls_rank_tol(rank_tol, T))
    selected, status, final, settings, _ = _pls_select(problem, criterion, lo, hi; tol, maxiter)
    beta = only(final.coefficients)
    basis = final.basis
    inverse = basis * only(final.inverses) * transpose(basis)
    influence = basis * only(final.influences) * transpose(basis)
    edf = diag(influence)
    edf_total = sum(edf)
    # No residual degrees of freedom leave the scale undefined (NaN).
    sigma2 = n - final.tau > sqrt(eps(T)) * n ? final.rss / (n - final.tau) : T(NaN)
    reml_scale = n > final.null_dimension ? (final.rss + final.pen) / (n - final.null_dimension) : T(NaN)
    Vp = Matrix(Symmetric(inverse .* sigma2))
    Ve = Matrix(Symmetric((influence * inverse) .* sigma2))
    fitted_values = Xc * beta
    settings = merge((; weights=weights === nothing ? :equal : :supplied), settings)
    PenalizedLeastSquares{T,typeof(settings)}(beta, selected, status, criterion, final.value,
        edf, edf_total, T[sum(edf[c]) for c in columns], sigma2, reml_scale, Vp, Ve,
        fitted_values, yc - fitted_values, w, n, final.null_dimension, matrices, columns, settings)
end

coef(f::PenalizedLeastSquares) = copy(f.coef)
fitted(f::PenalizedLeastSquares) = copy(f.fitted)
residuals(f::PenalizedLeastSquares) = copy(f.residuals)
nobs(f::PenalizedLeastSquares) = f.nobs
dof(f::PenalizedLeastSquares) = f.edf_total + 1
dof_residual(f::PenalizedLeastSquares) = f.nobs - f.edf_total

function vcov(f::PenalizedLeastSquares; type::Symbol=:bayesian)
    type === :bayesian && return copy(f.vcov_bayesian)
    type === :frequentist && return copy(f.vcov_frequentist)
    throw(ArgumentError("vcov type must be :bayesian or :frequentist"))
end

function predict(f::PenalizedLeastSquares, X::AbstractMatrix{<:Real})
    size(X, 2) == length(f.coef) || throw(DimensionMismatch(
        "X has $(size(X, 2)) columns but the fit has $(length(f.coef)) coefficients"))
    all(isfinite, X) || throw(ArgumentError("X must be finite"))
    X * f.coef
end

function Base.show(io::IO, f::PenalizedLeastSquares)
    print(io, "PenalizedLeastSquares(n=", f.nobs, ", p=", length(f.coef), ", criterion=",
        f.criterion, ", value=", f.criterion_value, ", lambda=", f.lambda, ", status=",
        f.lambda_status, ", edf=", f.edf_total, ", sigma2=", f.sigma2, ", null space=",
        f.null_space_dimension, ", settings=", f.settings, ")")
end

# Common-lambda REML for per-curve penalized splines: curves are independent
# blocks with their own coefficients, one smoothing parameter and one scale.
function _spline_reml_selection(fd, basis, order)
    T = promote_type(eltype(fd), eltype(basis))
    P = Matrix{T}(penalty(basis, order))
    k = nbasis(basis)
    blocks = _PLSBlock{T}[]
    rss0, total = zero(T), 0
    for i in 1:nobs(fd)
        t, y = fd[i]
        isempty(t) && throw(ArgumentError("curve $i has no observed values for REML"))
        block, extra = _pls_block(Matrix{T}(basis_matrix(basis, t)), Vector{T}(y))
        push!(blocks, block)
        rss0 += extra
        total += length(t)
    end
    maximum(abs, P) > 0 || throw(ArgumentError("REML needs a nonzero penalty; lower the penalty order"))
    problem = _pls_problem(blocks, total, rss0, zero(T), k, [Matrix(Symmetric(P))], [collect(1:k)],
        _pls_rank_tol(nothing, T))
    lambda, status, final, settings, nblocks = _pls_select(problem, :REML, zeros(T, 1), T[Inf];
        tol=1e-8, maxiter=200)
    coefs = permutedims(reduce(hcat, final.coefficients))
    # Effective degrees of freedom and null space refer to the stacked model.
    selection = (; criterion=:REML, lambda=only(lambda), status=only(status), score=final.value,
        sigma2=final.rss / (total - final.tau), edf=final.tau,
        null_space_dimension=nblocks * final.null_dimension, initial_lambda=only(settings.initial_lambda),
        tol=settings.tol, maxiter=settings.maxiter, rank_tol=settings.rank_tol,
        iterations=settings.iterations, convergence=settings.convergence)
    selection, coefs
end
