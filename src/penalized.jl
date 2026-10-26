# Penalized least squares with several quadratic penalties, and penalized IRLS
# for exponential-family responses. Smoothing parameters are chosen by REML
# (Laplace-approximate for non-Gaussian fits), GCV or UBRE on log(lambda) with
# Newton steps, after mgcv 1.9-3 (Wood 2011, JRSSB 73:3-36; Wood 2017,
# Generalized Additive Models).

"""
    PenalizedLeastSquares

Result of `StatsAPI.fit(PenalizedLeastSquares, X, y; penalties, ...)`, which
minimizes

    ‖W^{1/2}(y - Xβ)‖² + Σⱼ λⱼ βᵀSⱼβ

over `β` for known positive observation weights `W` or, for an
exponential-family response, the penalized deviance `D(β) + Σⱼ λⱼ βᵀSⱼβ` by
penalized iteratively reweighted least squares (PIRLS). Each penalty `Sⱼ` is
symmetric positive semidefinite, may be rank deficient, and acts on a block of
columns. Several penalties may share columns (tensor products); columns in no
penalty are unpenalized.

    fit(PenalizedLeastSquares, X, y; penalties, weights=nothing,
        family=:gaussian, link=nothing, criterion=:REML, lambda=nothing,
        lower=nothing, upper=nothing, rank_tol=eps(T)^(3/4), tol=1e-8,
        maxiter=200, irls_tol=1e-12, irls_maxiter=200)

`penalties` is a vector whose entries are `S => columns` pairs (a square matrix
and the distinct indices of the columns of `X` it acts on), or full
`size(X, 2)`-square matrices. `weights` defaults to ones (for the binomial
family they are the numbers of trials, `y` the proportions of successes).

`family` is `:gaussian` (default), `:binomial` or `:poisson`; `link=nothing`
selects the canonical link (`:identity`, `:logit`, `:log`). Other links are
`:log` and `:inverse` (Gaussian), `:probit` and `:cloglog` (binomial),
`:identity` and `:sqrt` (Poisson), as R's `family` objects. Binomial responses
must lie in `[0, 1]` with integer trials and successes, Poisson responses must
be nonnegative integers, and Gaussian responses must be positive for the log
link and nonzero for the inverse link (R's starting values); anything else
throws.

`criterion=:REML` (the default, as in refund's `pfr`/`pffr`), `:GCV` or
`:UBRE` selects mgcv 1.9-3's `method="REML"` or `method="GCV.Cp"`, which is GCV
when the scale is unknown (Gaussian family) and UBRE when it is known (one,
for the binomial and Poisson families); the other combinations throw:

- REML, Gaussian identity link: `V = Dₚ/(2φ) + (n-Mₚ)/2 log(2πφ) +
  log|XᵀWX + S_λ|/2 - log|S_λ|₊/2 - Σ log wᵢ/2`, with `Dₚ` the penalized
  weighted residual sum of squares, `Mₚ` the dimension of the null space of
  `S_λ = Σ λⱼSⱼ` and `φ` profiled out at `Dₚ/(n-Mₚ)`.
- Laplace-approximate REML otherwise: `V = Dₚ/(2φ) - ℓₛ + log|H|/2 -
  log|S_λ|₊/2 - Mₚ/2 log(2πφ)` with `Dₚ` the penalized deviance, `ℓₛ` the
  saturated log-likelihood, `H = XᵀWₙX + S_λ` the Hessian of `Dₚ/2` (Newton
  weights `Wₙ`, mgcv's `alpha`-weighted ones for non-canonical links), `φ = 1`
  for binomial and Poisson and profiled at `Dₚ/(n-Mₚ)` for the Gaussian family
  (mgcv optimizes `log φ` jointly; the minimum is the same).
- GCV: `V = n D/(n - τ)²`; UBRE: `V = D/n + 2τ/n - 1`; with `D` the
  (weighted) deviance and `τ = tr(A)` the trace of the influence matrix,
  computed with Fisher weights `W_F` as mgcv does.

PIRLS (mgcv `gam.fit3`) takes Newton steps on the penalized deviance, with
Fisher scoring when the Newton Hessian is not positive definite and step
halving while the penalized deviance does not decrease or the fit leaves the
family's domain. The first fit starts from R's family `mustart` with one Fisher
step, later fits from the previous coefficients. It stops once the Newton
decrement is at most `irls_tol` times `1 + Dₚ` (after that final step) and
throws after `irls_maxiter` iterations. A fit with fitted probabilities within
`10eps` of 0 or 1, or Poisson means within `10eps` of 0 (where mgcv and `glm`
warn), is rejected: the deviance is still decreasing along a diverging
direction, as when binomial classes are separated. During selection such
smoothing parameters are inadmissible; otherwise the fit throws. `irls_tol=1e-12` is tighter than
mgcv's `epsilon=1e-7`, a package choice so that coefficients are converged in
working precision.

Smoothing parameters are fixed by passing `lambda` (finite nonnegative values),
or optimized on `log λ` within `lower` and `upper` (defaults `0` and `Inf`; a
parameter with equal bounds is fixed). Newton's method uses exact first and
second derivatives (for PIRLS fits including the derivatives of the weights,
after Wood 2011), eigenvalue-floored Hessians, steps of at most 5 on the log
scale and up to 30 step halvings (mgcv `newton` defaults). It starts from
mgcv's `initial.sp` heuristic applied to the weighted design (`initial.spg`:
Fisher weights at the starting means for PIRLS fits), clamped to the bounds.
It stops (`settings.convergence`) at `:gradient` when every free gradient
entry is at most `tol` times `|V|` (GCV) or `1 + |V|` (REML, UBRE) and no
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
it is undefined (REML with `n ≤ Mₚ` or `Dₚ = 0` and estimated scale, GCV with
`n - τ ≈ 0`), `criterion_value` is `NaN` and `settings.criterion_defined` is
`false`, while `β`, the EDF and the fitted values are still computed. `sigma2`
is `NaN` when `n - τ ≈ 0` (and then so are both covariances), `reml_scale`
when `n ≤ Mₚ`. Designs that do not identify the penalties' null space throw.

Fields:

- `coef`: `β`; `fitted` (the means `μ = h(Xβ)`), `residuals` (response scale,
  `y - μ`), `weights`, `linear_predictors` (`Xβ`), `deviance` (`D`; the
  weighted residual sum of squares for the Gaussian family), `family`, `link`;
- `lambda`, `lambda_status`, `criterion`, `criterion_value`;
- `edf` (per coefficient, the diagonal of `F = (XᵀW_F X + S_λ)⁻¹XᵀW_F X`, mgcv
  `edf`), `edf_total = tr(F)`, the trace of the influence matrix (mgcv
  `sum(edf)`), and `edf_penalty`, with entry `j` the sum of `edf` over the
  columns penalty `j` acts on. Penalties on the same block (a tensor product)
  each report that block's EDF once, undivided, so `edf_penalty` need not sum
  to `edf_total`; the EDF of the distinct blocks plus the unpenalized columns
  does;
- `sigma2`: the scale mgcv reports as `sig2` for every criterion: one for the
  binomial and Poisson families, otherwise the Pearson statistic over
  `n - τ` (`D/(n - τ)` for the identity link); `reml_scale`: `Dₚ/(n - Mₚ)` for
  the Gaussian family, one otherwise;
- `vcov_bayesian`: `(XᵀW_F X + S_λ)⁻¹ sigma2` (mgcv `Vp`); `vcov_frequentist`:
  `F (XᵀW_F X + S_λ)⁻¹ sigma2` (mgcv `Ve`);
- `null_space_dimension` (`Mₚ`), copied `penalties` and `penalty_columns`;
- `settings`: weights source, bounds, initial values, tolerances, iterations,
  convergence, whether the criterion is defined (`criterion_defined`), final
  gradient and Hessian on the log scale, `scale` (`:known` or `:estimated`),
  `irls_tol`, `irls_maxiter` and the PIRLS iterations of the final fit
  (`irls_iterations`, zero for the closed-form Gaussian identity fit).

`coef`, `vcov` (`type=:bayesian` or `:frequentist`), `fitted`, `residuals`,
`nobs`, `dof` (`edf_total`, plus one for an estimated scale), `dof_residual`
(`n - edf_total`) and `predict(fit, Xnew; type=:link)` follow StatsAPI;
`type=:link` (mgcv `predict.gam`'s default) returns `Xβ`, `type=:response`
the means.
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
    linear_predictors::Vector{T}
    deviance::T
    family::Symbol
    link::Symbol
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

# One exponential-family regression for penalized IRLS: the design, prior
# weights, family, the constant of the criterion (`saturated`: the saturated
# log-likelihood, or Σ log wᵢ for the Gaussian family), and the coefficients
# of the last converged fit (`start`, original coordinates), which warm-start
# the next one.
struct _PIRLSProblem{T}
    X::Matrix{T}
    y::Vector{T}
    weights::Vector{T}
    family::_GLMFamily
    nobs::Int
    p::Int
    penalties::Vector{Matrix{T}}
    columns::Vector{Vector{Int}}
    groups::Vector{_PenaltyGroup{T}}
    unpenalized::Vector{Int}
    rank_tol::T
    saturated::T
    start::Vector{T}
    started::Base.RefValue{Bool}
    irls_tol::T
    irls_maxiter::Int
end

const _AnyPLSProblem{T} = Union{_PLSProblem{T},_PIRLSProblem{T}}

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

function _pls_basis(problem::_AnyPLSProblem{T}, lambda) where {T}
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
# Transformed penalties λⱼSⱼ, their sum and log|S_λ|₊ with derivatives on
# log(lambda); nothing when a penalty block is not positive definite.
function _pls_penalty_terms(problem, lambda::Vector{T}, derivatives::Bool) where {T}
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
    Slambda = M == 0 ? zeros(T, p, p) : sum(A)
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
    (; basis, rank, A, Slambda, logdet_penalty, gradient_penalty, hessian_penalty)
end

function _pls_evaluate(problem::_PLSProblem{T}, lambda::Vector{T}, criterion::Symbol;
        derivatives::Bool=true, solution::Bool=false, undefined_ok::Bool=false) where {T}
    p, M = problem.p, length(lambda)
    terms = _pls_penalty_terms(problem, lambda, derivatives)
    terms === nothing && return nothing
    (; basis, rank, A, Slambda, logdet_penalty, gradient_penalty, hessian_penalty) = terms
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

# Per-observation quantities of an exponential-family fit at linear predictor
# eta, and the penalized deviance; `valid` is false outside the family's domain.
function _pirls_state(problem::_PIRLSProblem{T}, eta::Vector{T}, beta::Vector{T}, Slambda) where {T}
    obs = map((y, w, e) -> _glm_observation(problem.family, y, w, e), problem.y, problem.weights, eta)
    valid = all(o -> o.valid, obs)
    dev = valid ? sum(o -> o.dev, obs) : T(NaN)
    Dp = dev + dot(beta, Slambda * beta)
    valid = valid && isfinite(Dp)
    (; obs, eta, beta, dev, Dp, valid)
end

_pirls_values(state, ::Val{name}) where {name} = [getfield(o, name) for o in state.obs]

# Weighted cross product Xᵀ diag(v) X.
_pirls_gram(X, v) = Matrix(Symmetric(transpose(X) * (v .* X)))

# Fitted probabilities numerically 0 or 1, or rates numerically 0 (gam.fit3's
# warning thresholds, 10 eps): the deviance is still decreasing along a
# diverging direction (separation), so the fit is rejected.
function _pirls_boundary(fam::_GLMFamily, state)
    limit = 10 * eps(eltype(state.eta))
    fam.family === :binomial && return any(o -> o.mu < limit || o.mu > 1 - limit, state.obs)
    fam.family === :poisson && return any(o -> o.mu < limit, state.obs)
    false
end

# Penalized IRLS (mgcv 1.9-3 gam.fit3): Newton's method on the penalized
# deviance Dp(β) = D(β) + βᵀS_λβ, with Fisher scoring where the Newton Hessian
# is not positive definite (as gam.fit3 does for non-canonical links) and step
# halving while Dp does not decrease or the fit leaves the family's domain. The
# first fit starts from R's family `mustart` with one Fisher step, later fits
# from the previous coefficients. Iterations stop once the Newton decrement
# gᵀH⁻¹g is at most `irls_tol` times 1 + |Dp| (after that final step); fits
# at the boundary (`_pirls_boundary`) are rejected.
function _pirls_fit(problem::_PIRLSProblem{T}, X::Matrix{T}, Slambda::Matrix{T}, beta0::Vector{T};
        warm::Bool=true) where {T}
    fam = problem.family
    state = nothing
    if warm
        state = _pirls_state(problem, X * beta0, beta0, Slambda)
        state.valid || (state = nothing)
    end
    if state === nothing
        mu = map((y, w) -> _glm_mustart(fam, y, w), problem.y, problem.weights)
        eta = map(m -> _glm_link(fam.link, m), mu)
        wf = zeros(T, length(eta))
        z = similar(eta)
        for i in eachindex(eta)
            m, mc, h1 = _glm_inverse_link(fam.link, eta[i])
            V = first(_glm_variance(fam, m, mc))
            wf[i] = problem.weights[i] * h1^2 / V
            z[i] = eta[i] + (problem.y[i] - m) / h1
        end
        F = _scaled_cholesky(_pirls_gram(X, wf) + Slambda)
        F === nothing && return nothing
        beta = _scaled_solve(F, transpose(X) * (wf .* z))
        state = _pirls_state(problem, X * beta, beta, Slambda)
        state.valid || return nothing
    end
    iterations = 0
    while true
        iterations > problem.irls_maxiter && return nothing
        iterations += 1
        g = transpose(X) * (_pirls_values(state, Val(:d1)) ./ 2) + Slambda * state.beta
        F = _scaled_cholesky(_pirls_gram(X, _pirls_values(state, Val(:wn))) + Slambda)
        if F === nothing && !_glm_canonical(fam)
            F = _scaled_cholesky(_pirls_gram(X, _pirls_values(state, Val(:wf))) + Slambda)
        end
        F === nothing && return nothing
        step = -_scaled_solve(F, g)
        decrement = -dot(g, step)
        isfinite(decrement) || return nothing
        converged = decrement <= problem.irls_tol * (1 + abs(state.Dp))
        accepted = false
        for _ in 0:_PLS_MAX_HALVINGS
            beta = state.beta + step
            trial = _pirls_state(problem, X * beta, beta, Slambda)
            # Near the optimum Dp changes at rounding level only.
            if trial.valid && trial.Dp <= state.Dp + 8 * eps(T) * (1 + abs(state.Dp))
                state, accepted = trial, true
                break
            end
            step ./= 2
        end
        converged && return _pirls_boundary(fam, state) ? nothing : (; state, iterations)
        if !accepted
            # No halved step decreases Dp: accept a fit within working precision.
            decrement <= sqrt(eps(T)) * (1 + abs(state.Dp)) && !_pirls_boundary(fam, state) &&
                return (; state, iterations)
            return nothing
        end
    end
end

# Laplace-approximate REML, GCV or UBRE at lambda for an exponential-family
# response (mgcv 1.9-3 gam.fit3 and gdi), with exact derivatives on log(lambda)
# (Wood 2011). REML uses the Newton Hessian H = XᵀWₙX + S_λ, Wₙ = D''/2; τ, the
# EDF, Vp and the scale use Fisher weights W_F (equal for canonical links), as
# mgcv does. β derivatives are implicit: dβ/dρⱼ = -H⁻¹λⱼSⱼβ.
function _pls_evaluate(problem::_PIRLSProblem{T}, lambda::Vector{T}, criterion::Symbol;
        derivatives::Bool=true, solution::Bool=false, undefined_ok::Bool=false) where {T}
    p, M, n = problem.p, length(lambda), problem.nobs
    terms = _pls_penalty_terms(problem, lambda, derivatives)
    terms === nothing && return nothing
    (; basis, rank, A, Slambda, logdet_penalty, gradient_penalty, hessian_penalty) = terms
    null_dimension = p - rank
    X = problem.X * basis
    fitted = _pirls_fit(problem, X, Slambda, transpose(basis) * problem.start; warm=problem.started[])
    fitted === nothing && return nothing
    state = fitted.state
    beta = state.beta
    wn, wf = _pirls_values(state, Val(:wn)), _pirls_values(state, Val(:wf))
    FN = _scaled_cholesky(_pirls_gram(X, wn) + Slambda)
    FN === nothing && return nothing
    canonical = _glm_canonical(problem.family)
    GF = _pirls_gram(X, wf)
    FF = canonical ? FN : _scaled_cholesky(GF + Slambda)
    FF === nothing && return nothing
    problem.start .= basis * beta
    problem.started[] = true
    dev, pen = state.dev, dot(beta, Slambda * beta)
    Dp = dev + pen
    HFinv = _scaled_inverse(FF)
    influence = HFinv * GF
    tau = tr(influence)
    known = _glm_scale_known(problem.family)
    gradient, hessian = zeros(T, M), zeros(T, M, M)
    db, hess_Dp, grad_Dp, hess_logdet = zeros(T, p, M), zeros(T, M, M), zeros(T, M), zeros(T, M, M)
    if derivatives && M > 0
        HNinv = canonical ? HFinv : _scaled_inverse(FN)
        wn1, wn2 = _pirls_values(state, Val(:wn1)), _pirls_values(state, Val(:wn2))
        Abeta = [A[j] * beta for j in 1:M]
        dbeta = [-(HNinv * Abeta[j]) for j in 1:M]
        deta = [X * dbeta[j] for j in 1:M]
        dHN = [A[j] + _pirls_gram(X, wn1 .* deta[j]) for j in 1:M]
        d2eta = [X * (-(HNinv * (dHN[k] * dbeta[j] + A[j] * dbeta[k] + (j == k ? Abeta[j] : zeros(T, p)))))
            for j in 1:M, k in 1:M]
        for j in 1:M
            db[:, j] = basis * dbeta[j]
            grad_Dp[j] = dot(beta, Abeta[j])
            for k in 1:M
                hess_Dp[j, k] = (j == k ? grad_Dp[j] : zero(T)) + 2 * dot(Abeta[j], dbeta[k])
            end
        end
        if criterion === :REML
            leverage = vec(sum((X * HNinv) .* X; dims=2))
            BN = [HNinv * dHN[j] for j in 1:M]
            grad_logdet = T[tr(BN[j]) for j in 1:M]
            for j in 1:M, k in 1:M
                v = wn2 .* deta[j] .* deta[k] .+ wn1 .* d2eta[j, k]
                hess_logdet[j, k] = (j == k ? tr(HNinv * A[j]) : zero(T)) + dot(v, leverage) -
                    _trace_product(BN[j], BN[k])
            end
            hess_logdet .= (hess_logdet .+ transpose(hess_logdet)) ./ 2
        else
            d1 = _pirls_values(state, Val(:d1))
            wf1, wf2 = _pirls_values(state, Val(:wf1)), _pirls_values(state, Val(:wf2))
            grad_dev = T[dot(d1, deta[j]) for j in 1:M]
            hess_dev = T[2 * dot(wn .* deta[j], deta[k]) + dot(d1, d2eta[j, k]) for j in 1:M, k in 1:M]
            P = HFinv * Slambda
            Q = P * HFinv
            leverageQ = vec(sum((X * Q) .* X; dims=2))
            C = [HFinv * (A[j] + _pirls_gram(X, wf1 .* deta[j])) for j in 1:M]
            E = [HFinv * A[j] for j in 1:M]
            grad_tau = T[_trace_product(C[j], P) - tr(E[j]) for j in 1:M]
            hess_tau = zeros(T, M, M)
            for j in 1:M, k in 1:M
                v = wf2 .* deta[j] .* deta[k] .+ wf1 .* d2eta[j, k]
                hess_tau[j, k] = -_trace_product(C[k] * C[j], P) - _trace_product(C[j] * C[k], P) +
                    dot(v, leverageQ) + (j == k ? _trace_product(E[j], P) - tr(E[j]) : zero(T)) +
                    _trace_product(C[j], E[k]) + _trace_product(C[k], E[j])
            end
        end
    else
        grad_logdet = zeros(T, M)
        grad_dev, hess_dev, grad_tau, hess_tau = zeros(T, M), zeros(T, M, M), zeros(T, M), zeros(T, M, M)
    end
    residual_df = n - tau
    nu = n - null_dimension
    defined = criterion === :REML ? (known || (nu > 0 && Dp > 0)) :
        criterion === :GCV ? residual_df > sqrt(eps(T)) * n : true
    value = T(NaN)
    if !defined
        undefined_ok || return nothing
        gradient, hessian = fill(T(NaN), M), fill(T(NaN), M, M)
    elseif criterion === :REML
        logdet_hessian = FN.logdet
        if known
            value = Dp / 2 - problem.saturated + logdet_hessian / 2 - logdet_penalty / 2 -
                null_dimension / 2 * log(2 * T(pi))
            gradient = grad_Dp ./ 2 .+ grad_logdet ./ 2 .- gradient_penalty ./ 2
            hessian = hess_Dp ./ 2 .+ hess_logdet ./ 2 .- hessian_penalty ./ 2
        else
            value = nu / 2 * (1 + log(2 * T(pi) * Dp / nu)) + logdet_hessian / 2 -
                logdet_penalty / 2 - problem.saturated / 2
            gradient = nu / 2 .* grad_Dp ./ Dp .+ grad_logdet ./ 2 .- gradient_penalty ./ 2
            hessian = nu / 2 .* (hess_Dp ./ Dp .- grad_Dp .* transpose(grad_Dp) ./ Dp^2) .+
                hess_logdet ./ 2 .- hessian_penalty ./ 2
        end
    elseif criterion === :GCV
        delta = residual_df
        value = n * dev / delta^2
        gradient = n .* grad_dev ./ delta^2 .+ 2n * dev .* grad_tau ./ delta^3
        hessian = n .* hess_dev ./ delta^2 .+
            2n .* (grad_dev .* transpose(grad_tau) .+ grad_tau .* transpose(grad_dev)) ./ delta^3 .+
            2n * dev .* hess_tau ./ delta^3 .+ 6n * dev .* grad_tau .* transpose(grad_tau) ./ delta^4
    else
        # UBRE (Mallows' Cp) with known scale one.
        value = dev / n + 2tau / n - 1
        gradient = grad_dev ./ n .+ 2 .* grad_tau ./ n
        hessian = hess_dev ./ n .+ 2 .* hess_tau ./ n
    end
    # Weights at rounding level (fitted probabilities numerically 0 or 1) can
    # leave the derivatives non-finite; such smoothing parameters are
    # inadmissible during selection.
    if defined && !(isfinite(value) && all(isfinite, gradient) && all(isfinite, hessian))
        undefined_ok || return nothing
        defined = false
        value, gradient, hessian = T(NaN), fill(T(NaN), M), fill(T(NaN), M, M)
    end
    (; value, defined, gradient, hessian=Matrix(Symmetric((hessian .+ transpose(hessian)) ./ 2)),
        rss=dev, pen, tau, null_dimension, basis, coefficients=[basis * beta], inverses=[HFinv],
        influences=[influence], eta=problem.X * (basis * beta), state, iterations=fitted.iterations,
        Dp, db, grad_Dp, hess_Dp, hess_logdet, hessian_penalty)
end

# Column sums of squares of the weighted design W^{1/2}X.
function _pls_design_scale(problem::_PLSProblem{T}) where {T}
    xx = zeros(T, problem.p)
    for block in problem.blocks
        xx .+= vec(sum(abs2, block.R; dims=1))
    end
    xx
end

# mgcv 1.9-3 initial.spg: Fisher weights at the family's starting means.
function _pls_design_scale(problem::_PIRLSProblem{T}) where {T}
    fam = problem.family
    wf = map(problem.y, problem.weights) do y, w
        m, mc, h1 = _glm_inverse_link(fam.link, _glm_link(fam.link, _glm_mustart(fam, y, w)))
        w * h1^2 / first(_glm_variance(fam, m, mc))
    end
    vec(sum(wf .* abs2.(problem.X); dims=1))
end

# mgcv 1.9-3 initial.sp (expensive=FALSE), on the weighted design.
function _pls_initial_lambda(problem)
    xx = _pls_design_scale(problem)
    T = eltype(xx)
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
    criterion === :GCV ? abs(value) : 1 + abs(value)
end

_pls_failure(::_PLSProblem, where) =
    "penalized fit is singular at the $where smoothing parameters; check that the data identify the penalty null space"
_pls_failure(::_PIRLSProblem, where) =
    "penalized IRLS failed at the $where smoothing parameters: the penalized Hessian is singular or not positive definite, or the iterations did not converge; check that the data identify the penalty null space (for binomial data, that the classes are not separated)"

function _pls_optimize(problem::_AnyPLSProblem{T}, criterion, rho0, lo, hi, free; tol, maxiter) where {T}
    rho = copy(rho0)
    current = _pls_evaluate(problem, exp.(rho), criterion)
    current === nothing && throw(ArgumentError(_pls_failure(problem, "initial")))
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

function _pls_criterion(criterion, fam::_GLMFamily=_GLMFamily(:gaussian, :identity))
    criterion in (:REML, :GCV, :UBRE) || throw(ArgumentError("criterion must be :REML, :GCV or :UBRE"))
    known = _glm_scale_known(fam)
    criterion === :GCV && known && throw(ArgumentError(
        "family=:$(fam.family) has known scale; use criterion=:UBRE (mgcv GCV.Cp chooses UBRE) or :REML"))
    criterion === :UBRE && !known && throw(ArgumentError(
        "criterion=:UBRE needs a known scale; use :GCV (mgcv GCV.Cp) or :REML for the Gaussian family"))
    criterion
end

function _pls_rank_tol(rank_tol, ::Type{T}) where {T}
    value = rank_tol === nothing ? eps(T)^(3/4) : rank_tol
    value isa Real && isfinite(value) && 0 < value < 1 ||
        throw(ArgumentError("rank_tol must be in (0, 1)"))
    T(value)
end

# Group ranks of the active penalties, independent of lambda's size.
function _pls_reference_rank(problem::_AnyPLSProblem{T}, active) where {T}
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

function _pls_exact_fit(problem::_PLSProblem{T}, final) where {T}
    total = problem.rss0 + sum(block -> sum(abs2, block.f), problem.blocks)
    final.rss + final.pen <= eps(T)^(3/2) * total
end
# A Gaussian family with a non-identity link has an estimated scale too.
_pls_exact_fit(problem::_PIRLSProblem{T}, final) where {T} = !_glm_scale_known(problem.family) &&
    final.rss + final.pen <= eps(T)^(3/2) * sum(abs2, problem.y .* sqrt.(problem.weights))

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
function _pls_select(problem::_AnyPLSProblem{T}, criterion, lo, hi; tol, maxiter) where {T}
    M = length(problem.penalties)
    rank = _pls_reference_rank(problem, hi .> 0)
    initial = clamp.(_pls_initial_lambda(problem), lo, hi)
    fixed = lo .== hi
    free = findall(!, fixed)
    rho0 = log.(initial)
    span = T(_PLS_LOG_RANGE)
    log_lo = T[fixed[j] ? rho0[j] : max(log(lo[j]), rho0[j] - span) for j in 1:M]
    log_hi = T[fixed[j] ? rho0[j] : min(log(hi[j]), rho0[j] + span) for j in 1:M]
    nblocks = problem isa _PLSProblem ? length(problem.blocks) : 1
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
    final === nothing && throw(ArgumentError(_pls_failure(problem, "selected")))
    check_rank(final)
    # Residuals at rounding level leave REML and GCV without a noise scale.
    isempty(free) || !_pls_exact_fit(problem, final) || throw(ArgumentError(
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
        penalties, weights=nothing, family::Symbol=:gaussian, link=nothing, criterion::Symbol=:REML,
        lambda=nothing, lower=nothing, upper=nothing, rank_tol=nothing, tol::Real=1e-8,
        maxiter::Integer=200, irls_tol::Real=1e-12, irls_maxiter::Integer=200)
    first(_pls_fit(X, y; penalties, weights, family, link, criterion, lambda, lower, upper,
        rank_tol, tol, maxiter, irls_tol, irls_maxiter))
end

# Return the fit and the solver's final evaluation (derivatives included).
function _pls_fit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; penalties, weights,
        family, link, criterion, lambda, lower, upper, rank_tol, tol, maxiter, irls_tol, irls_maxiter)
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
    fam = _glm_family(family, link)
    criterion = _pls_criterion(criterion, fam)
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    isfinite(irls_tol) && irls_tol > 0 || throw(ArgumentError("irls_tol must be finite and positive"))
    irls_maxiter >= 1 || throw(ArgumentError("irls_maxiter must be positive"))
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
    _glm_check_response(fam, yc, w)
    rank_value = _pls_rank_tol(rank_tol, T)
    if _glm_identity(fam)
        root = sqrt.(w)
        block, rss0 = _pls_block(root .* Xc, root .* yc)
        problem = _pls_problem([block], n, rss0, sum(log, w), p, matrices, columns, rank_value)
    else
        groups, unpenalized = _penalty_groups(matrices, columns, p)
        problem = _PIRLSProblem{T}(Xc, yc, w, fam, n, p, matrices, columns, groups, unpenalized,
            rank_value, _glm_saturated(fam, yc, w), zeros(T, p), Ref(false), T(irls_tol), Int(irls_maxiter))
    end
    _pls_finish(problem, criterion, lo, hi, tol, maxiter, Xc, yc, w, fam, matrices, columns,
        weights === nothing ? :equal : :supplied, irls_tol, irls_maxiter)
end

# Select the smoothing parameters and assemble the result (specialized on the
# problem type so that the result type is inferred).
function _pls_finish(problem::_AnyPLSProblem{T}, criterion, lo, hi, tol, maxiter, Xc, yc, w, fam,
        matrices, columns, weight_source, irls_tol, irls_maxiter) where {T}
    n = problem.nobs
    selected, status, final, settings, _ = _pls_select(problem, criterion, lo, hi; tol, maxiter)
    beta = only(final.coefficients)
    basis = final.basis
    inverse = basis * only(final.inverses) * transpose(basis)
    influence = basis * only(final.influences) * transpose(basis)
    edf = diag(influence)
    edf_total = sum(edf)
    known = _glm_scale_known(fam)
    if problem isa _PLSProblem
        eta = Xc * beta
        fitted_values = eta
        deviance = final.rss
        iterations::Int = 0
    else
        eta = final.eta
        fitted_values = _glm_response_values(fam, eta)
        deviance = final.rss
        iterations = final.iterations
    end
    # mgcv's scale: one when known, else the Pearson statistic over n - τ
    # (gam.control scale.est="fletcher", which is Pearson for the Gaussian
    # family); no residual degrees of freedom leave it undefined (NaN).
    pearson = problem isa _PLSProblem ? final.rss : sum(o -> o.pearson, final.state.obs)
    sigma2 = known ? one(T) : n - final.tau > sqrt(eps(T)) * n ? pearson / (n - final.tau) : T(NaN)
    reml_scale = known ? one(T) :
        n > final.null_dimension ? (final.rss + final.pen) / (n - final.null_dimension) : T(NaN)
    Vp = Matrix(Symmetric(inverse .* sigma2))
    Ve = Matrix(Symmetric((influence * inverse) .* sigma2))
    settings = merge((; weights=weight_source), settings,
        (; scale=known ? :known : :estimated, irls_tol=T(irls_tol), irls_maxiter=Int(irls_maxiter),
            irls_iterations=iterations))
    fit = PenalizedLeastSquares{T,typeof(settings)}(beta, selected, status, criterion, final.value,
        edf, edf_total, T[sum(edf[c]) for c in columns], sigma2, reml_scale, Vp, Ve,
        fitted_values, yc - fitted_values, w, eta, deviance, fam.family, fam.link, n,
        final.null_dimension, matrices, columns, settings)
    fit, final
end

coef(f::PenalizedLeastSquares) = copy(f.coef)
fitted(f::PenalizedLeastSquares) = copy(f.fitted)
residuals(f::PenalizedLeastSquares) = copy(f.residuals)
nobs(f::PenalizedLeastSquares) = f.nobs
dof(f::PenalizedLeastSquares) = f.edf_total + (f.settings.scale === :estimated)
dof_residual(f::PenalizedLeastSquares) = f.nobs - f.edf_total

function vcov(f::PenalizedLeastSquares; type::Symbol=:bayesian)
    type === :bayesian && return copy(f.vcov_bayesian)
    type === :frequentist && return copy(f.vcov_frequentist)
    throw(ArgumentError("vcov type must be :bayesian or :frequentist"))
end

function predict(f::PenalizedLeastSquares, X::AbstractMatrix{<:Real}; type::Symbol=:link)
    size(X, 2) == length(f.coef) || throw(DimensionMismatch(
        "X has $(size(X, 2)) columns but the fit has $(length(f.coef)) coefficients"))
    all(isfinite, X) || throw(ArgumentError("X must be finite"))
    type in (:link, :response) || throw(ArgumentError("type must be :link or :response"))
    eta = X * f.coef
    type === :link ? eta : _glm_response_values(_GLMFamily(f.family, f.link), eta)
end

function Base.show(io::IO, f::PenalizedLeastSquares)
    print(io, "PenalizedLeastSquares(n=", f.nobs, ", p=", length(f.coef), ", family=", f.family,
        ", link=", f.link, ", criterion=",
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
