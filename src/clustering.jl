# Functional clustering: L² k-means on curves or FPC scores (Clustering.jl),
# k-centres functional clustering (Chiou and Li, 2007) and Gaussian mixtures on
# FPC scores (GaussianMixtures.jl EM, EMCluster initialization and stopping).

"""
    FunctionalKMeans
    fit(FunctionalKMeans, fd::Union{GridFunctionalData,BasisFunctionalData}; k=2,
        init=:kmpp, nstart=1, maxiter=100, tol=1e-6, rng=Random.default_rng())
    fit(FunctionalKMeans, f::FPCA; k=2, kwargs...)

k-means clustering (Lloyd's algorithm, from Clustering.jl `kmeans`) of curves
under the L² distance or of the FPC scores of a fitted [`FPCA`](@ref).

Curves: complete grid data use the stored quadrature weights `w`, so the
squared distance between curves `x` and `y` is `sum(w .* (x .- y).^2)`;
basis data use the exact basis Gram matrix. Masked grid and irregular data
throw: interpolate them to a complete grid, or cluster FPC scores. Scores:
the Euclidean distance between score vectors, which equals the L² distance
between the reconstructed curves because the eigenfunctions are orthonormal
under the work-grid weights. Centres are cluster means.

Defaults and their sources:

- `k=2`: fda.usc 2.2.0 `kmeans.fd` (`ncl=2`).
- `init=:kmpp`: k-means++ seeding (Arthur and Vassilvitskii, 2007, expected
  cost within `O(log k)` of the optimum), Clustering.jl's default. This departs
  from fda.usc's `method="sample"`, available as `init=:spread`: among
  `min(100, binomial(n, k))` random sets of `k` curves, take the first with the
  largest sum of pairwise L² distances. `init=:rand` draws `k` distinct curves
  uniformly, as R's `kmeans` does for a number of centres. An integer vector
  gives the seed curves directly (fda.usc's `ncl` vector) and requires
  `nstart=1`.
- `nstart=1`: R `kmeans` and fda.usc use one start. Several starts keep the
  smallest total cost (the first on ties); all start costs are stored.
- `maxiter=100`: fda.usc `max.iter`.
- `tol=1e-6` is relative: Clustering.jl stops when the absolute change of the
  total cost is below `tol` times the total sum of squares about the overall
  mean, making the rule invariant to the units of the data (Clustering.jl's
  own `tol` is absolute). The resolved absolute value is stored.

fda.usc 2.2.0 `kmeans.fd` computes its distance matrix once, to the initial
centres, and never updates it, so its result is the nearest-initial-centre
partition followed by one update of depth-trimmed means (`func.trim.FM`,
`trim=0.05`). The package iterates Lloyd's algorithm to convergence with
plain means; see the clustering documentation for the audit.

Clustering.jl re-seeds an empty cluster at random during the iteration; a
final empty cluster throws. All settings are stored and shown.

The result is a `Clustering.ClusteringResult`, so `assignments`, `nclusters`,
`Clustering.counts` and `randindex` apply. Fields:

- `space`: `:L2` (curves) or `:fpc_scores`.
- `k`, `assignments`, `counts`.
- `centers`: cluster means as functional data. Grid data give a
  `GridFunctionalData` on the data grid, basis data a `BasisFunctionalData` in
  the data basis and FPC scores the FPCA reconstruction of the mean scores.
- `center_coordinates`: `k × d` cluster means of grid values, basis
  coefficients or scores.
- `costs`: squared L² (or score) distance of each curve to its own centre;
  `withinss` sums them by cluster and `totalcost` over all curves.
- `iterations`, `converged`: from the chosen start's Lloyd iteration.
- `seeds`: the chosen start's initial centre indices; `start_costs` holds
  every start's total cost and `best_start` the chosen one (the first minimum).
- `fpca`: the FPCA whose scores were clustered, or `nothing`.
- `settings`: every option, including the resolved numerical tolerance.
"""
struct FunctionalKMeans{T<:AbstractFloat,C<:AbstractFunctionalData,F} <: Clustering.ClusteringResult
    space::Symbol
    k::Int
    assignments::Vector{Int}
    counts::Vector{Int}
    centers::C
    center_coordinates::Matrix{T}
    costs::Vector{T}
    withinss::Vector{T}
    totalcost::T
    iterations::Int
    converged::Bool
    seeds::Vector{Int}
    start_costs::Vector{T}
    best_start::Int
    fpca::F
    ids::Union{Nothing,Vector{Any}}
    settings::NamedTuple
end

Base.@constprop :aggressive function fit(::Type{FunctionalKMeans}, fd::Union{GridFunctionalData,BasisFunctionalData};
        k::Integer=2, init=:kmpp, nstart::Integer=1, maxiter::Integer=100, tol::Real=1e-6,
        rng::AbstractRNG=Random.default_rng())
    points, metric = _kmeans_points(fd)
    run = _kmeans_run(points, k; init, nstart, maxiter, tol, rng)
    coordinates = _cluster_means(_kmeans_coordinates(fd), run.assignments, Int(k))
    centers = _kmeans_centers(fd, coordinates)
    settings = (; init=_init_setting(init), nstart=Int(nstart), maxiter=Int(maxiter), tol=float(tol),
        absolute_tol=run.absolute_tol, distance=metric, algorithm=:lloyd,
        implementation=:Clustering_kmeans, rng=typeof(rng))
    _kmeans_result(:L2, run, points, coordinates, centers, nothing, fd.ids, settings)
end

Base.@constprop :aggressive function fit(::Type{FunctionalKMeans}, f::FPCA; k::Integer=2, init=:kmpp,
        nstart::Integer=1, maxiter::Integer=100, tol::Real=1e-6, rng::AbstractRNG=Random.default_rng())
    points = Matrix(transpose(f.scores))
    run = _kmeans_run(points, k; init, nstart, maxiter, tol, rng)
    coordinates = _cluster_means(f.scores, run.assignments, Int(k))
    centers = reconstruct(f, coordinates)
    settings = (; init=_init_setting(init), nstart=Int(nstart), maxiter=Int(maxiter), tol=float(tol),
        absolute_tol=run.absolute_tol, distance=:euclidean_scores, algorithm=:lloyd,
        implementation=:Clustering_kmeans, rng=typeof(rng))
    _kmeans_result(:fpc_scores, run, points, coordinates, centers, f, f.ids, settings)
end

fit(::Type{FunctionalKMeans}, fd::AbstractFunctionalData; kwargs...) = throw(ArgumentError(
    "k-means on curves requires complete GridFunctionalData or BasisFunctionalData; " *
    "interpolate with to_grid, or cluster FPC scores with fit(FunctionalKMeans, fit(FPCA, fd))"))

_init_setting(init::Symbol) = init
_init_setting(init::AbstractVector{<:Integer}) = collect(Int, init)

function _kmeans_points(fd::GridFunctionalData)
    fd.observed === nothing || throw(ArgumentError(
        "k-means on curves requires complete grid data; interpolate with to_grid or cluster FPC scores"))
    root = sqrt.(fd.weights)
    Matrix(transpose(fd.values .* transpose(root))), :L2_stored_quadrature
end

function _kmeans_points(fd::BasisFunctionalData)
    factor = cholesky(Symmetric(gram(fd.basis))).U
    factor * transpose(fd.coefs), :L2_basis_gram
end

_kmeans_coordinates(fd::GridFunctionalData) = fd.values
_kmeans_coordinates(fd::BasisFunctionalData) = fd.coefs
_kmeans_centers(fd::GridFunctionalData, coordinates) = GridFunctionalData(coordinates, fd.grid)
_kmeans_centers(fd::BasisFunctionalData, coordinates) = BasisFunctionalData(coordinates, fd.basis)

function _cluster_means(x::AbstractMatrix{T}, labels, k) where {T}
    out = zeros(T, k, size(x, 2))
    for c in 1:k
        members = findall(==(c), labels)
        out[c, :] = sum(view(x, members, :); dims=1) ./ length(members)
    end
    out
end

function _cluster_counts(labels, k)
    out = zeros(Int, k)
    for c in labels
        out[c] += 1
    end
    out
end

function _validate_cluster_count(k, n)
    _integer_in(k, 1, n) || throw(ArgumentError("k must be an integer between 1 and the number of curves ($n)"))
    Int(k)
end

function _kmeans_run(points::Matrix{T}, k; init, nstart, maxiter, tol, rng) where {T}
    d, n = size(points)
    k = _validate_cluster_count(k, n)
    _integer_in(nstart, 1) || throw(ArgumentError("nstart must be a positive integer"))
    _integer_in(maxiter, 1) || throw(ArgumentError("maxiter must be a positive integer"))
    tol isa Real && isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    all(isfinite, points) || throw(ArgumentError("clustering coordinates must be finite"))
    total = sum(abs2, points .- sum(points; dims=2) ./ n)
    total > 0 || throw(ArgumentError("all curves are identical; there is nothing to cluster"))
    absolute_tol = Float64(tol * total)
    if init isa AbstractVector{<:Integer}
        nstart == 1 || throw(ArgumentError("supplied seed indices require nstart=1"))
        length(init) == k && allunique(init) && all(i -> 1 <= i <= n, init) || throw(ArgumentError(
            "init seeds must be $k distinct curve indices between 1 and $n"))
    elseif !(init in (:kmpp, :rand, :spread))
        throw(ArgumentError("init must be :kmpp, :rand, :spread or a vector of seed indices"))
    end
    best = nothing
    start_costs = T[]
    best_seeds = Int[]
    best_start = 0
    for start in 1:nstart
        seeds = _kmeans_seeds(points, k, init, rng)
        result = Clustering.kmeans(points, k; init=seeds, maxiter=Int(maxiter), tol=absolute_tol,
            display=:none, rng)
        any(iszero, Clustering.counts(result)) && throw(ArgumentError(
            "k-means ended with an empty cluster; reduce k or use another initialization"))
        cost = T(result.totalcost)
        push!(start_costs, cost)
        if best === nothing || cost < last(best).totalcost
            best = (start, result)
            best_seeds = seeds
            best_start = start
        end
    end
    result = last(best)
    (; k, assignments=copy(Clustering.assignments(result)), iterations=result.iterations,
        converged=result.converged, seeds=best_seeds, start_costs, best_start, absolute_tol)
end

_kmeans_seeds(points, k, init::AbstractVector{<:Integer}, rng) = collect(Int, init)
_kmeans_seeds(points, k, init::Symbol, rng) = init === :spread ?
    _spread_seeds(points, k, rng) : Clustering.initseeds(init, points, k; rng)

# fda.usc kmeans.center.ini(method = "sample"): the first of min(100, C(n,k))
# random k-subsets with the largest sum of pairwise distances.
function _spread_seeds(points, k, rng)
    n = size(points, 2)
    draws = Int(min(100, binomial(big(n), k)))
    best, best_total = Int[], -Inf
    for _ in 1:draws
        candidate = randperm(rng, n)[1:k]
        total = 0.0
        for a in 1:k, b in a+1:k
            total += sqrt(sum(abs2, view(points, :, candidate[a]) .- view(points, :, candidate[b])))
        end
        if total > best_total
            best, best_total = candidate, total
        end
    end
    best
end

function _kmeans_result(space, run, points::Matrix{T}, coordinates, centers, fpca, ids, settings) where {T}
    k = run.k
    labels = run.assignments
    centroid = _cluster_means(Matrix(transpose(points)), labels, k)
    costs = T[sum(abs2, view(points, :, i) .- view(centroid, labels[i], :)) for i in eachindex(labels)]
    withinss = zeros(T, k)
    for i in eachindex(labels)
        withinss[labels[i]] += costs[i]
    end
    FunctionalKMeans(space, k, labels, _cluster_counts(labels, k), centers, coordinates, costs,
        withinss, sum(costs), run.iterations, run.converged, run.seeds, run.start_costs,
        run.best_start, fpca, ids === nothing ? nothing : deepcopy(ids), settings)
end

nobs(m::FunctionalKMeans) = length(m.assignments)

"""
    predict(m::FunctionalKMeans, newdata)

Assign new curves to the nearest centre. Curve clusterings require data on
the training grid (complete) or in the training basis; score clusterings
project `newdata` with [`project_scores`](@ref) of the stored FPCA, so any
data type that FPCA accepts can be used. Ties go to the lowest cluster index.
"""
function predict(m::FunctionalKMeans, newdata::AbstractFunctionalData)
    if m.space === :fpc_scores
        return _nearest_centre(project_scores(m.fpca, newdata), m.center_coordinates, I)
    end
    _kmeans_predict(m, m.centers, newdata)
end

function _kmeans_predict(m, centers::GridFunctionalData, newdata::GridFunctionalData)
    length(newdata.grid) == length(centers.grid) ||
        throw(DimensionMismatch("prediction grid length must match the training grid"))
    newdata.grid == centers.grid || throw(ArgumentError("prediction grid must equal the training grid"))
    newdata.observed === nothing || throw(ArgumentError("prediction curves must be completely observed"))
    _nearest_centre(newdata.values, m.center_coordinates, Diagonal(centers.weights))
end

function _kmeans_predict(m, centers::BasisFunctionalData, newdata::BasisFunctionalData)
    newdata.basis == centers.basis || throw(ArgumentError("prediction data must use the training basis"))
    _nearest_centre(newdata.coefs, m.center_coordinates, gram(centers.basis))
end

_kmeans_predict(m, centers, newdata) = throw(ArgumentError(
    "prediction data must have the training representation ($(nameof(typeof(centers))))"))

function _nearest_centre(x, centers, metric)
    [argmin([dot(x[i, :] .- centers[c, :], metric * (x[i, :] .- centers[c, :]))
        for c in axes(centers, 1)]) for i in axes(x, 1)]
end

function Base.show(io::IO, m::FunctionalKMeans)
    print(io, "FunctionalKMeans(space=", m.space, ", n=", nobs(m), ", k=", m.k,
        ", counts=", m.counts, ", totalcost=", m.totalcost, ", iterations=", m.iterations,
        ", converged=", m.converged, ", best_start=", m.best_start, " of ", length(m.start_costs),
        ", settings=", m.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", m::FunctionalKMeans) = show(io, m)

"""
    KCFC
    fit(KCFC, fd::Union{GridFunctionalData,IrregularFunctionalData}; k=3,
        initial=nothing, fpca_options=(; ncomponents=FVE(0.9)),
        cluster_fpca_options=(; ncomponents=FVE(0.7)), maxiter=125,
        kmeans_options=(;), rng=Random.default_rng())

k-centres functional clustering (Chiou and Li, 2007), following fdapace 0.6.0
`kCFC` (also `FClust(cmethod="kCFC")`):

1. Fit [`FPCA`](@ref) to all curves with `fpca_options`.
2. Initial partition: `initial` labels in `1:k`, or k-means of the FPC scores
   with [`FunctionalKMeans`](@ref) (`maxiter=maxiter` and `rng` unless
   `kmeans_options` sets them). A cluster of at most three curves throws.
3. Fit an FPCA with `cluster_fpca_options` to each cluster.
4. Reclassify every curve to the cluster whose FPCA reconstructs it with the
   smallest integrated squared error (ISE, trapezoidal rule over the curve's
   observation times; ties go to the lowest index). Stop when the partition
   repeats any earlier reclassification (`:converged`), loses a cluster
   (`:lost_cluster`), has a cluster of at most `max(0.01n, 3)` curves
   (`:tiny_cluster`), or after `maxiter` steps (`:maxiter`). Otherwise refit
   the cluster FPCAs and repeat. The last three statuses warn, as in fdapace.

Reconstruction uses each cluster FPCA's own score solver through
[`project_scores`](@ref). For dense FPCA fits this is the trapezoidal
projection fdapace's `GetINScores` uses, so results agree with fdapace given
the same FPCAs. For irregular or masked data it is the PACE conditional
expectation, which extends kCFC to sparse designs; fdapace 0.6.0 stops for
sparse data. When the initial FPCA uses PACE and `cluster_fpca_options` has no
`grid`, cluster FPCAs use the initial FPCA's work grid, so every curve lies
inside every cluster's grid. Every curve needs at least two observations.

Defaults and their sources: `k=3`, `maxiter=125`, FVE thresholds 0.9 (initial)
and 0.7 (clusters) are fdapace 0.6.0 `kCFC`'s. Departures: the FPCAs use the
package's [`fit(FPCA, ...)`](@ref FPCA) defaults (dense cross-sectional
estimates on complete grids, PACE otherwise), whereas fdapace's default
options smooth the mean and covariance with GCV bandwidths; pass
`fpca_options`/`cluster_fpca_options` (for example `method=:pace`) to smooth.
fdapace's initial clustering is R `kmeans(algorithm="MacQueen")` after
`set.seed(123)`, which cannot be reproduced outside R; the package uses
Clustering.jl's Lloyd iteration with k-means++ seeding and records the
initial labels, so an R partition can be supplied as `initial`.

The result is a `Clustering.ClusteringResult`. Fields:

- `k`, `assignments`, `counts`: final partition.
- `initial_assignments`, `initial_fpca` (FPCA of all curves) and
  `initial_clustering` (the [`FunctionalKMeans`](@ref) of its scores, or
  `nothing` when initial labels were supplied).
- `fpcas`: the cluster FPCAs used in the final reclassification, in cluster
  order. As in fdapace, these were fitted to the previous partition.
- `centers`: their mean functions as `GridFunctionalData` on the common grid.
- `costs`: `n × k` integrated squared reconstruction errors of the final step.
- `history`: the partition produced by every reclassification step.
- `iterations` (fdapace `iterToConv`) and `status`: `:converged`, `:maxiter`,
  `:lost_cluster` or `:tiny_cluster`.
- `settings`: every option.
"""
struct KCFC{T<:AbstractFloat,K} <: Clustering.ClusteringResult
    k::Int
    assignments::Vector{Int}
    counts::Vector{Int}
    initial_assignments::Vector{Int}
    initial_fpca::FPCA{T}
    initial_clustering::K
    fpcas::Vector{FPCA{T}}
    centers::GridFunctionalData{T}
    costs::Matrix{T}
    history::Vector{Vector{Int}}
    iterations::Int
    status::Symbol
    ids::Union{Nothing,Vector{Any}}
    settings::NamedTuple
end

Base.@constprop :aggressive function fit(::Type{KCFC}, fd::Union{GridFunctionalData,IrregularFunctionalData};
        k::Integer=3, initial=nothing, fpca_options::NamedTuple=(;),
        cluster_fpca_options::NamedTuple=(;), maxiter::Integer=125, kmeans_options::NamedTuple=(;),
        rng::AbstractRNG=Random.default_rng())
    n = nobs(fd)
    _integer_in(k, 2, n) || throw(ArgumentError("k must be an integer between 2 and the number of curves ($n)"))
    k = Int(k)
    _integer_in(maxiter, 1) || throw(ArgumentError("maxiter must be a positive integer"))
    minimum(nobs_per_curve(fd)) >= 2 || throw(ArgumentError(
        "kCFC needs at least two observations per curve to integrate squared reconstruction errors"))
    initial_options = merge((; ncomponents=FVE(0.9)), fpca_options)
    initial_fpca = fit(FPCA, fd; initial_options...)
    labels0, initial_clustering = _kcfc_initial(initial, initial_fpca, k, maxiter, kmeans_options, rng)
    minimum(_cluster_counts(labels0, k)) > 3 || throw(ArgumentError(
        "kCFC initial clustering has a cluster of three or fewer curves; use a smaller k, " *
        "another seed or other initial labels"))
    cluster_options = merge((; ncomponents=FVE(0.7)), cluster_fpca_options)
    if initial_fpca.method === :pace && !haskey(cluster_options, :grid)
        cluster_options = merge(cluster_options, (; grid=initial_fpca.grid))
    end
    fpcas = _kcfc_fits(fd, labels0, k, cluster_options)
    history = Vector{Int}[]
    status = :maxiter
    T = eltype(initial_fpca.scores)
    costs = Matrix{T}(undef, n, k)
    labels = labels0
    step = 0
    for j in 1:maxiter
        step = j
        for c in 1:k
            costs[:, c] = _kcfc_costs(fpcas[c], fd)
        end
        labels = [argmin(view(costs, i, :)) for i in 1:n]
        push!(history, labels)
        sizes = _cluster_counts(labels, k)
        if any(iszero, sizes)
            status = :lost_cluster
            break
        elseif minimum(sizes) <= max(0.01n, 3)
            status = :tiny_cluster
            break
        elseif j > 1 && any(==(labels), @view history[1:j-1])
            status = :converged
            break
        end
        fpcas = _kcfc_fits(fd, labels, k, cluster_options)
    end
    status === :maxiter && @warn "kCFC did not converge after maxiter=$maxiter reclassification steps"
    status === :tiny_cluster && @warn "kCFC stopped because a cluster has at most max(0.01n, 3) curves; consider a smaller k"
    status === :lost_cluster && @warn "kCFC stopped because it lost a cluster; consider a smaller k"
    grid = initial_fpca.grid
    centers = GridFunctionalData(reduce(vcat, [transpose(vec(_fpca_mean_values(f.mean, grid))) for f in fpcas]), grid)
    settings = (; fpca_options=initial_options, cluster_fpca_options=cluster_options,
        maxiter=Int(maxiter), kmeans_options, initial=initial === nothing ? :kmeans_scores : :supplied,
        criterion=:integrated_squared_error, quadrature=:trapezoid_observed_times,
        scores=:cluster_fpca_project_scores, rng=typeof(rng))
    KCFC{T,typeof(initial_clustering)}(k, labels, _cluster_counts(labels, k), labels0, initial_fpca,
        initial_clustering, Vector{FPCA{T}}(fpcas), centers, costs, history, step - 1, status,
        fd.ids === nothing ? nothing : deepcopy(fd.ids), settings)
end

fit(::Type{KCFC}, fd::AbstractFunctionalData; kwargs...) =
    throw(ArgumentError("kCFC requires GridFunctionalData or IrregularFunctionalData"))

function _kcfc_initial(::Nothing, f, k, maxiter, kmeans_options, rng)
    options = merge((; maxiter=Int(maxiter), rng), kmeans_options)
    clustering = fit(FunctionalKMeans, f; k, options...)
    copy(clustering.assignments), clustering
end

function _kcfc_initial(initial, f, k, maxiter, kmeans_options, rng)
    isempty(kmeans_options) || throw(ArgumentError("kmeans_options require initial=nothing"))
    _validate_labels(initial, nobs(f), k, "initial"), nothing
end

function _validate_labels(labels, n, k, name)
    labels isa AbstractVector{<:Integer} || throw(ArgumentError("$name must be a vector of integer labels"))
    length(labels) == n || throw(DimensionMismatch("$name must have one label per curve ($n)"))
    all(c -> 1 <= c <= k, labels) && length(unique(labels)) == k ||
        throw(ArgumentError("$name labels must use every cluster 1:$k"))
    collect(Int, labels)
end

_kcfc_fits(fd, labels, k, options) = map(c -> fit(FPCA, fd[findall(==(c), labels)]; options...), 1:k)

function _kcfc_costs(f::FPCA, fd)
    residual = _observation_residuals(fd, f.mean, f.basis, project_scores(f, fd))
    [_trapezoid_squares(residual[i]...) for i in 1:nobs(residual)]
end

function _trapezoid_squares(t, r)
    total = zero(eltype(r))
    for j in 1:length(t)-1
        total += (t[j+1] - t[j]) * (abs2(r[j]) + abs2(r[j+1])) / 2
    end
    total
end

nobs(m::KCFC) = length(m.assignments)

function Base.show(io::IO, m::KCFC)
    print(io, "KCFC(n=", nobs(m), ", k=", m.k, ", counts=", m.counts, ", status=", m.status,
        ", iterations=", m.iterations, ", components=", map(ncomponents, m.fpcas),
        ", initial=", m.settings.initial, ", settings=", _show_vectors(m.settings), ")")
end

# Work grids in stored options are shown by length.
_show_vectors(x::NamedTuple) = map(_show_vectors, x)
_show_vectors(x::AbstractVector{<:Real}) = length(x) > 10 ? "$(length(x))-point vector" : x
_show_vectors(x) = x
Base.show(io::IO, ::MIME"text/plain", m::KCFC) = show(io, m)

"""
    FunctionalGaussianMixture
    fit(FunctionalGaussianMixture, fd::AbstractFunctionalData;
        fpca_options=(; ncomponents=FVE(0.9)), kwargs...)
    fit(FunctionalGaussianMixture, f::FPCA; k=3, init=:emem, criterion=BIC(),
        maxiter=1000, tol=1e-6, parameter_tol=nothing, short_iter=200,
        short_tol=0.01, restarts=10, rng=Random.default_rng())

Model-based clustering in the FPC subspace, following fdapace 0.6.0
`FClust(cmethod="EMCluster")`: fit [`FPCA`](@ref) to all curves (FVE 0.9 by
default, as in FClust) and a `k`-component Gaussian mixture with unrestricted
covariance per component (EMCluster's only model, mclust "VVV") to the scores.
EM steps are GaussianMixtures.jl `em!`; the initialization and stopping rules
follow EMCluster 0.2-18:

- `init=:emem` is EMCluster's `em.EM`: repeat random starts (`k` distinct
  curves as centres, nearest-centre partition, class means and covariances
  with denominator `n_c - 1`, proportions `n_c/n`; partitions with a class of
  at most `K` curves are redrawn) followed by short EM runs that stop when the
  log-likelihood gain relative to the start's total gain is at most
  `short_tol=0.01`, until `short_iter=200` short iterations are spent. Each
  start counts at least one iteration, so the loop ends; EMCluster can count a
  start as minus one. The start with the largest log likelihood (the last on
  ties, as in EMCluster) is run to convergence.
- `init` may instead be initial labels in `1:k`, converted to parameters as
  above (for supplied initializations and reference comparisons).
- EM stops when the relative log-likelihood change is at most `tol=1e-6`
  (EMCluster `em.eps`), after `maxiter=1000` iterations (`em.iter`), or when the
  log likelihood would decrease; that step is reverted, as in EMCluster.
  Decreases within `100 eps |loglik|` count as rounding, not as decreases.
  The log-likelihood rule leaves parameters about `sqrt(tol)` from the EM
  fixed point; with a positive number `parameter_tol`, EM instead stops when
  no weight, mean or covariance entry changes by more than `parameter_tol`.
  (Zero is not allowed: rounding can keep an EM step from returning exactly
  the same parameters.) EMCluster with `em.eps=0` stopped while one more EM
  step still moved means by `2e-8` (`test/reference/clustering_m8.R`).
- `restarts=10`: if a final cluster has at most `K` curves, or the final EM
  degenerates (a component's posterior weight sum falls to at most `K`, or a
  covariance becomes singular), `:emem` is repeated up to this many times in
  total (EMCluster `min.n.iter`). After the last attempt a degenerate EM
  throws, and a small cluster is kept with status `:unstable` and a warning.

FClust calls `emcluster` a second time with the same criterion after
`em.EM`, which adds at least one EM iteration; the package runs one EM.
Random draws use `rng`; EMCluster's use R's generator and cannot be
reproduced. With supplied labels, a component whose posterior weight sum
falls to at most `K` throws, rather than producing a singular covariance.

`k=3` is FClust's default. A vector or range of `k` fits each value with
`init=:emem` and keeps the smallest `criterion` (`AIC()` or `BIC()`; the first
on ties), using EMCluster's `em.aic`/`em.bic` with
`dof = kK + kK(K+1)/2 + k - 1`. A `k` that fails in all `restarts` attempts
gets criterion `Inf` and is listed in `selection.degenerate`. A failure is a
degenerate final EM, an initialization that finds no partition into classes
of more than `K` curves in 1000 draws, or every short run degenerating. A `k`
with `k(K + 1) > n` fails without any attempt. The fit throws, naming the
reason for each `k`, only if every `k` fails. FClust
has no selection; BIC is the default because it is consistent for the number
of mixture components (Keribin, 2000).

The result is a `Clustering.ClusteringResult`. Fields:

- `k`, `assignments` (largest posterior probability), `counts`, `posterior`
  (`n × k`).
- `weights`, `means` (`k × K` in score space), `covariances` (one `K × K`
  matrix per component).
- `loglikelihood`, `aic`, `bic`, `dof` (number of free parameters).
- `iterations`, `status` (`:converged`, `:maxiter`, `:likelihood_decrease`;
  `:unstable` when a final cluster has at most `K` curves).
- `fpca`, `centers` (cluster mean curves `mean + means * eigenfunctions'`).
- `initialization`: the method and, for `:emem`, the number of short-EM starts
  and their best log likelihood.
- `selection`: the criterion table when `k` was a range (`k`, `values`,
  `loglikelihood` (`NaN` when degenerate), `dof`, `degenerate`, `chosen`).
- `settings`: every option.
"""
struct FunctionalGaussianMixture{T<:AbstractFloat} <: Clustering.ClusteringResult
    k::Int
    assignments::Vector{Int}
    counts::Vector{Int}
    posterior::Matrix{T}
    weights::Vector{T}
    means::Matrix{T}
    covariances::Vector{Matrix{T}}
    loglikelihood::T
    aic::T
    bic::T
    dof::Int
    iterations::Int
    status::Symbol
    fpca::FPCA{T}
    centers::GridFunctionalData{T}
    initialization::NamedTuple
    selection::NamedTuple
    ids::Union{Nothing,Vector{Any}}
    settings::NamedTuple
end

Base.@constprop :aggressive function fit(::Type{FunctionalGaussianMixture}, fd::AbstractFunctionalData;
        fpca_options::NamedTuple=(;), kwargs...)
    options = merge((; ncomponents=FVE(0.9)), fpca_options)
    fit(FunctionalGaussianMixture, fit(FPCA, fd; options...); kwargs...)
end

Base.@constprop :aggressive function fit(::Type{FunctionalGaussianMixture}, f::FPCA; k=3, init=:emem,
        criterion=BIC(), maxiter::Integer=1000, tol::Real=1e-6, parameter_tol=nothing,
        short_iter::Integer=200, short_tol::Real=0.01, restarts::Integer=10,
        rng::AbstractRNG=Random.default_rng())
    X = copy(f.scores)
    n, p = size(X)
    _integer_in(maxiter, 1) || throw(ArgumentError("maxiter must be a positive integer"))
    _integer_in(short_iter, 1) || throw(ArgumentError("short_iter must be a positive integer"))
    _integer_in(restarts, 1) || throw(ArgumentError("restarts must be a positive integer"))
    tol isa Real && isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    short_tol isa Real && isfinite(short_tol) && short_tol >= 0 ||
        throw(ArgumentError("short_tol must be finite and nonnegative"))
    parameter_tol === nothing || (parameter_tol isa Real && isfinite(parameter_tol) && parameter_tol > 0) ||
        throw(ArgumentError("parameter_tol must be nothing or finite and positive"))
    criterion isa Union{AIC,BIC} || throw(ArgumentError("criterion must be AIC() or BIC()"))
    candidates = k isa Integer ? [k] : k isa AbstractVector{<:Integer} ? collect(k) :
        throw(ArgumentError("k must be an integer or a vector of integers"))
    isempty(candidates) && throw(ArgumentError("k must contain at least one value"))
    for c in candidates
        _integer_in(c, 1, n - 1) || throw(ArgumentError("every k must be between 1 and nobs-1 ($(n-1))"))
    end
    allunique(candidates) || throw(ArgumentError("k values must be distinct"))
    if !(init === :emem)
        k isa Integer || throw(ArgumentError("supplied initial labels require a single k"))
        init isa AbstractVector{<:Integer} || throw(ArgumentError("init must be :emem or a vector of labels"))
    end
    em = (; maxiter=Int(maxiter), tol=float(tol), parameter_tol)
    fits = [_gmm_fit(X, Int(c), init, em, Int(short_iter), float(short_tol), Int(restarts), rng)
        for c in candidates]
    # A failed k is reported by a String naming the reason.
    failed = candidates[[x isa String for x in fits]]
    if length(failed) == length(candidates)
        reasons = join(("k = $c: $x" for (c, x) in zip(candidates, fits)), "; ")
        throw(ArgumentError("$reasons; reduce k or the number of components"))
    end
    # In a range, a k that fails in every attempt is recorded and skipped.
    values = [x isa String ? convert(eltype(X), Inf) : criterion isa BIC ? x.bic : x.aic for x in fits]
    chosen = argmin(values)
    r = fits[chosen]
    selection = length(candidates) == 1 ? (; criterion=nameof(typeof(criterion)), k=candidates, values) :
        (; criterion=nameof(typeof(criterion)), k=candidates, values,
            loglikelihood=[x isa String ? convert(eltype(X), NaN) : x.loglik for x in fits],
            dof=[p * c + c * p * (p + 1) ÷ 2 + c - 1 for c in candidates], degenerate=failed,
            chosen=candidates[chosen])
    r.status === :unstable && @warn "Gaussian mixture: a final cluster has at most $p curves (no stable solution)"
    centers = reconstruct(f, r.means)
    settings = (; covariance=:full, init=init isa Symbol ? init : :supplied_labels,
        maxiter=Int(maxiter), tol=float(tol), parameter_tol, short_iter=Int(short_iter),
        short_tol=float(short_tol), restarts=Int(restarts), criterion=nameof(typeof(criterion)),
        implementation=:GaussianMixtures_em, rng=typeof(rng))
    FunctionalGaussianMixture{eltype(X)}(candidates[chosen], r.labels, _cluster_counts(r.labels, candidates[chosen]),
        r.posterior, r.weights, r.means, r.covariances, r.loglik, r.aic, r.bic, r.dof, r.iterations,
        r.status, f, centers, r.initialization, selection,
        f.ids === nothing ? nothing : deepcopy(f.ids), settings)
end

function _gmm_fit(X::Matrix{T}, k, init, em, short_iter, short_tol, restarts, rng) where {T}
    n, p = size(X)
    if init === :emem
        k * (p + 1) > n && return "em.EM initialization needs k classes of more than $p curves each, " *
            "but there are only $n curves"
        attempt = 0
        local result
        while true
            attempt += 1
            start = _gmm_emem(X, k, rng, short_iter, short_tol)
            if start isa String
                attempt >= restarts && return start
                continue
            end
            result = _gmm_em!(start.gmm, X; em..., short=false, strict=false)
            if result.status === :degenerate
                attempt >= restarts && return "the final EM degenerated (a component with posterior " *
                    "weight sum at most $p or a singular covariance) in all $restarts em.EM attempts"
                continue
            end
            labels = _gmm_labels(result.posterior)
            stable = minimum(_cluster_counts(labels, k)) > p
            if stable || attempt >= restarts
                status = stable ? result.status : :unstable
                initialization = (; method=:emem, attempts=attempt, starts=start.starts,
                    short_iterations=start.iterations, short_loglikelihood=start.loglik)
                return _gmm_summary(result, labels, status, initialization)
            end
        end
    end
    labels0 = _validate_labels(init, n, k, "init")
    gmm = _gmm_from_labels(X, labels0, k)
    gmm === nothing && throw(ArgumentError(
        "every initial class needs more curves than FPC scores ($p) and a positive definite covariance"))
    result = _gmm_em!(gmm, X; em..., short=false, strict=true)
    labels = _gmm_labels(result.posterior)
    status = minimum(_cluster_counts(labels, k)) > p ? result.status : :unstable
    _gmm_summary(result, labels, status, (; method=:supplied_labels, labels=labels0))
end

function _gmm_summary(result, labels, status, initialization)
    gmm = result.gmm
    n, p = size(result.posterior, 1), gmm.d
    k = gmm.n
    dof = p * k + k * p * (p + 1) ÷ 2 + k - 1
    loglik = result.loglik
    (; labels, posterior=result.posterior, weights=copy(gmm.w), means=copy(gmm.μ),
        covariances=map(_gmm_covariance, gmm.Σ), loglik, aic=-2loglik + 2dof,
        bic=-2loglik + dof * log(n), dof, iterations=result.iterations, status, initialization)
end

_gmm_labels(posterior) = [argmax(view(posterior, i, :)) for i in axes(posterior, 1)]

# EMCluster `initials`: class means, covariances with denominator n_c - 1 and
# proportions n_c / n. Classes with at most p members are rejected.
function _gmm_from_labels(X::Matrix{T}, labels, k) where {T}
    n, p = size(X)
    weights = zeros(T, k)
    means = zeros(T, k, p)
    factors = UpperTriangular{T,Matrix{T}}[]
    for c in 1:k
        members = findall(==(c), labels)
        length(members) > p || return nothing
        Y = X[members, :]
        mu = vec(sum(Y; dims=1)) ./ length(members)
        centred = Y .- transpose(mu)
        S = Symmetric(transpose(centred) * centred ./ (length(members) - 1))
        factor = _gmm_precision_factor(S)
        factor === nothing && return nothing
        weights[c] = length(members) / n
        means[c, :] = mu
        push!(factors, factor)
    end
    GaussianMixtures.GMM(weights, means, factors, [GaussianMixtures.History("initial parameters from labels")], n)
end

# GaussianMixtures stores full covariances as the upper Cholesky factor of
# their inverse.
function _gmm_precision_factor(S)
    decomposition = cholesky(S; check=false)
    issuccess(decomposition) || return nothing
    precision = cholesky(Symmetric(inv(decomposition)); check=false)
    issuccess(precision) || return nothing
    precision.U
end

# Log densities and posteriors from GaussianMixtures, and the total log likelihood.
function _gmm_estep(gmm, X)
    posterior, logdensity = GaussianMixtures.gmmposterior(gmm, X)
    logweights = log.(gmm.w)
    loglik = zero(eltype(posterior))
    for i in axes(logdensity, 1)
        terms = view(logdensity, i, :) .+ logweights
        top = maximum(terms)
        loglik += top + log(sum(exp.(terms .- top)))
    end
    posterior, loglik
end

function _gmm_parameters(gmm)
    (; w=copy(gmm.w), mu=copy(gmm.μ), sigma=deepcopy(gmm.Σ))
end

function _gmm_restore!(gmm, saved)
    gmm.w = saved.w
    gmm.μ = saved.mu
    gmm.Σ = saved.sigma
    gmm
end

_gmm_covariance(factor) = (inverse = inv(factor); inverse * transpose(inverse))

function _gmm_change(gmm, saved)
    covariances = map(_gmm_covariance, gmm.Σ)
    previous = map(_gmm_covariance, saved.sigma)
    max(maximum(abs, gmm.w .- saved.w), maximum(abs, gmm.μ .- saved.mu),
        maximum(c -> maximum(abs, c), covariances .- previous))
end

# One EMCluster-style EM run with GaussianMixtures `em!` steps. `short=true`
# uses shortemcluster's gain relative to the run's start; otherwise emcluster's
# relative change. A likelihood decrease reverts the step and stops.
function _gmm_em!(gmm, X; maxiter, tol, parameter_tol, short, strict)
    n, p = size(X)
    posterior, loglik = _gmm_estep(gmm, X)
    start = loglik
    iterations = 0
    status = :maxiter
    while iterations < maxiter
        effective = vec(sum(posterior; dims=1))
        if any(<=(p), effective)
            strict && throw(ArgumentError("a mixture component has posterior weight sum " *
                "at most the number of FPC scores ($p); reduce k or the number of components"))
            return (; gmm, posterior, loglik, iterations, status=:degenerate)
        end
        saved = _gmm_parameters(gmm)
        try
            GaussianMixtures.em!(gmm, X; nIter=1, parallel=false)
        catch err
            err isa PosDefException || rethrow()
            strict && throw(ArgumentError("a mixture covariance became singular; reduce k or the number of components"))
            _gmm_restore!(gmm, saved)
            return (; gmm, posterior, loglik, iterations, status=:degenerate)
        end
        newposterior, newloglik = _gmm_estep(gmm, X)
        if !(newloglik >= loglik - 100 * eps(typeof(loglik)) * abs(loglik))
            _gmm_restore!(gmm, saved)
            status = :likelihood_decrease
            break
        end
        iterations += 1
        change = short ? abs((loglik - newloglik) / (start - newloglik)) : abs((loglik - newloglik) / loglik)
        parameter_change = parameter_tol === nothing ? nothing : _gmm_change(gmm, saved)
        posterior, loglik = newposterior, newloglik
        converged = parameter_tol === nothing || short ? !(change > tol) : parameter_change <= parameter_tol
        if converged
            status = :converged
            break
        end
    end
    (; gmm, posterior, loglik, iterations, status)
end

# EMCluster shortems/randomEMinit with a total short-iteration budget.
function _gmm_emem(X, k, rng, short_iter, short_tol)
    n, p = size(X)
    best = nothing
    best_loglik = -Inf
    spent = 0
    starts = 0
    while spent < short_iter
        gmm = nothing
        draws = 0
        while gmm === nothing
            draws += 1
            draws > 1000 && return "em.EM initialization drew 1000 partitions, each with a class of " *
                "at most $p curves or a singular class covariance"
            seeds = randperm(rng, n)[1:k]
            labels = [argmin([sum(abs2, view(X, i, :) .- view(X, s, :)) for s in seeds]) for i in 1:n]
            gmm = _gmm_from_labels(X, labels, k)
        end
        result = _gmm_em!(gmm, X; maxiter=short_iter - spent, tol=short_tol, parameter_tol=nothing,
            short=true, strict=false)
        starts += 1
        spent += max(result.iterations, 1)
        if result.status !== :degenerate && result.loglik >= best_loglik
            best, best_loglik = deepcopy(result.gmm), result.loglik
        end
    end
    best === nothing && return "every em.EM short run degenerated"
    (; gmm=best, loglik=best_loglik, starts, iterations=spent)
end

nobs(m::FunctionalGaussianMixture) = length(m.assignments)
loglikelihood(m::FunctionalGaussianMixture) = m.loglikelihood
aic(m::FunctionalGaussianMixture) = m.aic
bic(m::FunctionalGaussianMixture) = m.bic
dof(m::FunctionalGaussianMixture) = m.dof

function _gmm(m::FunctionalGaussianMixture{T}) where {T}
    factors = UpperTriangular{T,Matrix{T}}[_gmm_precision_factor(Symmetric(S)) for S in m.covariances]
    GaussianMixtures.GMM(copy(m.weights), copy(m.means), factors, GaussianMixtures.History[], nobs(m))
end

"""
    posterior_probabilities(m::FunctionalGaussianMixture[, newdata])

Return the `n × k` matrix of posterior cluster probabilities: the training
values, or those of `newdata` projected with [`project_scores`](@ref) of the
stored FPCA.
"""
posterior_probabilities(m::FunctionalGaussianMixture) = copy(m.posterior)
posterior_probabilities(m::FunctionalGaussianMixture, newdata) =
    first(_gmm_estep(_gmm(m), project_scores(m.fpca, newdata)))

"""
    predict(m::FunctionalGaussianMixture, newdata)

Cluster labels of new curves: the largest posterior probability of their
projected FPC scores (ties go to the lowest index).
"""
predict(m::FunctionalGaussianMixture, newdata::AbstractFunctionalData) =
    _gmm_labels(posterior_probabilities(m, newdata))

function Base.show(io::IO, m::FunctionalGaussianMixture)
    print(io, "FunctionalGaussianMixture(n=", nobs(m), ", k=", m.k, ", K=", size(m.means, 2),
        ", counts=", m.counts, ", loglikelihood=", m.loglikelihood, ", bic=", m.bic,
        ", status=", m.status, ", iterations=", m.iterations, ", initialization=", m.initialization.method,
        ", selection=", m.selection, ", settings=", m.settings, ")")
end
Base.show(io::IO, ::MIME"text/plain", m::FunctionalGaussianMixture) = show(io, m)
