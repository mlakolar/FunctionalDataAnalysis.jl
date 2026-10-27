# --- Class labels ---------------------------------------------------------------

# Classes are the sorted distinct labels (R's factor levels); codes index them.
function _classification_labels(y, n)
    y isa AbstractVector || throw(ArgumentError("class labels must be a vector"))
    Base.require_one_based_indexing(y)
    length(y) == n || throw(DimensionMismatch("class labels need one entry per curve ($n), got $(length(y))"))
    any(ismissing, y) && throw(ArgumentError("class labels must not be missing"))
    any(v -> v isa AbstractFloat && !isfinite(v), y) && throw(ArgumentError("numeric class labels must be finite"))
    classes = try
        sort!(collect(unique(y)))
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError("class labels must be sortable with isless"))
    end
    length(classes) >= 2 || throw(ArgumentError("classification needs at least two classes"))
    lookup = Dict(c => g for (g, c) in enumerate(classes))
    (classes, Int[lookup[v] for v in y])
end

_classification_counts(codes, G) = [count(==(g), codes) for g in 1:G]

function _classification_output(classes, P, type)
    type === :label && return classes[[argmax(view(P, i, :)) for i in axes(P, 1)]]
    type === :probability && return P
    throw(ArgumentError("type must be :label or :probability"))
end

# --- Curves on a common grid (shared with depth) ----------------------------------

_classification_components(prepared::NamedTuple) = [prepared]
_classification_components(prepared::Vector) = prepared

# Training curves as one matrix per component on its evaluation grid, with the
# representation and any FPCA fit used to reach the grid (see `depth`).
function _classification_curves(x, grid, fpca)
    prepared = _depth_prepare(x, nothing, grid, fpca, true)
    components = [(; values=c.target, grid=c.grid, representation=c.representation, fpca=c.fpca)
        for c in _classification_components(prepared)]
    (components, x isa MultivariateFunctionalData)
end

# New curves in the training representation: the training grid, exact basis
# evaluation, or the stored FPCA reconstruction (never refitted).
function _classification_new_values(fd, c)
    fd isa FPCA && throw(ArgumentError("new curves must be functional data, not an FPCA fit"))
    fd isa Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData} || throw(ArgumentError(
        "new curves must be GridFunctionalData, IrregularFunctionalData or BasisFunctionalData"))
    values = if c.fpca !== nothing
        # Training curves reuse their stored scores; basis curves meet a raw
        # FPCA on its grid (the FPCRegression prediction rules).
        reconstruct(c.fpca, _regression_project(c.fpca, fd, nothing); grid=c.grid).values
    elseif fd isa GridFunctionalData
        fd.observed === nothing || throw(ArgumentError(
            "the classifier was trained on complete curves; new grid curves must be complete"))
        fd.grid == c.grid || throw(ArgumentError(
            "new grid curves must use the training evaluation grid; convert with to_grid"))
        copy(fd.values)
    elseif fd isa BasisFunctionalData
        _component_domain(fd, c.grid)
        evaluate(fd, c.grid)
    else
        throw(ArgumentError("the classifier was trained on complete curves without FPCA; irregular " *
            "new curves need a classifier trained on irregular data or on an FPCA fit"))
    end
    all(isfinite, values) || throw(ArgumentError("new curves must be finite on the evaluation grid"))
    values
end

function _classification_new_curves(components, multivariate, x)
    if multivariate
        x isa MultivariateFunctionalData || throw(ArgumentError(
            "the classifier was trained on multivariate curves; new curves must be MultivariateFunctionalData"))
        ncomponents(x) == length(components) || throw(DimensionMismatch(
            "new curves have $(ncomponents(x)) components; the classifier has $(length(components))"))
        return [_classification_new_values(component(x, j), c) for (j, c) in enumerate(components)]
    end
    x isa MultivariateFunctionalData && throw(ArgumentError(
        "the classifier was trained on univariate curves; new curves must be univariate"))
    [_classification_new_values(x, only(components))]
end

# Trapezoidal L² distances; for several components sqrt(Σ_c ‖x_c - y_c‖²).
function _classification_distances(xs, ys, components)
    length(components) == 1 && return _depth_distances(only(xs), only(ys), only(components).grid)
    sqrt.(sum(abs2.(_depth_distances(x, y, c.grid)) for (x, y, c) in zip(xs, ys, components)))
end

_classification_fd(values, components, rows) = length(components) == 1 ?
    GridFunctionalData(only(values)[rows, :], only(components).grid) :
    MultivariateFunctionalData([GridFunctionalData(v[rows, :], c.grid) for (v, c) in zip(values, components)])

_classification_settings(components) = (; grid=[c.grid for c in components],
    representation=[c.representation for c in components], fpca=[c.fpca for c in components])

# --- k nearest neighbours -----------------------------------------------------------

function _knn_candidates(k, available, default_max)
    if k === nothing
        top = min(default_max, available)
        top >= 3 || throw(ArgumentError(
            "the default k grid 3, 5, ... needs at least three neighbours; give k explicitly"))
        return collect(3:2:top)
    end
    candidates = k isa Integer ? [k] : k isa AbstractVector{<:Integer} ? collect(k) :
        throw(ArgumentError("k must be a positive integer or a vector of them"))
    !isempty(candidates) && !any(c -> c isa Bool, candidates) ||
        throw(ArgumentError("k must be a positive integer or a nonempty vector of them"))
    all(c -> 1 <= c <= available, candidates) || throw(ArgumentError(
        "every k must lie between 1 and $available (the number of available neighbours)"))
    sort!(unique(Int.(candidates)))
end

# fda.usc S.KNN: the neighbours are the references within the k-th smallest
# distance (ties at that distance included); probabilities are class shares.
# `self` leaves each reference out of its own neighbourhood.
function _knn_probabilities(D::AbstractMatrix{T}, codes, G, k, self) where {T}
    n, n0 = size(D)
    P = zeros(T, n, G)
    row = Vector{T}(undef, n0)
    for i in 1:n
        row .= view(D, i, :)
        self && (row[i] = T(Inf))
        radius = partialsort(row, k)
        members = 0
        for j in 1:n0
            if row[j] <= radius
                P[i, codes[j]] += 1
                members += 1
            end
        end
        P[i, :] ./= members
    end
    P
end

# Largest probability; ties go to the class of the nearest tied neighbour.
function _knn_labels(P, D, codes, self)
    labels = Vector{Int}(undef, size(P, 1))
    for i in axes(P, 1)
        best = maximum(view(P, i, :))
        tied = findall(==(best), view(P, i, :))
        choice, nearest = first(tied), typemax(eltype(D))
        if length(tied) > 1
            for j in axes(D, 2)
                (self && j == i) && continue
                if codes[j] in tied && D[i, j] < nearest
                    choice, nearest = codes[j], D[i, j]
                end
            end
        end
        labels[i] = choice
    end
    labels
end

# Leave-one-out misclassification for each candidate k; the first minimum wins.
function _knn_select(D::AbstractMatrix{T}, codes, G, candidates) where {T}
    n = length(codes)
    errors = Vector{T}(undef, length(candidates))
    fits = map(enumerate(candidates)) do (c, k)
        P = _knn_probabilities(D, codes, G, k, true)
        labels = _knn_labels(P, D, codes, true)
        errors[c] = count(labels .!= codes) / T(n)
        (; P, labels)
    end
    chosen = argmin(errors)
    (; k=candidates[chosen], errors, probabilities=fits[chosen].P, labels=fits[chosen].labels)
end

"""
    FunctionalKNN
    fit(FunctionalKNN, x, y; k=nothing, grid=nothing, fpca=(;))

k-nearest-neighbour classifier for curves with the L² distance (fda.usc 2.2.0
`classif.knn`). `x` is `GridFunctionalData`, `IrregularFunctionalData`,
`BasisFunctionalData`, `MultivariateFunctionalData` or an `FPCA` fit; `y` holds
one label per curve, of any sortable type. The classes are the sorted distinct
labels, stored in `classes`; probability columns follow that order.

Curves reach a common grid as in [`depth`](@ref): complete grid data use their
grid, basis data are evaluated on `grid`, and irregular or incomplete curves
are represented by an FPCA reconstruction (`fit(FPCA, x; method=:pace,
fpca...)`). Distances use the stored trapezoidal weights, as fda.usc's
`metric.lp`; for multivariate data they are `sqrt(Σ_c ‖x_c - y_c‖²)`.

The neighbours of a curve are the training curves within its k-th smallest
distance (ties included, as fda.usc's `S.KNN`), and the class probabilities
are their class shares. The label is the most probable class; ties go to the
class of the nearest tied neighbour. fda.usc intends the same tie-break but
indexes curves by class number, so its tied labels are arbitrary.

`k=nothing` chooses `k` among `3, 5, ..., min(15, n-1)` by leave-one-out
misclassification, fda.usc's default grid and criterion; a vector of
candidates (or one integer) can be given instead. Ties choose the smallest `k`.
fda.usc's grid ends at `min(15, n)`, which can include `k = n`, more than the
`n - 1` other curves, when `n <= 15`. `loo_error` stores the criterion for
every candidate, and `fitted` returns the leave-one-out labels at the chosen
`k`, as fda.usc's `group.est`; `fitted_probabilities` holds their
probabilities. Use `predict` for new curves.
"""
struct FunctionalKNN{T<:AbstractFloat,L,C,S<:NamedTuple}
    classes::Vector{L}
    labels::Vector{Int}
    curves::C
    k::Int
    candidates::Vector{Int}
    loo_error::Vector{T}
    fitted_labels::Vector{Int}
    fitted_probabilities::Matrix{T}
    settings::S
end

function fit(::Type{FunctionalKNN}, x, y; k=nothing, grid=nothing, fpca::NamedTuple=(;))
    components, multivariate = _classification_curves(x, grid, fpca)
    n = size(first(components).values, 1)
    classes, codes = _classification_labels(y, n)
    values = [c.values for c in components]
    D = _classification_distances(values, values, components)
    candidates = _knn_candidates(k, n - 1, 15)
    chosen = _knn_select(D, codes, length(classes), candidates)
    settings = merge((; selection=:leave_one_out, tie_rule=:smallest_k,
        distance=multivariate ? :euclidean_L2_trapezoidal : :L2_trapezoidal,
        neighbourhood=:kth_distance_ties_included, label_tie_break=:nearest_neighbour,
        multivariate, fpca_options=fpca), _classification_settings(components),
        (; reference="fda.usc 2.2.0 classif.knn"))
    FunctionalKNN(classes, codes, components, chosen.k, candidates, chosen.errors, chosen.labels,
        chosen.probabilities, settings)
end

"""
    predict(f::FunctionalKNN, x; type=:label)

Classify new curves `x` with the stored training curves and `k`. New curves
are brought to the training representation: the training grid (complete grid
data must use it), exact evaluation of basis data, or the stored FPCA
reconstruction when the classifier was trained through FPCA (no refit).
`type=:label` returns labels; `type=:probability` the `n × G` matrix of
neighbour class shares, columns in `f.classes` order.
"""
function predict(f::FunctionalKNN, x; type::Symbol=:label)
    values = _classification_new_curves(f.curves, f.settings.multivariate, x)
    D = _classification_distances(values, [c.values for c in f.curves], f.curves)
    P = _knn_probabilities(D, f.labels, length(f.classes), f.k, false)
    type === :label && return f.classes[_knn_labels(P, D, f.labels, false)]
    _classification_output(f.classes, P, type)
end

# --- Depth-based classifiers --------------------------------------------------------

# Resolve random choices once so every class and every prediction uses them.
_classification_method(method::_DepthMethod, components, values, shared) = method
function _classification_method(method::RandomProjectionDepth, components, values, shared)
    method.projections === nothing || return method
    grid = first(components).grid
    all(c -> c.grid == grid, components) || throw(ArgumentError(
        "multivariate RandomProjectionDepth needs every component on the same evaluation grid"))
    RandomProjectionDepth(; projections=_depth_projections(method, grid), scale=method.scale)
end
# classif.DD passes the bandwidth of the pooled sample to every class.
function _classification_method(method::ModalDepth, components, values, shared)
    shared && method.bandwidth === nothing || return method
    pooled = depth(_classification_fd(values, components, axes(first(values), 1)), method)
    ModalDepth(; bandwidth=pooled.settings.bandwidth, kernel=method.kernel)
end

# Depth of every target curve with respect to each class of training curves.
function _classification_depths(targets, components, values, codes, G, method, weights)
    target = _classification_fd(targets, components, axes(first(targets), 1))
    parts = [depth(target, _classification_fd(values, components, findall(==(g), codes)), method; weights)
        for g in 1:G]
    (reduce(hcat, [d.values for d in parts]), parts)
end

_classification_bandwidths(::_DepthMethod, parts) = nothing
_classification_bandwidths(::ModalDepth, parts) = [d.settings.bandwidth for d in parts]

function _classification_depth_inputs(x, y, grid, fpca)
    components, multivariate = _classification_curves(x, grid, fpca)
    values = [c.values for c in components]
    n = size(first(values), 1)
    classes, codes = _classification_labels(y, n)
    counts = _classification_counts(codes, length(classes))
    all(>=(2), counts) || throw(ArgumentError(
        "depth classifiers need at least two curves in every class; class sizes are $counts"))
    (; components, multivariate, values, classes, codes, counts)
end

"""
    DepthClassifier
    fit(DepthClassifier, x, y; method=RandomProjectionDepth(), weights=nothing, grid=nothing, fpca=(;))

Maximum-depth classifier (fda.usc 2.2.0 `classif.depth`; Cuevas, Febrero and
Fraiman, 2007): each curve is assigned to the class in which it is deepest.
`depths` holds the training depths, one column per class in `classes` order,
each computed with [`depth`](@ref) with respect to that class's curves (a
training curve belongs to its own class's reference, as in fda.usc). The first
class wins ties, as R's `which.max`. `x`, `y`, `grid` and `fpca` are as in
[`FunctionalKNN`](@ref); `method` is any depth method of [`depth`](@ref) and
`weights` its multivariate component weights. Every class needs two curves.

The default `RandomProjectionDepth()` is fda.usc's default depth (`"RP"`), with
the projections of [`RandomProjectionDepth`](@ref) (50 Gaussian-process
directions, fda.usc `depth.RP`). Generated projections are drawn once, from
the method's `rng`, and shared by all classes and predictions; the resolved
method is stored as `method`. This departs from `classif.depth`, which draws
500 directions uniformly on the sphere of grid vectors: those directions
depend on the grid resolution, while `depth.RP`'s do not. Supply
`RandomProjectionDepth(projections=...)` to fix the directions. `ModalDepth`
without a bandwidth computes one per class from that class's distances, as
`classif.depth` does; the bandwidths are stored in `settings`.

`fitted` returns the training labels (resubstitution, fda.usc's `group.est`).
No class probabilities are defined; `predict` returns labels or depths.
"""
struct DepthClassifier{T<:AbstractFloat,L,M<:_DepthMethod,C,S<:NamedTuple}
    classes::Vector{L}
    labels::Vector{Int}
    curves::C
    method::M
    depths::Matrix{T}
    fitted_labels::Vector{Int}
    settings::S
end

function fit(::Type{DepthClassifier}, x, y; method::_DepthMethod=RandomProjectionDepth(),
        weights=nothing, grid=nothing, fpca::NamedTuple=(;))
    data = _classification_depth_inputs(x, y, grid, fpca)
    resolved = _classification_method(method, data.components, data.values, false)
    D, parts = _classification_depths(data.values, data.components, data.values, data.codes,
        length(data.classes), resolved, weights)
    labels = [argmax(view(D, i, :)) for i in axes(D, 1)]
    settings = merge((; rule=:maximum_depth, tie_rule=:first_class, weights,
        class_bandwidths=_classification_bandwidths(resolved, parts), multivariate=data.multivariate,
        fpca_options=fpca), _classification_settings(data.components),
        (; reference="fda.usc 2.2.0 classif.depth"))
    DepthClassifier(data.classes, data.codes, data.components, resolved, D, labels, settings)
end

"""
    predict(f::DepthClassifier, x; type=:label)

Classify new curves by maximum depth with respect to the stored training
classes, with the stored (resolved) depth method. New curves are brought to
the training representation as in
`predict(::FunctionalKNN, x)`. `type=:label` returns
labels and `type=:depth` the `n × G` depth matrix.
"""
function predict(f::DepthClassifier, x; type::Symbol=:label)
    type in (:label, :depth) || throw(ArgumentError(
        "type must be :label or :depth; maximum-depth classification defines no class probabilities"))
    targets = _classification_new_curves(f.curves, f.settings.multivariate, x)
    D, _ = _classification_depths(targets, f.curves, [c.values for c in f.curves], f.labels,
        length(f.classes), f.method, f.settings.weights)
    type === :depth ? D : _classification_output(f.classes, D, :label)
end

# --- Gaussian discriminant rules ----------------------------------------------------

function _discriminant_prior(prior, counts, ::Type{T}) where {T}
    G = length(counts)
    prior === :equal && return fill(one(T) / G, G)
    prior === :proportional && return T.(counts ./ sum(counts))
    prior isa AbstractVector{<:Real} || throw(ArgumentError(
        "prior must be :equal, :proportional or a vector of class probabilities"))
    length(prior) == G || throw(DimensionMismatch("prior needs one probability per class ($G)"))
    p = T.(collect(prior))
    all(v -> isfinite(v) && v > 0, p) || throw(ArgumentError("prior probabilities must be positive and finite"))
    abs(sum(p) - 1) <= sqrt(eps(T)) || throw(ArgumentError("prior probabilities must sum to one"))
    p
end

# Class means and moment covariances (MASS lda: pooled, divisor n - G; MASS
# qda: per class, divisor n_g - 1).
function _discriminant_fit(X::Matrix{T}, codes, G, quadratic, prior) where {T}
    n, p = size(X)
    counts = _classification_counts(codes, G)
    means = zeros(T, G, p)
    for i in 1:n
        means[codes[i], :] .+= view(X, i, :)
    end
    means ./= counts
    centered = X .- means[codes, :]
    covariances = if quadratic
        all(>(p), counts) || throw(ArgumentError(
            "quadratic discriminant analysis needs more than $p curves (the number of features) in every class; class sizes are $counts"))
        [let r = centered[codes .== g, :]; Matrix(Symmetric(transpose(r) * r ./ (counts[g] - 1))) end
            for g in 1:G]
    else
        n - G >= p || throw(ArgumentError(
            "linear discriminant analysis needs at least $(p + G) curves for $p features and $G classes"))
        [Matrix(Symmetric(transpose(centered) * centered ./ (n - G)))]
    end
    all(S -> issuccess(cholesky(Symmetric(S); check=false)), covariances) || throw(ArgumentError(
        "a discriminant covariance matrix is singular; use fewer features (components)"))
    (; means, covariances, priors=_discriminant_prior(prior, counts, T), quadratic)
end

# Posterior class probabilities of Gaussian classes, normalized on the log scale.
function _discriminant_probabilities(model, X::AbstractMatrix)
    T = promote_type(eltype(X), eltype(model.means))
    G = size(model.means, 1)
    factors = map(S -> cholesky(Symmetric(S)), model.covariances)
    logp = Matrix{T}(undef, size(X, 1), G)
    for g in 1:G
        F = model.quadratic ? factors[g] : only(factors)
        z = (X .- transpose(view(model.means, g, :))) / F.U
        logp[:, g] .= log(model.priors[g]) .- vec(sum(abs2, z; dims=2)) ./ 2 .- logdet(F) / 2
    end
    logp .-= maximum(logp; dims=2)
    P = exp.(logp)
    P ./= sum(P; dims=2)
end

"""
    FunctionalDiscriminant
    FunctionalLDA
    FunctionalQDA
    fit(FunctionalLDA, x, y; fpca=nothing, fpca_options=(;), ncomponents=nothing, prior=:equal)
    fit(FunctionalQDA, x, y; fpca=nothing, fpca_options=(;), ncomponents=nothing, prior=:equal)

Functional linear (`FunctionalLDA`) and quadratic (`FunctionalQDA`)
discriminant analysis on FPC scores, as fda.usc 2.2.0 `classif.lda` and
`classif.qda` with a principal-component basis (`create.pc.basis`). Both are
`FunctionalDiscriminant` results. Each class is Gaussian in the leading FPC
scores of the covariates; LDA pools the covariance, `S = Σ_g Σ_{i∈g} (z_i -
m_g)(z_i - m_g)' / (n - G)`, and QDA estimates one covariance per class with
divisor `n_g - 1`, as R's `MASS::lda` and `MASS::qda`. `probabilities` are the
posterior class probabilities, columns in `classes` order.

`x` is one univariate functional dataset (grid, irregular or basis), a
tuple/vector of them, or `MultivariateFunctionalData`; scores of several
covariates are concatenated. The FPCA follows [`FPCRegression`](@ref): each
covariate is fitted with `fit(FPCA, x_j)` defaults, except that complete grid
data use `method=:face` (in the regression study the unsmoothed dense route
kept dozens of noise components under FVE(0.99); here they would also make the
class covariances singular). Irregular and incomplete curves use PACE, whose
conditional-expectation scores make sparse covariates usable. `fpca`,
`fpca_options` and `ncomponents` are as in `FPCRegression`, and
`settings.fpca_method` and `settings.fpca_method_source` record the routes.

`prior=:equal` (default) gives every class probability `1/G`, which is what
fda.usc's `classif.lda`/`classif.qda` pass to MASS with their default
`weights="equal"`. `:proportional` uses the class frequencies (MASS's own
default), or give a vector of probabilities in `classes` order.

LDA needs at least `K + G` curves for `K` scores and QDA more than `K` curves
in every class; a singular covariance throws (use fewer components).
`fitted` returns the training labels (resubstitution, fda.usc's
`group.est`).
"""
struct FunctionalDiscriminant{Q,T<:AbstractFloat,L,S<:NamedTuple}
    classes::Vector{L}
    labels::Vector{Int}
    fpca::Tuple{Vararg{FPCA}}
    counts::Vector{Int}
    means::Matrix{T}
    covariances::Vector{Matrix{T}}
    priors::Vector{T}
    probabilities::Matrix{T}
    fitted_labels::Vector{Int}
    settings::S
end

"""
    FunctionalLDA

Functional linear discriminant analysis on FPC scores (pooled covariance):
`fit(FunctionalLDA, x, y; ...)` returns a [`FunctionalDiscriminant`](@ref),
which documents the options.
"""
const FunctionalLDA = FunctionalDiscriminant{false}

"""
    FunctionalQDA

Functional quadratic discriminant analysis on FPC scores (one covariance per
class): `fit(FunctionalQDA, x, y; ...)` returns a
[`FunctionalDiscriminant`](@ref), which documents the options.
"""
const FunctionalQDA = FunctionalDiscriminant{true}

fit(::Type{FunctionalLDA}, x, y; kwargs...) = _fit_discriminant(false, x, y; kwargs...)
fit(::Type{FunctionalQDA}, x, y; kwargs...) = _fit_discriminant(true, x, y; kwargs...)

function _fit_discriminant(quadratic::Bool, x, y; fpca=nothing, fpca_options=(;), ncomponents=nothing,
        prior=:equal)
    name = quadratic ? "FunctionalQDA" : "FunctionalLDA"
    curves, supplied, options, rules = _regression_arguments(x, fpca, fpca_options, ncomponents, :fpca)
    classes, codes = _classification_labels(y, nobs(first(curves)))
    models = map(curves, supplied, options, rules) do c, f, o, r
        effective, source = _regression_fpca_options(c, o, f; model=name)
        _regression_fpca(c, f, effective, r; source, model=name)
    end
    counts = Int[supplied[j] === nothing ? FunctionalDataAnalysis.ncomponents(models[j]) :
        _regression_count(rules[j], models[j]) for j in eachindex(models)]
    projected = map(_regression_fit_scores, curves, models)
    T = _floattype(promote_type(map(eltype, projected)...))
    X = reduce(hcat, [Matrix{T}(s[:, 1:k]) for (s, k) in zip(projected, counts)])
    all(isfinite, X) || throw(ArgumentError("FPC scores must be finite"))
    model = _discriminant_fit(X, codes, length(classes), quadratic, prior)
    P = _discriminant_probabilities(model, X)
    settings = (; fpca_source=map(f -> f === nothing ? :fitted : :supplied, supplied),
        fpca_options=deepcopy(options), fpca_method=map(f -> f.method, models),
        fpca_method_source=map((c, o, f) -> last(_regression_fpca_options(c, o, f; model=name)),
            curves, options, supplied),
        fpca_knots=_regression_face_knots(models), ncomponents=deepcopy(rules),
        prior=prior isa Symbol ? prior : :supplied,
        covariance=quadratic ? :class_moment : :pooled_moment, tie_rule=:first_class,
        reference=quadratic ? "fda.usc 2.2.0 classif.qda on create.pc.basis; MASS qda" :
            "fda.usc 2.2.0 classif.lda on create.pc.basis; MASS lda")
    FunctionalDiscriminant{quadratic,T,eltype(classes),typeof(settings)}(classes, codes, models,
        counts, model.means, model.covariances, model.priors, P,
        [argmax(view(P, i, :)) for i in axes(P, 1)], settings)
end

_discriminant_model(f::FunctionalDiscriminant{Q}) where {Q} =
    (; means=f.means, covariances=f.covariances, priors=f.priors, quadratic=Q)

"""
    predict(f::FunctionalDiscriminant, x; type=:label, evaluator=nothing)

Classify new curves from their scores on the stored FPCA fits (no refit; see
[`project_scores`](@ref)), with the stored component counts. `type=:label`
returns labels and `type=:probability` the posterior class probabilities,
columns in `f.classes` order. `evaluator` is as in
`predict(::FPCRegression, x)` for representations the
FPCA cannot project natively.
"""
function predict(f::FunctionalDiscriminant, x; type::Symbol=:label, evaluator=nothing)
    type in (:label, :probability) || throw(ArgumentError("type must be :label or :probability"))
    curves = _regression_inputs(x)
    length(curves) == length(f.fpca) || throw(DimensionMismatch(
        "prediction functional covariate count must match the classifier ($(length(f.fpca)))"))
    evaluators = _regression_each(evaluator, length(curves), "evaluator")
    projected = map(_regression_project, f.fpca, curves, evaluators)
    T = eltype(f.means)
    X = reduce(hcat, [Matrix{T}(s[:, 1:k]) for (s, k) in zip(projected, f.counts)])
    _classification_output(f.classes, _discriminant_probabilities(_discriminant_model(f), X), type)
end

# --- DD-classifier --------------------------------------------------------------------

# Logistic regression on the depths with an intercept (fda.usc classif.glm2boost):
# one fit for two classes (class 2 against class 1), one against the rest for each
# class otherwise.
function _dd_glm(D::Matrix{T}) where {T}
    [ones(T, size(D, 1)) D]
end
function _dd_glm_fit(D::Matrix{T}, codes, G) where {T}
    design = _dd_glm(D)
    family = _glm_family(:binomial, nothing)
    targets = G == 2 ? [2] : collect(1:G)
    fits = map(targets) do g
        try
            _regression_glm(design, T.(codes .== g), family)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("logistic regression on the depths failed for class $g ($(err.msg)); " *
                "depths that separate the classes make the likelihood unbounded, use classifier=:lda, :qda, :knn or :dd1"))
        end
    end
    (; coefficients=reduce(hcat, [f.b for f in fits]), iterations=[Int(f.irls_iterations) for f in fits])
end
function _dd_glm_probabilities(coefficients, D::Matrix{T}) where {T}
    # The inverse link returns μ and 1 - μ (accurate in both tails) first.
    parts = map(eta -> _glm_inverse_link(:logit, eta), _dd_glm(D) * coefficients)
    size(coefficients, 2) == 1 && return [getindex.(parts, 2) getindex.(parts, 1)]
    mu = first.(parts)
    mu ./ sum(mu; dims=2)
end

# Li, Cuesta-Albertos and Liu (2012), degree one: class 2 when D₂ > s·D₁, with
# the slope s minimizing the training misclassification (smallest on ties).
function _dd1_fit(D::Matrix{T}, codes) where {T}
    size(D, 2) == 2 || throw(ArgumentError(
        "classifier=:dd1 separates two classes; use :glm, :lda, :qda or :knn for more"))
    candidates = sort!(unique(vcat(zero(T), [D[i, 2] / D[i, 1] for i in axes(D, 1) if D[i, 1] > 0])))
    errors = [count(i -> _dd1_label(D, i, s) != codes[i], axes(D, 1)) / T(size(D, 1)) for s in candidates]
    candidates[argmin(errors)]
end
_dd1_label(D, i, slope) = D[i, 2] > slope * D[i, 1] ? 2 : 1

function _dd_euclidean(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    [sqrt(sum(abs2(X[i, c] - Y[j, c]) for c in axes(X, 2))) for i in axes(X, 1), j in axes(Y, 1)]
end

"""
    DDClassifier
    fit(DDClassifier, x, y; method=FraimanMuniz(), classifier=:glm, k=nothing,
        prior=:proportional, weights=nothing, grid=nothing, fpca=(;))

DD-classifier (Li, Cuesta-Albertos and Liu, 2012; fda.usc 2.2.0 `classif.DD`):
every curve is mapped to its depths with respect to each class (the DD-plot,
stored as `depths`, columns in `classes` order) and a second classifier works
on those depths. `x`, `y`, `method`, `weights`, `grid` and `fpca` are as in
[`DepthClassifier`](@ref); the defaults `method=FraimanMuniz()` and
`classifier=:glm` are fda.usc's (`depth="FM"`, `classif="glm"`). Generated
random projections are drawn once, and `ModalDepth` without a bandwidth uses
the bandwidth of the pooled training sample for every class, as `classif.DD`
does; the resolved method is stored as `method`.

`classifier` selects the second stage, whose fitted quantities are in `stage`:
- `:glm` (default): logistic regression with an intercept on the depths, by
  maximum likelihood (the IRLS of [`FPCRegression`](@ref)'s binomial family);
  for more than two classes one class against the rest, as fda.usc. Two-class
  probabilities are the fitted logistic probabilities; with more classes the
  one-against-rest probabilities are normalized to sum to one. Classes that the
  depths separate perfectly make IRLS diverge and throw.
- `:lda`, `:qda`: Gaussian discriminant analysis of the depths with `prior`
  (`:proportional` by default, since `classif.DD` calls `MASS::lda`/`qda`
  with MASS's default prior).
- `:knn`: k nearest neighbours in the Euclidean distance between depth
  vectors, with `k` chosen by leave-one-out among `3, 5, ..., min(19, n-1)` (the
  `classif.DD` default grid) unless `k` is given; see [`FunctionalKNN`](@ref).
- `:dd1`: the degree-one DD rule of Li et al. for two classes: class 2 when
  `D₂ > s D₁`, with the slope `s` (0 or a ratio `D₂/D₁` of a training curve)
  minimizing the training misclassification, smallest on ties. No
  probabilities are defined. fda.usc's `MCR0` searches the ratios `D₁/D₂` and
  counts its second error term over all curves but the first
  (`dep[-ind, 1]` with a logical `ind`), so its slopes differ.

fda.usc's polynomial rules of degree two and three (`DD2`, `DD3`), which
search random subsets of curves with a smoothed loss, and its `gam`, `np` and
tree rules are not implemented. `fitted` returns the training labels
(resubstitution) and `fitted_probabilities` their probabilities (`nothing`
for `:dd1`).
"""
struct DDClassifier{T<:AbstractFloat,L,M<:_DepthMethod,C,R<:NamedTuple,P,S<:NamedTuple}
    classes::Vector{L}
    labels::Vector{Int}
    curves::C
    method::M
    classifier::Symbol
    stage::R
    depths::Matrix{T}
    fitted_labels::Vector{Int}
    fitted_probabilities::P
    settings::S
end

function _dd_stage(classifier, D::Matrix{T}, codes, G, k, prior) where {T}
    if classifier === :glm
        k === nothing || throw(ArgumentError("k applies only to classifier=:knn"))
        prior === :proportional || throw(ArgumentError("prior applies only to classifier=:lda or :qda"))
        return _dd_glm_fit(D, codes, G)
    elseif classifier in (:lda, :qda)
        k === nothing || throw(ArgumentError("k applies only to classifier=:knn"))
        model = _discriminant_fit(D, codes, G, classifier === :qda, prior)
        return merge(model, (; prior=prior isa Symbol ? prior : :supplied))
    elseif classifier === :knn
        prior === :proportional || throw(ArgumentError("prior applies only to classifier=:lda or :qda"))
        candidates = _knn_candidates(k, size(D, 1) - 1, 19)
        chosen = _knn_select(_dd_euclidean(D, D), codes, G, candidates)
        return (; k=chosen.k, candidates, loo_error=chosen.errors, depths=copy(D),
            fitted_probabilities=chosen.probabilities, fitted_labels=chosen.labels)
    elseif classifier === :dd1
        k === nothing || throw(ArgumentError("k applies only to classifier=:knn"))
        prior === :proportional || throw(ArgumentError("prior applies only to classifier=:lda or :qda"))
        return (; slope=_dd1_fit(D, codes))
    end
    throw(ArgumentError("classifier must be :glm, :lda, :qda, :knn or :dd1"))
end

# Probabilities (nothing for :dd1) and labels of the second stage on depths D.
function _dd_predict(classifier, stage, D::Matrix{T}, codes, G) where {T}
    if classifier === :glm
        P = _dd_glm_probabilities(stage.coefficients, D)
    elseif classifier in (:lda, :qda)
        P = _discriminant_probabilities(stage, D)
    elseif classifier === :knn
        distances = _dd_euclidean(D, stage.depths)
        P = _knn_probabilities(distances, codes, G, stage.k, false)
        return P, _knn_labels(P, distances, codes, false)
    else
        return nothing, [_dd1_label(D, i, stage.slope) for i in axes(D, 1)]
    end
    P, [argmax(view(P, i, :)) for i in axes(P, 1)]
end

function fit(::Type{DDClassifier}, x, y; method::_DepthMethod=FraimanMuniz(), classifier::Symbol=:glm,
        k=nothing, prior=:proportional, weights=nothing, grid=nothing, fpca::NamedTuple=(;))
    classifier in (:glm, :lda, :qda, :knn, :dd1) ||
        throw(ArgumentError("classifier must be :glm, :lda, :qda, :knn or :dd1"))
    data = _classification_depth_inputs(x, y, grid, fpca)
    G = length(data.classes)
    resolved = _classification_method(method, data.components, data.values, true)
    D, _ = _classification_depths(data.values, data.components, data.values, data.codes, G,
        resolved, weights)
    stage = _dd_stage(classifier, D, data.codes, G, k, prior)
    P, labels = classifier === :knn ? (stage.fitted_probabilities, stage.fitted_labels) :
        _dd_predict(classifier, stage, D, data.codes, G)
    settings = merge((; classifier, weights, multivariate=data.multivariate, fpca_options=fpca,
        fitted=classifier === :knn ? :leave_one_out : :resubstitution),
        _classification_settings(data.components), (; reference="fda.usc 2.2.0 classif.DD"))
    DDClassifier(data.classes, data.codes, data.components, resolved, classifier, stage, D, labels, P,
        settings)
end

"""
    predict(f::DDClassifier, x; type=:label)

Classify new curves: their depths with respect to the stored training classes
(stored depth method), then the stored second stage. New curves are brought to
the training representation as in
`predict(::FunctionalKNN, x)`. `type=:label` returns labels,
`type=:probability` the class probabilities (columns in `f.classes` order;
not defined for `classifier=:dd1`) and `type=:depth` the `n × G` depths.
"""
function predict(f::DDClassifier, x; type::Symbol=:label)
    type in (:label, :probability, :depth) || throw(ArgumentError("type must be :label, :probability or :depth"))
    type === :probability && f.classifier === :dd1 &&
        throw(ArgumentError("classifier=:dd1 defines no class probabilities"))
    targets = _classification_new_curves(f.curves, f.settings.multivariate, x)
    G = length(f.classes)
    D, _ = _classification_depths(targets, f.curves, [c.values for c in f.curves], f.labels, G,
        f.method, f.settings.weights)
    type === :depth && return D
    P, labels = _dd_predict(f.classifier, f.stage, D, f.labels, G)
    type === :label ? f.classes[labels] : P
end

# --- Shared accessors and display -----------------------------------------------------

const _Classifier = Union{FunctionalKNN,DepthClassifier,FunctionalDiscriminant,DDClassifier}

nobs(f::_Classifier) = length(f.labels)

"""
    fitted(f::Union{FunctionalKNN,DepthClassifier,DDClassifier,FunctionalDiscriminant})

Training labels as each reference reports them: leave-one-out labels at the
chosen `k` for [`FunctionalKNN`](@ref) and `DDClassifier(classifier=:knn)`
(fda.usc `classif.knn`), resubstitution labels otherwise.
"""
fitted(f::_Classifier) = f.classes[f.fitted_labels]

_classification_error(f) = count(f.fitted_labels .!= f.labels) / length(f.labels)

Base.show(io::IO, f::FunctionalKNN) = print(io, "FunctionalKNN(n=", nobs(f), ", classes=", f.classes,
    ", k=", f.k, " (leave-one-out over ", f.candidates, "), loo_error=", f.loo_error[findfirst(==(f.k), f.candidates)],
    ", distance=:", f.settings.distance, ", representation=", f.settings.representation, ")")
Base.show(io::IO, f::DepthClassifier) = print(io, "DepthClassifier(n=", nobs(f), ", classes=", f.classes,
    ", method=", f.method, ", weights=", f.settings.weights, ", class_bandwidths=",
    f.settings.class_bandwidths, ", training_error=", _classification_error(f),
    ", representation=", f.settings.representation, ")")
Base.show(io::IO, f::DDClassifier) = print(io, "DDClassifier(n=", nobs(f), ", classes=", f.classes,
    ", method=", f.method, ", classifier=:", f.classifier, _dd_show_stage(f.classifier, f.stage),
    ", weights=", f.settings.weights, ", training_error=", _classification_error(f),
    " (", f.settings.fitted, "), representation=", f.settings.representation, ")")
_dd_show_stage(classifier, stage) = classifier === :glm ? string(", coefficients=", _show_summary(stage.coefficients)) :
    classifier === :knn ? string(", k=", stage.k, " (leave-one-out over ", stage.candidates, ")") :
    classifier === :dd1 ? string(", slope=", stage.slope) : string(", prior=", stage.priors)
Base.show(io::IO, f::FunctionalDiscriminant{Q}) where {Q} = print(io, Q ? "FunctionalQDA" : "FunctionalLDA",
    "(n=", nobs(f), ", classes=", f.classes, ", ncomponents=", f.counts, ", prior=", f.priors,
    " (:", f.settings.prior, "), fpca_method=", f.settings.fpca_method, " (", f.settings.fpca_method_source,
    "), training_error=", _classification_error(f), ")")
for T in (:FunctionalKNN, :DepthClassifier, :DDClassifier, :FunctionalDiscriminant)
    @eval Base.show(io::IO, ::MIME"text/plain", f::$T) = show(io, f)
end
