using DelimitedFiles: readdlm
import StatsAPI

const _CLASSIF_INPUT = readdlm(joinpath(@__DIR__, "data", "m8_classification_input.csv"), ',', Any; comments=true)
const _CLASSIF_GOLDEN = readdlm(joinpath(@__DIR__, "data", "m8_classification_golden.csv"), ',', Any; comments=true)

function _classif_matrix(table, rows)
    out = zeros(Int(maximum(table[rows, end-2])), Int(maximum(table[rows, end-1])))
    for row in eachrow(table[rows, :])
        out[Int(row[end-2]), Int(row[end-1])] = Float64(row[end])
    end
    out
end
_classif_input(name) = _classif_matrix(_CLASSIF_INPUT, _CLASSIF_INPUT[:, 1] .== name)
_classif_golden(case, quantity) = _classif_matrix(_CLASSIF_GOLDEN,
    (_CLASSIF_GOLDEN[:, 1] .== case) .& (_CLASSIF_GOLDEN[:, 2] .== quantity))
_classif_close(actual, case, quantity) =
    golden_relative_error(actual, _classif_golden(case, quantity)) < 1e-8
_classif_labels(codes) = ["a", "b", "c"][Int.(vec(codes))]
_classif_same(actual, case, quantity) = actual == _classif_labels(_classif_golden(case, quantity))

@testset "classification goldens against fda.usc 2.2.0" begin
    grid = vec(_classif_input("grid"))
    train = GridFunctionalData(_classif_input("x_train"), grid)
    test = GridFunctionalData(_classif_input("x_test"), grid)
    y = _classif_labels(_classif_input("g_train"))

    @testset "kNN" begin
        f = StatsAPI.fit(FunctionalKNN, train, y)
        @test f.classes == ["a", "b"]
        @test f.candidates == collect(3:2:15)
        @test _classif_close(f.loo_error, "knn", "loo_error")
        @test f.k == Int(only(_classif_golden("knn", "k")))
        @test _classif_close(f.fitted_probabilities, "knn", "loo_probability")
        @test _classif_same(StatsAPI.fitted(f), "knn", "loo_class")
        @test _classif_close(predict(f, test; type=:probability), "knn", "probability")
        @test _classif_same(predict(f, test), "knn", "class")
    end

    @testset "maximum depth" begin
        proj = _classif_input("proj")
        for (case, method) in (("maxdepth_FM", FraimanMuniz(integration=:uniform)),
                ("maxdepth_mode", ModalDepth()), ("maxdepth_RP", RandomProjectionDepth(projections=proj)),
                ("maxdepth_MB", ModifiedBandDepth()))
            f = StatsAPI.fit(DepthClassifier, train, y; method)
            @test _classif_close(f.depths, case, "depth")
            @test _classif_same(StatsAPI.fitted(f), case, "class")
            @test _classif_close(predict(f, test; type=:depth), case, "new_depth")
            @test _classif_same(predict(f, test), case, "new_class")
        end
    end

    @testset "DD-classifier" begin
        fm = FraimanMuniz(integration=:uniform)
        f = StatsAPI.fit(DDClassifier, train, y; method=fm)
        @test f.classifier === :glm
        @test _classif_close(f.depths, "dd_glm", "depth")
        @test _classif_close(predict(f, test; type=:depth), "dd_glm", "new_depth")
        @test _classif_close(f.stage.coefficients, "dd_glm", "coef")
        @test _classif_close(f.fitted_probabilities[:, 2], "dd_glm", "fitted")
        @test _classif_same(StatsAPI.fitted(f), "dd_glm", "class")
        @test _classif_close(predict(f, test; type=:probability)[:, 2], "dd_glm", "new_fitted")
        @test _classif_same(predict(f, test), "dd_glm", "new_class")
        for rule in (:lda, :qda)
            case = "dd_$rule"
            g = StatsAPI.fit(DDClassifier, train, y; method=fm, classifier=rule)
            @test _classif_close(g.fitted_probabilities, case, "posterior")
            @test _classif_same(StatsAPI.fitted(g), case, "class")
            @test _classif_close(predict(g, test; type=:probability), case, "new_posterior")
            @test _classif_same(predict(g, test), case, "new_class")
        end
        g = StatsAPI.fit(DDClassifier, train, y; method=fm, classifier=:knn)
        @test g.stage.candidates == collect(3:2:19)
        @test _classif_close(g.stage.loo_error, "dd_knn", "loo_error")
        @test g.stage.k == Int(only(_classif_golden("dd_knn", "k")))
        @test _classif_close(g.fitted_probabilities, "dd_knn", "loo_probability")
        @test _classif_close(predict(g, test; type=:probability), "dd_knn", "new_probability")
        g = StatsAPI.fit(DDClassifier, train, y; method=ModalDepth(), classifier=:lda)
        @test _classif_close([g.method.bandwidth], "dd_mode", "bandwidth")
        @test _classif_close(g.depths, "dd_mode", "depth")
        @test _classif_close(predict(g, test; type=:depth), "dd_mode", "new_depth")
    end

    @testset "functional LDA and QDA" begin
        x = GridFunctionalData(_classif_input("x_lda"), grid)
        xnew = GridFunctionalData(_classif_input("x_lda_test"), grid)
        labels = _classif_labels(_classif_input("g_lda"))
        options = (; method=:dense)
        for (T, case) in ((FunctionalLDA, "lda"), (FunctionalQDA, "qda"))
            f = StatsAPI.fit(T, x, labels; fpca_options=options, ncomponents=3)
            @test f isa FunctionalDiscriminant
            @test f.counts == [3]
            @test _classif_close(f.priors, case, "prior")
            @test _classif_close(f.probabilities, case, "posterior")
            @test _classif_same(StatsAPI.fitted(f), case, "class")
            @test _classif_close(predict(f, xnew; type=:probability), case, "new_posterior")
            @test _classif_same(predict(f, xnew), case, "new_class")
        end
    end
end

# Constant curves on [0, 1]: the L² distance is the difference of the constants.
_classif_constant(levels) = GridFunctionalData(repeat(Float64.(levels), 1, 3), [0.0, 0.5, 1.0])

@testset "kNN rules" begin
    train = _classif_constant([0.0, 10.0, 0.5, 11.0])
    y = [:a, :a, :b, :b]
    f = StatsAPI.fit(FunctionalKNN, train, y; k=2)
    @test f.classes == [:a, :b] && f.k == 2 && f.candidates == [2]
    # Both neighbours of 0.2 and of 0.3 split 1:1; the nearest neighbour decides.
    @test predict(f, _classif_constant([0.2, 0.3])) == [:a, :b]
    @test predict(f, _classif_constant([0.2]); type=:probability) == [0.5 0.5]
    # Neighbours tied at the k-th distance all count.
    g = StatsAPI.fit(FunctionalKNN, train, y; k=1)
    @test predict(g, _classif_constant([0.25]); type=:probability) == [0.5 0.5]
    @test predict(g, _classif_constant([10.4])) == [:a]
    # Leave-one-out: each curve's nearest other curve has the other label.
    @test StatsAPI.fitted(g) == [:b, :b, :a, :a] && g.loo_error == [1.0]
    @test nobs(g) == 4
    h = StatsAPI.fit(FunctionalKNN, train, y; k=[3, 1, 3])
    @test h.candidates == [1, 3] && length(h.loo_error) == 2
    shown = sprint(show, f)
    @test occursin("FunctionalKNN(", shown) && occursin("k=2", shown) && occursin("L2_trapezoidal", shown)
    @test sprint(show, MIME"text/plain"(), f) == shown
    # Default grid 3, 5, ... up to min(15, n - 1).
    grid = collect(range(0, 1; length=11))
    values = [sin(i * t) + i / 5 for i in 1:20, t in grid]
    labels = repeat(["x", "y"], 10)
    @test StatsAPI.fit(FunctionalKNN, GridFunctionalData(values, grid), labels).candidates == collect(3:2:15)
    @test StatsAPI.fit(FunctionalKNN, GridFunctionalData(values[1:6, :], grid), labels[1:6]).candidates == [3, 5]
end

@testset "classifier representations" begin
    grid = collect(range(0, 1; length=21))
    rng = Random.Xoshiro(84)
    n = 30
    labels = repeat([1, 2], n ÷ 2)
    values = [(labels[i] == 2 ? 0.8 : 0.0) + randn(rng) * 0.3 + (0.5 + 0.2 * randn(rng)) * sin(2pi * t)
        for i in 1:n, t in grid]
    fd = GridFunctionalData(values, grid)

    # Irregular curves: FPCA reconstruction, stored and reused for new curves.
    observed = [sort(randperm(rng, 21)[1:9]) for _ in 1:n]
    irregular = IrregularFunctionalData([grid[k] for k in observed], [values[i, observed[i]] for i in 1:n])
    f = StatsAPI.fit(FunctionalKNN, irregular, labels; k=1, fpca=(; ncomponents=2))
    @test f.settings.representation == [:fpca_reconstruction]
    @test only(f.settings.fpca) isa FPCA && only(f.settings.fpca).method == :pace
    @test predict(f, irregular) == labels       # each training curve is its own nearest neighbour
    @test_throws ArgumentError predict(f, f.curves[1].fpca)
    fpca = only(f.settings.fpca)
    @test StatsAPI.fit(FunctionalKNN, fpca, labels; k=1).fitted_probabilities == f.fitted_probabilities

    # Complete training curves: new curves must be complete on the same grid.
    g = StatsAPI.fit(FunctionalKNN, fd, labels)
    @test g.settings.representation == [:grid]
    @test_throws ArgumentError predict(g, irregular)
    @test_throws ArgumentError predict(g, GridFunctionalData(values[:, 1:20], grid[1:20]))
    masked = Matrix{Union{Missing,Float64}}(values)
    masked[1, 2] = missing
    @test_throws ArgumentError predict(g, GridFunctionalData(masked, grid))
    @test_throws ArgumentError predict(g, MultivariateFunctionalData([fd, fd]))
    @test_throws ArgumentError predict(g, fd; type=:depth)

    # Basis curves are evaluated exactly on the grid.
    basis = FourierBasis((0.0, 1.0), 3)
    coefs = [labels .== 2 randn(rng, n) 0.5 .+ 0.1 .* randn(rng, n)]
    bfd = BasisFunctionalData(coefs, basis)
    b = StatsAPI.fit(FunctionalKNN, bfd, labels; grid)
    @test b.settings.representation == [:basis_evaluated]
    @test predict(b, bfd; type=:probability) ==
        predict(b, GridFunctionalData(evaluate(bfd, grid), grid); type=:probability)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, bfd, labels)

    # Multivariate curves add squared component distances; a zero component
    # changes nothing.
    zero_fd = GridFunctionalData(zeros(n, 21), grid)
    mfd = MultivariateFunctionalData([fd, zero_fd])
    m = StatsAPI.fit(FunctionalKNN, mfd, labels)
    @test m.loo_error ≈ g.loo_error && m.fitted_probabilities ≈ g.fitted_probabilities
    @test m.settings.distance == :euclidean_L2_trapezoidal
    @test predict(m, mfd; type=:probability) ≈ predict(g, fd; type=:probability)
    @test_throws ArgumentError predict(m, fd)
    @test_throws DimensionMismatch predict(m, MultivariateFunctionalData([fd, zero_fd, fd]))
    # Component depths average with weights: two equal components give the
    # univariate band depth.
    d1 = StatsAPI.fit(DepthClassifier, fd, labels; method=ModifiedBandDepth())
    d2 = StatsAPI.fit(DepthClassifier, MultivariateFunctionalData([fd, fd]), labels;
        method=ModifiedBandDepth(), weights=[0.5, 0.5])
    @test d2.depths ≈ d1.depths && StatsAPI.fitted(d2) == StatsAPI.fitted(d1)
    @test predict(d2, MultivariateFunctionalData([fd, fd])) == predict(d1, fd)
    @test_throws ArgumentError StatsAPI.fit(DepthClassifier, fd, labels; weights=[1.0])
end

@testset "depth classifiers" begin
    grid = collect(range(0, 1; length=21))
    rng = Random.Xoshiro(85)
    n = 90
    labels = repeat(["p", "q", "r"], n ÷ 3)
    shift = Dict("p" => 0.0, "q" => 0.3, "r" => 0.6)
    values = [shift[labels[i]] + 0.5 * randn(rng) + (0.5 + 0.2 * randn(rng)) * sin(2pi * t) +
        0.05 * randn(rng) for i in 1:n, t in grid]
    fd = GridFunctionalData(values, grid)

    # Generated projections are drawn once and shared by classes and predictions.
    f = StatsAPI.fit(DepthClassifier, fd, labels; method=RandomProjectionDepth(rng=Random.Xoshiro(1)))
    @test f.method.projections isa Matrix && size(f.method.projections) == (50, 21)
    @test predict(f, fd; type=:depth) == f.depths
    @test predict(f, fd) == StatsAPI.fitted(f)
    @test StatsAPI.fitted(f) == f.classes[[argmax(f.depths[i, :]) for i in 1:n]]
    @test_throws ArgumentError predict(f, fd; type=:probability)
    @test f.settings.class_bandwidths === nothing
    shown = sprint(show, f)
    @test occursin("DepthClassifier(", shown) && occursin("RandomProjectionDepth", shown)
    # Modal depth: one bandwidth per class (classif.depth); DD shares the pooled one.
    mode = StatsAPI.fit(DepthClassifier, fd, labels; method=ModalDepth())
    @test length(mode.settings.class_bandwidths) == 3 && mode.method.bandwidth === nothing
    @test mode.settings.class_bandwidths[1] ≈
        depth(fd[findall(==("p"), labels)], ModalDepth()).settings.bandwidth
    dd_mode = StatsAPI.fit(DDClassifier, fd, labels; method=ModalDepth(), classifier=:lda)
    @test dd_mode.method.bandwidth ≈ depth(fd, ModalDepth()).settings.bandwidth
    fixed = StatsAPI.fit(DDClassifier, fd, labels; method=ModalDepth(bandwidth=0.3), classifier=:qda)
    @test fixed.method.bandwidth == 0.3

    # Several classes: one-against-rest logistic probabilities, normalized.
    dd = StatsAPI.fit(DDClassifier, fd, labels)
    @test size(dd.stage.coefficients) == (4, 3)
    @test vec(sum(dd.fitted_probabilities; dims=2)) ≈ ones(n)
    @test StatsAPI.fitted(dd) == dd.classes[[argmax(dd.fitted_probabilities[i, :]) for i in 1:n]]
    @test predict(dd, fd; type=:probability) ≈ dd.fitted_probabilities
    @test predict(dd, fd; type=:depth) ≈ dd.depths
    @test occursin("classifier=:glm", sprint(show, dd))
    knn = StatsAPI.fit(DDClassifier, fd, labels; classifier=:knn, k=5)
    @test knn.stage.k == 5 && knn.settings.fitted == :leave_one_out
    @test occursin("k=5", sprint(show, knn))
    lda = StatsAPI.fit(DDClassifier, fd, labels; classifier=:lda, prior=:equal)
    @test lda.stage.priors ≈ fill(1 / 3, 3)
    @test occursin("prior=", sprint(show, lda))
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; classifier=:dd1)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; classifier=:DD2)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; k=3)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; prior=:equal)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; classifier=:knn, prior=:equal)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd, labels; classifier=:lda, k=3)
    @test_throws ArgumentError predict(dd, fd; type=:scores)

    # Degree-one DD rule for two classes.
    two = findall(!=("r"), labels)
    dd1 = StatsAPI.fit(DDClassifier, fd[two], labels[two]; classifier=:dd1)
    s = dd1.stage.slope
    D = dd1.depths
    @test StatsAPI.fitted(dd1) == [D[i, 2] > s * D[i, 1] ? "q" : "p" for i in axes(D, 1)]
    @test dd1.fitted_probabilities === nothing
    @test_throws ArgumentError predict(dd1, fd; type=:probability)
    @test predict(dd1, fd[two]) == StatsAPI.fitted(dd1)
    @test occursin("slope=", sprint(show, dd1))
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, fd[two], labels[two]; classifier=:dd1, k=3)
    # The slope minimizes training misclassification, smallest on ties.
    Dsmall = [1.0 0.5; 1.0 0.8; 1.0 1.5; 1.0 2.0]
    @test FunctionalDataAnalysis._dd1_fit(Dsmall, [1, 1, 2, 2]) == 0.8
    @test FunctionalDataAnalysis._dd1_fit([1.0 2.0; 1.0 3.0; 0.0 1.0], [2, 2, 2]) == 0.0

    # Separated depths make the logistic likelihood unbounded.
    apart = GridFunctionalData([Float64(i + 10 * (i > 6)) + 0.1 * sin(2pi * t) for i in 1:12, t in grid], grid)
    @test_throws ArgumentError StatsAPI.fit(DDClassifier, apart, repeat(["p"], 6) |> v -> vcat(v, fill("q", 6)))

    # Every class needs two curves for its depth reference.
    @test_throws ArgumentError StatsAPI.fit(DepthClassifier, fd[1:4], ["p", "q", "r", "r"])
end

@testset "functional discriminant analysis" begin
    # Gaussian posteriors by hand for one feature.
    X = reshape([0.0, 1.0, 2.0, 4.0, 5.0, 6.0], :, 1)
    model = FunctionalDataAnalysis._discriminant_fit(X, [1, 1, 1, 2, 2, 2], 2, false, :equal)
    @test model.means == reshape([1.0, 5.0], :, 1) && only(model.covariances[1]) == 1.0
    x0 = reshape([2.5], 1, 1)
    density(x, m) = exp(-(x - m)^2 / 2)
    @test FunctionalDataAnalysis._discriminant_probabilities(model, x0) ≈
        [density(2.5, 1) density(2.5, 5)] ./ (density(2.5, 1) + density(2.5, 5))
    quadratic = FunctionalDataAnalysis._discriminant_fit(X, [1, 1, 1, 2, 2, 2], 2, true, [0.25, 0.75])
    @test quadratic.covariances == [fill(1.0, 1, 1), fill(1.0, 1, 1)] && quadratic.priors == [0.25, 0.75]
    @test FunctionalDataAnalysis._discriminant_probabilities(quadratic, x0) ≈
        [0.25density(2.5, 1) 0.75density(2.5, 5)] ./ (0.25density(2.5, 1) + 0.75density(2.5, 5))

    grid = collect(range(0, 1; length=25))
    rng = Random.Xoshiro(86)
    n = 40
    labels = repeat(["u", "v"], n ÷ 2)
    values = [(labels[i] == "v" ? 0.6 : 0.0) * sin(pi * t) + randn(rng) * sin(pi * t) +
        0.6 * randn(rng) * sin(2pi * t) + 0.02 * randn(rng) for i in 1:n, t in grid]
    fd = GridFunctionalData(values, grid)
    lda = StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2)
    @test lda.settings.fpca_method == (:face,) && lda.settings.fpca_method_source == (:short_grid_knots,)
    @test lda.settings.prior == :equal && lda.priors == [0.5, 0.5]
    @test predict(lda, fd; type=:probability) ≈ lda.probabilities
    # Linear rule: the log-odds is affine in the scores.
    Z = scores(lda.fpca[1])[:, 1:2]
    logodds = log.(lda.probabilities[:, 2] ./ lda.probabilities[:, 1])
    design = [ones(n) Z]
    @test design * (design \ logodds) ≈ logodds
    proportional = StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2, prior=[0.2, 0.8])
    @test proportional.settings.prior == :supplied
    expected = lda.probabilities .* [0.2 0.8]
    @test proportional.probabilities ≈ expected ./ sum(expected; dims=2)
    qda = StatsAPI.fit(FunctionalQDA, fd, labels; ncomponents=2, prior=:proportional)
    @test length(qda.covariances) == 2 && qda.priors == [0.5, 0.5]
    shown = sprint(show, qda)
    @test occursin("FunctionalQDA(", shown) && occursin("ncomponents=[2]", shown)
    @test occursin("FunctionalLDA(", sprint(show, MIME"text/plain"(), lda))
    supplied = StatsAPI.fit(FunctionalLDA, fd, labels; fpca=lda.fpca[1], ncomponents=2)
    @test supplied.probabilities ≈ lda.probabilities && supplied.settings.fpca_source == (:supplied,)

    # Sparse covariates use PACE scores.
    observed = [sort(randperm(rng, 25)[1:8]) for _ in 1:n]
    irregular = IrregularFunctionalData([grid[k] for k in observed], [values[i, observed[i]] for i in 1:n])
    sparse = StatsAPI.fit(FunctionalLDA, irregular, labels; ncomponents=2)
    @test sparse.settings.fpca_method == (:pace,) && sparse.settings.fpca_method_source == (:fpca_default,)
    @test predict(sparse, irregular; type=:probability) ≈ sparse.probabilities
    @test predict(sparse, irregular[1:5]) == StatsAPI.fitted(sparse)[1:5]
    # Several covariates concatenate their scores.
    both = StatsAPI.fit(FunctionalLDA, (fd, irregular), labels; ncomponents=(2, 1))
    @test both.counts == [2, 1] && size(both.means) == (2, 3)
    @test predict(both, (fd, irregular); type=:probability) ≈ both.probabilities
    @test_throws DimensionMismatch predict(both, fd)

    @test_throws ArgumentError StatsAPI.fit(FunctionalQDA, fd[1:6], labels[1:6]; ncomponents=3,
        fpca_options=(; method=:dense))
    @test_throws ArgumentError StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2, prior=:uniform)
    @test_throws DimensionMismatch StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2, prior=[1.0])
    @test_throws ArgumentError StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2, prior=[0.5, 0.6])
    @test_throws ArgumentError StatsAPI.fit(FunctionalLDA, fd, labels; ncomponents=2, prior=[0.0, 1.0])
    @test_throws ArgumentError predict(lda, fd; type=:depth)
    @test_throws ArgumentError StatsAPI.fit(FunctionalLDA, fd, labels; fpca_options=(; kernel=nothing))
end

@testset "classification label validation" begin
    train = _classif_constant([0.0, 1.0, 2.0, 3.0, 4.0])
    @test_throws DimensionMismatch StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2]; k=1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, fill(1, 5); k=1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, missing, 1, 2]; k=1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1.0, 2.0, NaN, 1.0, 2.0]; k=1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, Any[1, "a", 1, "a", 1]; k=1)
    @test StatsAPI.fit(FunctionalKNN, train, 5:-1:1; k=1).classes == collect(1:5)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, (1, 2, 1, 2, 1); k=1)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2, 1]; k=5)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2, 1]; k=0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2, 1]; k=true)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2, 1]; k=Int[])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train, [1, 2, 1, 2, 1]; k=2.0)
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, train[1:3], [1, 2, 1])
    @test_throws ArgumentError StatsAPI.fit(FunctionalKNN, 1.0, [1, 2])
    f = StatsAPI.fit(FunctionalKNN, train, [2, 1, 2, 1, 2]; k=1)
    @test f.classes == [1, 2]
    @test_throws ArgumentError predict(f, train; type=:class)
end
