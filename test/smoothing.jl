import KernelSmoothing
import StatsAPI

@testset "Per-curve smoothing" begin
    t = collect(range(0.0, 1.0; length=11))
    values = permutedims(hcat(1 .+ 2t, -2 .+ 3t))
    fd = GridFunctionalData(values, t; ids=[:a, :b])
    grid = collect(range(-0.1, 1.1; length=25))
    method = @inferred LocalPolynomial(bandwidth=0.4)
    result = @inferred smooth_curves(fd, method; grid)
    inside = findall(x -> 0 <= x <= 1, grid)
    @test result.values[1, inside] ≈ 1 .+ 2grid[inside]
    @test result.values[2, inside] ≈ -2 .+ 3grid[inside]
    @test all(.!result.observed[:, [1, 2, 24, 25]])
    @test ids(result) == [:a, :b]
    @test ids(result) !== ids(fd)
    @test result.metadata.bandwidth == 0.4
    @test result.metadata.minobs == 0
    @test result[[1]].metadata == result.metadata
    @test result[[1]].metadata.grid !== result.metadata.grid
    @test occursin("bandwidth", sprint(show, result))
    @test occursin("degree", sprint(show, method))
    @test StatsAPI.fit(LocalPolynomial, fd; bandwidth=0.4, grid=t).values ≈ values

    irregular = IrregularFunctionalData([t, t[3:9]], [values[1, :], values[2, 3:9]];
        ids=[:a, :b])
    ir = smooth_curves(irregular, method; grid=t)
    @test ir.values[1, :] ≈ values[1, :]
    @test ir.values[2, 3:9] ≈ values[2, 3:9]
    @test all(.!ir.observed[2, [1, 2, 10, 11]])
    @test ids(ir) == [:a, :b]
    k = KernelSmoothing.GaussianKernel()
    nonlinear = GridFunctionalData(reshape(sin.(t), 1, :), t)
    direct = KernelSmoothing.fit(KernelSmoothing.LocalPolynomialRegression, t, sin.(t);
        bandwidth=0.3, degree=2, kernel=k, points=t, weights=ones(length(t)), minobs=0)
    actual = smooth_curves(nonlinear, LocalPolynomial(bandwidth=0.3, degree=2, kernel=k); grid=t)
    @test vec(actual.values) ≈ KernelSmoothing.coef(direct)

    masked = GridFunctionalData(Union{Missing,Float64}[1 2 missing 4 5], collect(0.0:4.0))
    @test smooth_curves(masked, LocalPolynomial(bandwidth=3.0); grid=collect(0.0:4.0)).values ≈
        reshape(collect(1.0:5.0), 1, :)
    for h in (0, -1, NaN, Inf, [0.2])
        @test_throws ArgumentError LocalPolynomial(bandwidth=h)
    end
    @test_throws ArgumentError LocalPolynomial(bandwidth=1, degree=-1)
    @test_throws ArgumentError LocalPolynomial(bandwidth=1, degree=1.5)
    @test_throws ArgumentError LocalPolynomial(bandwidth=1, kernel=:bad)
    @test_throws ArgumentError smooth_curves(fd, method; grid=[0.5, 0.0])
    @test_throws ArgumentError smooth_curves(fd, method; grid=[0.0, Inf])
    @test_throws ArgumentError smooth_curves(fd, method; grid=[0.0])
    err = try
        smooth_curves(fd, LocalPolynomial(bandwidth=0.001); grid=t)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("curve 1", sprint(showerror, err))
    @test occursin("point 0.0", sprint(showerror, err))
    @test occursin("larger bandwidth", sprint(showerror, err))
    emptycurve = GridFunctionalData(fill(missing, 1, 11), t)
    @test_throws ArgumentError smooth_curves(emptycurve, method; grid=t)
    singleton = IrregularFunctionalData([[0.5]], [[2.0]])
    @test_throws ArgumentError smooth_curves(singleton, method; grid=[0.5, 0.6])
    @test all(.!smooth_curves(singleton, method; grid=[0.0, 0.1]).observed)
    @test smooth_curves(singleton, LocalPolynomial(bandwidth=0.2, degree=0);
        grid=[0.5, 0.6]).values[1, 1] ≈ 2

    basis = MonomialBasis((0.0, 1.0), 2)
    spline = @inferred PenalizedSpline(basis; lambda=0.0)
    fitted = @inferred smooth_curves(fd, spline)
    @test fitted.coefs ≈ [1 2 0; -2 3 0] atol=1e-12
    @test ids(fitted) == [:a, :b]
    @test fitted.metadata.lambda == 0
    @test fitted[1].metadata == fitted.metadata
    @test fitted[[2]].metadata == fitted.metadata
    @test occursin("lambda", sprint(show, fitted))
    @test StatsAPI.fit(PenalizedSpline, fd; basis, lambda=0).coefs ≈ fitted.coefs
    @test smooth_curves(irregular, spline).coefs ≈ fitted.coefs atol=1e-12

    # Analytic GCV for orthogonal columns [1, t] at t=-1,0,1,
    # order zero: XtX = diag(3,2), y = [1,0,1].
    raw = GridFunctionalData([1.0 0 1; 2 0 2], [-1.0, 0, 1]; ids=[:x, :y])
    b = MonomialBasis((-1.0, 1.0), 1)
    # Integrated order-zero penalty is diag(2, 2/3).
    candidates = [0.0, 0.1, 1.0, 10.0]
    selector = @inferred GCV(candidates)
    candidates[1] = 5
    @test selector.lambdas[1] == 0
    gcv = smooth_curves(raw, PenalizedSpline(b; lambda=selector, order=0))
    analytic = map(selector.lambdas) do l
        a = 3 + 2l
        df = 3/a + 2/(2 + 2l/3)
        5 * 3 * (2 - 8/a + 12/a^2) / (3 - df)^2
    end
    @test gcv.metadata.selection.scores ≈ analytic
    @test gcv.metadata.lambda == selector.lambdas[argmin(analytic)]
    @test gcv.coefs ≈ to_basis(raw, b; lambda=gcv.metadata.lambda, order=0).coefs
    @test ids(gcv) == [:x, :y]
    @test gcv.metadata.selection.lambdas !== selector.lambdas

    # Unequal curve lengths and an interior optimum in an unsorted candidate
    # grid. Form H explicitly, independently of the selector's df computation.
    # For [1, t, t^2] on [-1, 1], the order-2 penalty is diag(0, 0, 8).
    times = [collect(range(-1.0, 1.0; length=9)),
        [-1.0, -0.8, -0.3, 0.0, 0.2, 0.6, 1.0]]
    noisy = [1 .+ t .+ 2t.^2 .+ 0.8 .* (-1.0).^(1:length(t)) for t in times]
    unequal = IrregularFunctionalData(times, noisy; ids=[:long, :short])
    quadratic = MonomialBasis((-1.0, 1.0), 2)
    unordered = [1.0, 0.0, 10.0, 0.1, 0.001, 0.01]
    explicit_scores = map(unordered) do lambda
        sum(zip(times, noisy)) do (t, y)
            n = length(t)
            X = hcat(ones(n), t, t.^2)
            H = X * ((transpose(X) * X + lambda * Diagonal([0.0, 0.0, 8.0])) \ transpose(X))
            n * sum(abs2, y - H * y) / (n - tr(H))^2
        end
    end
    best = findfirst(==(0.1), unordered)
    @test explicit_scores[best] < minimum(explicit_scores[eachindex(unordered) .!= best])
    @test minimum(unordered) < unordered[best] < maximum(unordered)
    @test best != firstindex(unordered) && best != lastindex(unordered)
    mixed_gcv = smooth_curves(unequal,
        PenalizedSpline(quadratic; lambda=GCV(unordered), order=2))
    @test mixed_gcv.metadata.selection.lambdas == unordered
    @test mixed_gcv.metadata.selection.scores ≈ explicit_scores
    @test mixed_gcv.metadata.lambda == 0.1
    @test mixed_gcv.coefs ≈ to_basis(unequal, quadratic; lambda=0.1, order=2).coefs
    @test ids(mixed_gcv) == [:long, :short]

    # Metadata are provenance of the original fit, even after subsetting.
    for smoothed in (result, mixed_gcv)
        left, right = smoothed[[1]], smoothed[[2]]
        joined = vcat(left, right)
        @test joined.metadata == smoothed.metadata
        @test ids(joined) == ids(smoothed)
        if smoothed isa GridFunctionalData
            @test joined.metadata.grid !== left.metadata.grid
            changed = GridFunctionalData(right.values, right.grid; ids=ids(right),
                metadata=(; bandwidth=0.9))
            @test isempty(to_grid(smoothed, t, PiecewiseLinear()).metadata)
            @test isempty(to_basis(smoothed, basis).metadata)
        else
            @test left.metadata.selection.scores ≈ explicit_scores
            @test joined.metadata.selection.scores !== left.metadata.selection.scores
            changed = BasisFunctionalData(right.coefs, right.basis; ids=ids(right),
                metadata=(; lambda=0.9))
            @test isempty(to_grid(smoothed, [-1.0, 0.0, 1.0]).metadata)
        end
        @test_throws ArgumentError vcat(left, changed)

        # Multivariate wrapping and concatenation preserve component provenance,
        # both when labels must be reconstructed and when inputs are unlabeled.
        for labeled in (true, false)
            source = if labeled
                smoothed
            elseif smoothed isa GridFunctionalData
                GridFunctionalData(FunctionalDataAnalysis._grid_matrix(smoothed), smoothed.grid;
                    metadata=smoothed.metadata)
            else
                BasisFunctionalData(smoothed.coefs, smoothed.basis; metadata=smoothed.metadata)
            end
            different = if labeled
                changed
            elseif changed isa GridFunctionalData
                GridFunctionalData(changed.values, changed.grid; metadata=changed.metadata)
            else
                BasisFunctionalData(changed.coefs, changed.basis; metadata=changed.metadata)
            end
            wrapped = MultivariateFunctionalData([source])
            @test component(wrapped, 1).metadata == smoothed.metadata
            @test component(wrapped[[1]], 1).metadata == smoothed.metadata
            combined = vcat(wrapped[[1]], wrapped[[2]])
            stored = component(combined, 1)
            @test stored.metadata == smoothed.metadata
            @test ids(combined) == ids(source)
            if stored isa GridFunctionalData
                @test component(wrapped, 1).metadata.grid !== smoothed.metadata.grid
                @test stored.metadata.grid !== component(wrapped, 1).metadata.grid
            else
                @test component(wrapped, 1).metadata.selection.scores !== smoothed.metadata.selection.scores
                @test stored.metadata.selection.scores !== component(wrapped, 1).metadata.selection.scores
            end
            @test_throws ArgumentError vcat(wrapped[[1]], MultivariateFunctionalData([different]))
        end
    end
    saturated = GridFunctionalData([0.0 1 0], [-1.0, 0, 1])
    selected = smooth_curves(saturated,
        PenalizedSpline(MonomialBasis((-1.0, 1.0), 2); lambda=GCV([0.0, 1.0])))
    @test selected.metadata.selection.scores[1] == Inf
    @test selected.metadata.lambda == 1
    @test_throws ArgumentError smooth_curves(saturated,
        PenalizedSpline(MonomialBasis((-1.0, 1.0), 2); lambda=GCV([0.0])))
    for ls in (Float64[], [-1.0], [NaN], [Inf], [1.0, 1.0])
        @test_throws ArgumentError GCV(ls)
    end
    for l in (-1, NaN, Inf, :bad)
        @test_throws ArgumentError PenalizedSpline(b; lambda=l)
    end
    @test_throws ArgumentError PenalizedSpline(b; lambda=1, order=-1)
    @test_throws ArgumentError smooth_curves(emptycurve, PenalizedSpline(basis; lambda=1))
    @test_throws ArgumentError smooth_curves(fd, PenalizedSpline(MonomialBasis((2.0, 3.0), 1); lambda=1))
    @test_throws ArgumentError smooth_curves(singleton,
        PenalizedSpline(MonomialBasis((0.0, 1.0), 2); lambda=0))
end
