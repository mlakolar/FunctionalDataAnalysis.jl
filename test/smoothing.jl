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

@testset "Complete-grid block smoothing and bounded cache" begin
    FDA = FunctionalDataAnalysis
    for T in (Float32,Float64), kernel in (KernelSmoothing.EpanechnikovKernel(),KernelSmoothing.GaussianKernel()), nu in 0:2
        grid = collect(range(T(0),T(1);length=41))
        values = T[sin(i*t)+i*t^2 for i in 1:4,t in grid]
        fd = GridFunctionalData(@view(values[:,:]),grid;ids=[:a,:b,:c,:d],metadata=(;source=:input))
        method = LocalPolynomial(;bandwidth=T(.4),derivative=nu,kernel)
        block = @inferred smooth_curves(fd,method;grid=@view(grid[:]))
        rowwise = @inferred smooth_curves(to_irregular(fd),method;grid)
        @test eltype(block) === T
        @test block.observed === nothing
        @test block.grid == fd.grid
        @test block.weights == fd.weights
        @test block.values ≈ rowwise.values rtol=0 atol=(T===Float32 ? 1e-5 : 1e-10)
        @test block.metadata == rowwise.metadata
        @test block.metadata.grid !== grid
        @test block.grid !== fd.grid && block.values !== fd.values
        @test ids(block) == ids(fd) && ids(block) !== ids(fd)
        @test fd.metadata == (;source=:input)
        saved = copy(fd.values)
        block.values[1,1] += one(T)
        @test fd.values == saved
    end

    grid = collect(range(0.,1.;length=41))
    fd = GridFunctionalData([sin(i*t)+i*t^2 for i in 1:4,t in grid],grid;ids=1:4)
    method = LocalPolynomial(;bandwidth=.4,derivative=1)
    block = smooth_curves(fd,method;grid)
    extended_grid = vcat(-.1,grid,1.1)
    extended = smooth_curves(fd,method;grid=extended_grid)
    @test extended.values[:,2:end-1] ≈ block.values rtol=0 atol=1e-10
    @test all(.!extended.observed[:,[1,end]])
    @test all(extended.observed[:,2:end-1])
    @test all(iszero,extended.values[:,[1,end]])
    @test extended.metadata.grid == extended_grid
    @test ids(extended) == ids(fd)
    masked_values = Matrix{Union{Missing,Float64}}(fd.values)
    masked_values[2,21] = missing
    masked = GridFunctionalData(masked_values,grid;ids=ids(fd))
    frommask = smooth_curves(masked,method;grid)
    fromirregular = smooth_curves(to_irregular(masked),method;grid)
    @test frommask.values == fromirregular.values
    @test frommask.observed === fromirregular.observed === nothing
    @test frommask.metadata == fromirregular.metadata
    @test ids(frommask) == ids(fd)

    # The first curve is valid, while the second has a finite Float64
    # derivative that overflows the Float32 output. Diagnose the same curve
    # and point as the rowwise path after passing the preceding estimates.
    overflow_grid = collect(range(0f0,.5f0;length=11))
    overflow_values = vcat(zeros(Float32,1,11),
        reshape(Float32.(4e38 .* Float64.(overflow_grid)),1,:))
    overflow_fd = GridFunctionalData(overflow_values,overflow_grid)
    overflow_method = LocalPolynomial(;bandwidth=.5f0,derivative=1)
    errors = map((overflow_fd,to_irregular(overflow_fd))) do input
        try
            smooth_curves(input,overflow_method;grid=overflow_grid)
        catch err
            err
        end
    end
    @test all(err -> err isa ArgumentError,errors)
    @test all(err -> occursin("curve 2",sprint(showerror,err)),errors)
    @test sprint(showerror,errors[1]) == sprint(showerror,errors[2])

    # More distinct grids than the bound, followed by a value-equal cache hit.
    cache = Dict{Vector{Float32},Tuple{Vector{Int},Matrix{Float64}}}()
    insertion_order = Vector{Float32}[]
    points = collect(range(0f0,1f0;length=31))
    method32 = LocalPolynomial(;bandwidth=.5f0,kernel=KernelSmoothing.GaussianKernel())
    first_grid = collect(range(0f0,1f0;length=21))
    last_grid = first_grid
    for j in 1:(FDA._CURVE_SMOOTHER_CACHE_LIMIT+3)
        last_grid = collect(range(0f0,1f0;length=20+j))
        indices,matrix = @inferred FDA._cached_curve_smoother!(cache,insertion_order,last_grid,points,method32)
        @test indices == eachindex(points)
        @test matrix isa Matrix{Float64}
        @test length(cache) <= FDA._CURVE_SMOOTHER_CACHE_LIMIT
        @test length(insertion_order) == length(cache)
    end
    @test !haskey(cache,first_grid)
    @test length(cache) == FDA._CURVE_SMOOTHER_CACHE_LIMIT
    previous = cache[last_grid]
    before_order = copy(insertion_order)
    hit = FDA._cached_curve_smoother!(cache,insertion_order,copy(last_grid),points,method32)
    @test hit[2] === previous[2]
    @test insertion_order == before_order
end
