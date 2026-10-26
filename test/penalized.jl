import StatsAPI
using DelimitedFiles

include(joinpath(@__DIR__, "reference", "mgcv_m7_inputs.jl"))

# Dense textbook formulas in the working precision of the inputs; used to
# check the solver independently of its reparameterized factorizations.
# Generic eigenvalues are unavailable for BigFloat, so that path supplies the
# positive penalty eigenvalues.
function naive_pls(X, y, w, penalties, columns, lambda, criterion; positive=nothing)
    n, p = size(X)
    S = zeros(eltype(X), p, p)
    for (P, c, l) in zip(penalties, columns, lambda)
        S[c, c] .+= l .* P
    end
    XtWX = transpose(X) * (w .* X)
    H = XtWX + S
    beta = H \ (transpose(X) * (w .* y))
    r = y - X * beta
    rss = sum(w .* r .^ 2)
    tau = tr(H \ XtWX)
    if positive === nothing
        total = zeros(p, p)
        for (P, c) in zip(penalties, columns)
            total[c, c] .+= P ./ norm(P)
        end
        values = eigvals(Symmetric(total))
        rank = count(>(1e-10 * maximum(values)), values)
        positive = sort(eigvals(Symmetric(S)); rev=true)[1:rank]
    end
    Mp = p - length(positive)
    if criterion === :REML
        Dp = rss + dot(beta, S * beta)
        phi = Dp / (n - Mp)
        value = Dp / (2phi) + (n - Mp) / 2 * log(2 * oftype(phi, pi) * phi) + logdet(H) / 2 -
            sum(log, positive) / 2 - sum(log, w) / 2
    else
        value = n * rss / (n - tau)^2
    end
    (; beta, value, rss, tau, sigma2=rss / (n - tau), H, XtWX)
end

@testset "Penalized least squares against mgcv 1.9-3" begin
    inputs = m7_pls_inputs()
    stored = readdlm(joinpath(@__DIR__, "data", "m7_pls_input.csv"), ',', Any; comments=true)
    for name in keys(inputs)
        value = inputs[name]
        matrix = value isa AbstractVector ? reshape(value, :, 1) : value
        dims = stored[stored[:, 1] .== "dim_$name", :]
        rebuilt = zeros(Int(dims[1, 2]), Int(dims[1, 3]))
        for row in eachrow(stored[stored[:, 1] .== String(name), :])
            rebuilt[Int(row[2]), Int(row[3])] = Float64(row[4])
        end
        # The fixture is exactly the package's own bases and penalties.
        @test rebuilt == matrix
    end

    golden = readdlm(joinpath(@__DIR__, "data", "m7_mgcv.csv"), ',', Any; comments=true)
    function reference(case, method, quantity)
        rows = golden[(golden[:, 1] .== case) .& (golden[:, 2] .== method) .&
            (golden[:, 3] .== quantity), :]
        nrow, ncol = Int(maximum(rows[:, 4])), Int(maximum(rows[:, 5]))
        out = zeros(nrow, ncol)
        for row in eachrow(rows)
            out[Int(row[4]), Int(row[5])] = Float64(row[6])
        end
        ncol == 1 ? vec(out) : out
    end
    relative_error(a, b) = norm(a - b) / norm(b)

    z, B, SB = inputs.z, inputs.spline_design, inputs.spline_penalty
    Fd, SF = inputs.fourier_design, inputs.fourier_penalty
    designs = Dict(
        "single" => (hcat(z, B), [SB => 2:11]),
        "weighted" => (hcat(z, B), [SB => 2:11]),
        "lower" => (hcat(z, B), [SB => 2:11]),
        "disjoint" => (hcat(B, Fd), [SB => 1:10, SF => 11:16]),
        "bound" => (hcat(B, Fd), [SB => 1:10, SF => 11:16]),
        "tensor" => (inputs.tensor_design,
            [inputs.tensor_penalty1 => 1:25, inputs.tensor_penalty2 => 1:25]))
    for case in ("single", "weighted", "disjoint", "tensor", "bound", "lower"),
            (method, criterion) in (("REML", :REML), ("GCV.Cp", :GCV))
        @testset "$case $method" begin
            X, penalties = designs[case]
            y = reference(case, "input", "y")
            # R's strtod may round the 17-digit input in the last bit; use
            # the weights mgcv actually received.
            weights = case == "weighted" ? reference("all", "input", "weights_weighted") : nothing
            @test case != "weighted" || isapprox(weights, inputs.w; rtol=4eps())
            lambda = reference(case, method, "lambda")
            # Compare at the smoothing parameters mgcv selected.
            at_mgcv = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties, weights, criterion, lambda)
            @test at_mgcv.lambda == lambda
            @test all(==(:fixed), at_mgcv.lambda_status)
            @test at_mgcv.settings.convergence === :fixed
            @test relative_error(coef(at_mgcv), reference(case, method, "coef")) < 1e-8
            @test relative_error(fitted(at_mgcv), reference(case, method, "fitted")) < 1e-8
            @test relative_error(at_mgcv.edf, reference(case, method, "edf")) < 1e-8
            @test abs(at_mgcv.edf_total / sum(reference(case, method, "edf")) - 1) < 1e-8
            # Per-penalty EDF: mgcv's edf summed over each penalty's columns;
            # penalties on one tensor block each report that block once.
            @test relative_error(at_mgcv.edf_penalty,
                [sum(reference(case, method, "edf")[last(P)]) for P in penalties]) < 1e-8
            unpenalized = setdiff(1:size(X, 2), reduce(vcat, collect.(last.(penalties))))
            blocks = unique(collect.(last.(penalties)))
            @test sum(sum(at_mgcv.edf[b]) for b in blocks) + sum(at_mgcv.edf[unpenalized]) ≈
                at_mgcv.edf_total rtol=1e-12
            @test abs(at_mgcv.sigma2 / only(reference(case, method, "sig2")) - 1) < 1e-8
            @test abs(at_mgcv.criterion_value / only(reference(case, method, "score")) - 1) < 1e-8
            Vp = reference(case, method, "Vp")
            upper = triu(trues(size(Vp)))
            @test relative_error(vcov(at_mgcv)[upper], Vp[upper]) < 1e-8

            # Our optimizer reaches a criterion no worse than mgcv's.
            lower = case == "lower" ? reference(case, method, "min_sp") : nothing
            ours = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties, weights, criterion, lower)
            @test ours.settings.convergence === :gradient
            @test ours.criterion_value <= at_mgcv.criterion_value +
                8eps() * (1 + abs(at_mgcv.criterion_value))
            if case == "lower"
                @test ours.lambda == lower
                @test ours.lambda_status == [:lower_bound]
            elseif case == "bound"
                @test ours.lambda_status == [:effectively_infinite, :estimated]
                @test ours.lambda[1] >= lambda[1] / 10
            else
                @test all(==(:estimated), ours.lambda_status)
                # mgcv stops at a gradient tolerance of 1e-6, so its smoothing
                # parameters agree only loosely; criteria are compared above.
                @test relative_error(ours.lambda, lambda) < 1e-2
            end
        end
    end
end

@testset "Penalized least squares: algebra, derivatives and stability" begin
    inputs = m7_pls_inputs()
    X = inputs.tensor_design
    penalties = [inputs.tensor_penalty1, inputs.tensor_penalty2]
    columns = [1:25, 1:25]
    y = [sin(2pi * a) * (1 + b) + 0.1 * cos(7i) for (i, (a, b)) in enumerate(zip(inputs.x1, inputs.x2))]
    w = inputs.w
    for criterion in (:REML, :GCV), lambda in ([1e-3, 1e-1], [2e-5, 3.0])
        model = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns,
            weights=w, criterion, lambda)
        reference = naive_pls(X, y, w, penalties, columns, lambda, criterion)
        @test model.criterion_value ≈ reference.value rtol=1e-10
        @test coef(model) ≈ reference.beta rtol=1e-10
        @test model.edf_total ≈ reference.tau rtol=1e-10
        @test model.sigma2 ≈ reference.sigma2 rtol=1e-10
        @test vcov(model) ≈ inv(reference.H) * reference.sigma2 rtol=1e-10
        @test vcov(model; type=:frequentist) ≈
            (reference.H \ reference.XtWX) * inv(reference.H) * reference.sigma2 rtol=1e-10
        @test model.edf ≈ diag(reference.H \ reference.XtWX) rtol=1e-10
        @test model.edf_penalty ≈ [model.edf_total, model.edf_total]
        @test model.null_space_dimension == 4
        @test fitted(model) + residuals(model) ≈ y
        @test model.reml_scale ≈ (sum(w .* residuals(model).^2) +
            sum(lambda[j] * dot(coef(model), penalties[j] * coef(model)) for j in 1:2)) / (80 - 4)

        # Exact derivatives on log(lambda) against central differences.
        problem_value(rho) = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns,
            weights=w, criterion, lambda=exp.(rho)).criterion_value
        gradient(rho) = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns,
            weights=w, criterion, lambda=exp.(rho)).settings.gradient
        rho = log.(lambda)
        h = 1e-4
        steps = [h .* (1:2 .== j) for j in 1:2]
        numeric_gradient = [(problem_value(rho + s) - problem_value(rho - s)) / 2h for s in steps]
        numeric_hessian = reduce(hcat, [(gradient(rho + s) - gradient(rho - s)) / 2h for s in steps])
        scale = criterion === :REML ? 1.0 : model.criterion_value
        @test norm(model.settings.gradient - numeric_gradient) <= 1e-6 * scale
        @test norm(model.settings.hessian - numeric_hessian) <= 1e-6 * scale
    end

    # Overlapping penalties whose smoothing parameters differ by 1e20: the
    # stable transform agrees with 256-bit textbook formulas.
    # Tensor penalties P⊗I and I⊗P have eigenvalues λ₁μᵢ + λ₂μⱼ, with μ those
    # of the marginal penalty (two exact zeros: its linear null space).
    # The marginal null space is exact in 256 bits: constants and the Greville
    # abscissae (B-spline coefficients of t) for knots 0,0,0,0,1/2,1,1,1,1.
    lambda = [1e-8, 1e12]
    margin = BSplineBasis(4, (0.0, 1.0), 5)
    mu = eigvals(Symmetric(penalty(margin, 2)))
    mu[1:2] .= 0
    big_reference = setprecision(BigFloat, 256) do
        N = hcat(ones(BigFloat, 5), big.([0 // 1, 1 // 6, 1 // 2, 5 // 6, 1 // 1]))
        projection = I - N * ((transpose(N) * N) \ transpose(N))
        P = projection * big.(penalty(margin, 2)) * projection
        identity5 = Matrix{BigFloat}(I, 5, 5)
        sums = [big(lambda[1]) * a + big(lambda[2]) * b for a in mu for b in mu]
        naive_pls(big.(X), big.(y), big.(w), [kron(P, identity5), kron(identity5, P)], columns,
            big.(lambda), :REML; positive=filter(>(0), sums))
    end
    stable = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns, weights=w, lambda)
    @test stable.criterion_value ≈ Float64(big_reference.value) rtol=1e-10
    @test coef(stable) ≈ Float64.(big_reference.beta) rtol=1e-8
    @test stable.edf_total ≈ Float64(big_reference.tau) rtol=1e-8

    # Analytic one-penalty case: X = I and S = I give beta = y/(1+lambda).
    yy = [1.0, -2.0, 0.5, 3.0]
    ridge = StatsAPI.fit(PenalizedLeastSquares, Matrix(1.0I, 4, 4), yy; penalties=[Matrix(1.0I, 4, 4)],
        lambda=[3.0], criterion=:GCV)
    @test coef(ridge) ≈ yy ./ 4
    @test ridge.edf_total ≈ 1
    @test ridge.criterion_value ≈ 4 * sum(abs2, yy .* (3 / 4)) / 3^2
    @test StatsAPI.predict(ridge, [1.0 0 0 0]) ≈ [yy[1] / 4]
    @test StatsAPI.nobs(ridge) == 4
    @test StatsAPI.dof(ridge) ≈ 2
    @test StatsAPI.dof_residual(ridge) ≈ 3
    @test occursin("criterion=GCV", sprint(show, ridge))

    # A binding user upper bound, and an inferred concrete result.
    data_bound = StatsAPI.fit(PenalizedLeastSquares, hcat(inputs.z, inputs.spline_design),
        sin.(2pi .* inputs.x1) .+ 0.2 .* cos.(5 .* (1:80)); penalties=[inputs.spline_penalty => 2:11])
    capped = StatsAPI.fit(PenalizedLeastSquares, hcat(inputs.z, inputs.spline_design),
        sin.(2pi .* inputs.x1) .+ 0.2 .* cos.(5 .* (1:80));
        penalties=[inputs.spline_penalty => 2:11], upper=[data_bound.lambda[1] / 100])
    @test capped.lambda_status == [:upper_bound]
    @test capped.lambda == [data_bound.lambda[1] / 100]
    @test capped.criterion_value > data_bound.criterion_value
    inferred = @inferred StatsAPI.fit(PenalizedLeastSquares, X, y;
        penalties=penalties .=> columns, criterion=:GCV)
    @test inferred isa PenalizedLeastSquares{Float64}
    @test eltype(coef(StatsAPI.fit(PenalizedLeastSquares, Float32.(X), Float32.(y);
        penalties=[Float32.(P) => 1:25 for P in penalties]))) == Float32
    @test StatsAPI.vcov(inferred) == inferred.vcov_bayesian

    # A penalty fixed at zero drops out of the null space count.
    partial = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns,
        lower=[0.0, 0.5], upper=[Inf, 0.5])
    @test partial.lambda_status[2] === :fixed
    @test partial.lambda[2] == 0.5
    zeroed = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=penalties .=> columns, lambda=[1e-3, 0.0])
    @test zeroed.null_space_dimension == 10
end

@testset "Penalized least squares: input validation" begin
    X = [1.0 0 0; 0 1 0; 0 0 1; 1 1 1]
    y = [1.0, 2.0, 3.0, 4.0]
    S = [1.0 -1 0; -1 2 -1; 0 -1 1]
    good = (; penalties=[S])
    @test StatsAPI.fit(PenalizedLeastSquares, X, y; good...) isa PenalizedLeastSquares
    @test_throws DimensionMismatch StatsAPI.fit(PenalizedLeastSquares, X, y[1:3]; good...)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, [1.0, NaN, 3, 4]; good...)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, [X[1:3, :]; Inf 0 0], y; good...)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[S + [0 1 0; 0 0 0; 0 0 0]])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[-S])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[zeros(3, 3)])
    @test_throws DimensionMismatch StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[S => 1:2])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[[1.0;;] => [4]])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[[1.0 0; 0 1] => [1, 1]])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; penalties=[1.0])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., weights=[1.0, 0, 1, 1])
    @test_throws DimensionMismatch StatsAPI.fit(PenalizedLeastSquares, X, y; good..., weights=[1.0, 1])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., criterion=:ML)
    @test_throws DimensionMismatch StatsAPI.fit(PenalizedLeastSquares, X, y; good..., lambda=[1.0, 2.0])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., lambda=[-1.0])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., lambda=[1.0], lower=[0.5])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., lower=[2.0], upper=[1.0])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., lower=[Inf])
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., tol=0)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., rank_tol=2)
    # The constant null-space direction is not identified by a zero column.
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, [X[:, 1:2] zeros(4)], y;
        penalties=[[1.0;;] => [1]])
    # Noise-free responses in the penalty null space leave no scale to estimate.
    t = collect(range(0.0, 1.0; length=21))
    basis = BSplineBasis(4, (0.0, 1.0), 6)
    for criterion in (:REML, :GCV)
        @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, basis_matrix(basis, t), 1 .+ t;
            penalties=[penalty(basis, 2)], criterion)
    end
    @test_throws ArgumentError smooth_curves(GridFunctionalData(permutedims(hcat(1 .+ t, 2 .- t)), t),
        PenalizedSpline(basis; lambda=REML()))
    @test coef(StatsAPI.fit(PenalizedLeastSquares, basis_matrix(basis, t), 1 .+ t;
        penalties=[penalty(basis, 2)], lambda=[1.0])) ≈ to_basis(GridFunctionalData(
        reshape(1 .+ t, 1, :), t), basis; lambda=1.0).coefs[1, :]
    # With fixed lambda an undefined criterion is reported as NaN, not an error.
    B, P = basis_matrix(basis, t), penalty(basis, 2)
    flat = StatsAPI.fit(PenalizedLeastSquares, B, zeros(length(t)); penalties=[P], lambda=[1.0])
    @test isnan(flat.criterion_value) && !flat.settings.criterion_defined
    @test flat.lambda_status == [:fixed] && flat.settings.convergence === :fixed
    @test all(iszero, coef(flat)) && flat.sigma2 == 0 && flat.reml_scale == 0
    @test flat.edf_total ≈ tr((B' * B + P) \ (B' * B))
    @test occursin("criterion_defined = false", sprint(show, flat))
    @test StatsAPI.fit(PenalizedLeastSquares, B, zeros(length(t)); penalties=[P], lambda=[1.0],
        criterion=:GCV).settings.criterion_defined
    # n = Mₚ = 2: the null space interpolates both points, so n - τ = 0.
    line = StatsAPI.fit(PenalizedLeastSquares, B[[1, end], :], [1.0, 3.0]; penalties=[P], lambda=[1.0])
    @test isnan(line.criterion_value) && !line.settings.criterion_defined
    @test line.null_space_dimension == 2 && isnan(line.reml_scale) && isnan(line.sigma2)
    @test line.fitted ≈ [1.0, 3.0]
    @test isnan(StatsAPI.fit(PenalizedLeastSquares, B[[1, end], :], [1.0, 3.0]; penalties=[P],
        lambda=[1.0], criterion=:GCV).criterion_value)
    # Selection still rejects these points.
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, B, zeros(length(t)); penalties=[P])
    # A criterion that overflows (GCV's n·RSS beyond floatmax) is undefined as
    # well; β is linear in y at fixed lambda.
    wiggly = 1 .+ t .+ 0.3 .* sinpi.(7 .* t)
    huge = StatsAPI.fit(PenalizedLeastSquares, B, 1e154 .* wiggly; penalties=[P], lambda=[1.0],
        criterion=:GCV)
    @test isnan(huge.criterion_value) && !huge.settings.criterion_defined
    @test coef(huge) ≈ 1e154 .* coef(StatsAPI.fit(PenalizedLeastSquares, B, wiggly; penalties=[P],
        lambda=[1.0])) rtol=1e-10
    # maxiter=0 stops at the initial smoothing parameters.
    stopped = StatsAPI.fit(PenalizedLeastSquares, B, wiggly; penalties=[P], maxiter=0)
    @test stopped.settings.convergence === :maxiter && stopped.settings.iterations == 0
    @test stopped.lambda ≈ stopped.settings.initial_lambda rtol=1e-12
    model = StatsAPI.fit(PenalizedLeastSquares, X, y; good...)
    @test_throws ArgumentError vcov(model; type=:sandwich)
    @test_throws DimensionMismatch StatsAPI.predict(model, ones(2, 2))
end

@testset "PenalizedSpline REML through the penalized solver" begin
    basis = BSplineBasis(4, (0.0, 1.0), 7)
    t = collect(range(0.0, 1.0; length=15))
    values = [sin(2pi * x) + i * x + 0.2 * cos(11 * i * x + i) for i in 1:3, x in t]
    fd = GridFunctionalData(values, t; ids=[:a, :b, :c])
    method = @inferred PenalizedSpline(basis; lambda=REML())
    @test method.lambda isa REML
    @test sprint(show, REML()) == "REML()"
    @test occursin("lambda=REML()", sprint(show, method))
    smoothed = smooth_curves(fd, method)
    selection = smoothed.metadata.selection
    @test selection.criterion === :REML
    @test selection.status === :estimated
    @test selection.convergence === :gradient
    @test selection.tol == 1e-8 && selection.maxiter == 200 && selection.rank_tol == eps()^(3/4)
    @test smoothed.metadata.lambda == selection.lambda
    @test ids(smoothed) == [:a, :b, :c]
    @test occursin("REML", sprint(show, smoothed))
    @test StatsAPI.fit(PenalizedSpline, fd; basis, lambda=REML()).coefs == smoothed.coefs
    @test smoothed.coefs ≈ to_basis(fd, basis; lambda=selection.lambda).coefs rtol=1e-10

    # Stacked model: block-diagonal design and penalty, one lambda, one scale.
    k = nbasis(basis)
    Xi = basis_matrix(basis, t)
    stacked = zeros(3 * length(t), 3k)
    for i in 1:3
        stacked[(i-1)*length(t) .+ (1:length(t)), (i-1)*k .+ (1:k)] .= Xi
    end
    P = penalty(basis, 2)
    joint = StatsAPI.fit(PenalizedLeastSquares, stacked, vec(permutedims(values));
        penalties=[kron(Matrix(1.0I, 3, 3), P)])
    @test selection.lambda ≈ only(joint.lambda) rtol=1e-6
    @test selection.score ≈ joint.criterion_value rtol=1e-10
    @test selection.sigma2 ≈ joint.sigma2 rtol=1e-6
    @test selection.edf ≈ joint.edf_total rtol=1e-6
    @test selection.null_space_dimension == joint.null_space_dimension == 6
    @test smoothed.coefs ≈ permutedims(reshape(coef(joint), k, 3)) rtol=1e-6

    # Irregular curves with fewer points than basis functions in one curve.
    irregular = IrregularFunctionalData([t, t[1:2:end], [0.1, 0.5, 0.9]],
        [values[1, :], values[2, 1:2:end], [0.0, 1.0, 0.5]])
    reml_irregular = smooth_curves(irregular, method)
    @test reml_irregular.coefs ≈ to_basis(irregular, basis; lambda=reml_irregular.metadata.lambda).coefs rtol=1e-8

    # The default remains a required choice; GCV behaviour is unchanged.
    @test_throws ArgumentError PenalizedSpline(basis; lambda=:ML)
    @test_throws ArgumentError PenalizedSpline(basis; lambda=:REML)
    @test_throws ArgumentError smooth_curves(fd, PenalizedSpline(basis; lambda=REML(), order=4))

    # FPCA prediction freezes the REML-selected training lambda.
    trainingfit = StatsAPI.fit(FPCA, fd; presmooth=method, ncomponents=1)
    newdata = GridFunctionalData(reshape(1 .+ t .+ 0.3 .* t .^ 2, 1, :), t)
    frozen = PenalizedSpline(basis; lambda=trainingfit.settings.preprocessing.lambda)
    frozenvalues = evaluate(smooth_curves(newdata, frozen), t)
    @test StatsAPI.predict(trainingfit, newdata) ≈ (frozenvalues .- trainingfit.mean.values) *
        (trainingfit.weights .* eigenfunctions(trainingfit).values)
end
