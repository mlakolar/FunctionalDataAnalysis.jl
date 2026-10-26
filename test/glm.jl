import StatsAPI
using DelimitedFiles

include(joinpath(@__DIR__, "reference", "mgcv_m7_inputs.jl"))

# Deterministic, Julia-version-independent values in (0, 1) (additive
# recurrence with plastic-number increments).
_glm_uniform(n, seed) = [mod(i * 0.7548776662466927 + seed * 0.5698402909980532 +
    0.37 * sinpi(0.123 * i * seed), 1) for i in 1:n]
_glm_rel(a, b) = norm(a - b) / norm(b)

@testset "Penalized IRLS against mgcv 1.9-3" begin
    inputs = m7_pls_inputs()
    golden = readdlm(joinpath(@__DIR__, "data", "m7g_mgcv.csv"), ',', Any; comments=true)
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
    single = (hcat(inputs.z, inputs.spline_design), [inputs.spline_penalty => 2:11])
    disjoint = (hcat(inputs.spline_design, inputs.fourier_design),
        [inputs.spline_penalty => 1:10, inputs.fourier_penalty => 11:16])
    cases = [("binomial_single", :binomial, nothing, single),
        ("binomial_trials", :binomial, nothing, disjoint),
        ("probit_single", :binomial, :probit, single),
        ("poisson_single", :poisson, nothing, single),
        ("poisson_disjoint", :poisson, nothing, disjoint),
        ("gaussian_log", :gaussian, :log, single)]
    for (case, family, link, (X, penalties)) in cases, method in ("REML", "GCV.Cp")
        @testset "$case $method" begin
            criterion = method == "REML" ? :REML : family === :gaussian ? :GCV : :UBRE
            y = reference(case, "input", "y")
            weights = reference(case, "input", "weights")
            lambda = reference(case, method, "lambda")
            at_mgcv = StatsAPI.fit(PenalizedLeastSquares, X, y; weights, family, link, penalties,
                criterion, lambda)
            @test at_mgcv.lambda == lambda && all(==(:fixed), at_mgcv.lambda_status)
            @test at_mgcv.family === family
            @test at_mgcv.link === (link === nothing ? (family === :binomial ? :logit : :log) : link)
            @test abs(at_mgcv.criterion_value / only(reference(case, method, "score")) - 1) < 1e-8
            @test _glm_rel(StatsAPI.coef(at_mgcv), reference(case, method, "coef")) < 1e-8
            @test _glm_rel(StatsAPI.fitted(at_mgcv), reference(case, method, "fitted")) < 1e-8
            @test _glm_rel(at_mgcv.linear_predictors, reference(case, method, "eta")) < 1e-8
            @test _glm_rel(at_mgcv.edf, reference(case, method, "edf")) < 1e-8
            @test abs(at_mgcv.deviance / only(reference(case, method, "deviance")) - 1) < 1e-8
            @test abs(at_mgcv.sigma2 / only(reference(case, method, "sig2")) - 1) < 1e-8
            Vp = reference(case, method, "Vp")
            upper = triu(trues(size(Vp)))
            @test _glm_rel(StatsAPI.vcov(at_mgcv)[upper], Vp[upper]) < 1e-8
            if family === :gaussian
                @test at_mgcv.settings.scale === :estimated
                method == "REML" &&
                    @test abs(at_mgcv.reml_scale / only(reference(case, method, "reml_scale")) - 1) < 1e-8
                @test StatsAPI.dof(at_mgcv) ≈ at_mgcv.edf_total + 1
            else
                @test at_mgcv.settings.scale === :known && at_mgcv.sigma2 == 1 && at_mgcv.reml_scale == 1
                @test StatsAPI.dof(at_mgcv) == at_mgcv.edf_total
            end
            @test StatsAPI.fitted(at_mgcv) + StatsAPI.residuals(at_mgcv) ≈ y
            @test StatsAPI.predict(at_mgcv, X) ≈ at_mgcv.linear_predictors
            @test StatsAPI.predict(at_mgcv, X; type=:response) ≈ StatsAPI.fitted(at_mgcv)

            # Our optimum is no worse than mgcv's. The references were run with
            # mgcv's conv.tol tightened to 1e-10, so ours is tightened too.
            ours = StatsAPI.fit(PenalizedLeastSquares, X, y; weights, family, link, penalties,
                criterion, tol=1e-11)
            @test ours.settings.convergence === :gradient
            @test ours.criterion_value <= at_mgcv.criterion_value +
                8eps() * (1 + abs(at_mgcv.criterion_value))
            if case == "poisson_disjoint" && method == "REML"
                # mgcv stops its first parameter at 4e6, an effective infinity.
                @test ours.lambda_status == [:effectively_infinite, :estimated]
                @test ours.lambda[1] > lambda[1]
            else
                @test all(==(:estimated), ours.lambda_status)
                @test _glm_rel(ours.lambda, lambda) < 1e-4
            end
            @test ours.settings.irls_iterations >= 1
        end
    end
end

@testset "Penalized IRLS: derivatives, score equations and validation" begin
    inputs = m7_pls_inputs()
    X = hcat(ones(80), inputs.spline_design[:, 2:end], inputs.fourier_design)
    penalties = [inputs.spline_penalty[2:end, 2:end] => 2:10, inputs.fourier_penalty => 11:16]
    eta = 0.5 .* sinpi.(2 .* inputs.x1) .+ 0.4 .* cospi.(4 .* inputs.x2)
    u = _glm_uniform(80, 3)
    bernoulli = Float64.(u .< 1 ./ (1 .+ exp.(-2 .* eta)))
    counts = Float64.(floor.(exp.(1 .+ eta) .+ 3 .* u))
    responses = Dict(:binomial => bernoulli, :poisson => counts,
        :gaussian => exp.(1 .+ eta) .+ 0.4 .* (u .- 0.5))
    links = [(:binomial, :logit), (:binomial, :probit), (:binomial, :cloglog), (:poisson, :log),
        (:poisson, :identity), (:poisson, :sqrt), (:gaussian, :log), (:gaussian, :inverse)]
    for (family, link) in links
        y = family === :gaussian && link === :inverse ? 1 ./ (2 .+ eta) .+ 0.05 .* (u .- 0.5) :
            family === :poisson && link === :identity ? counts .+ 1 : responses[family]
        known = family !== :gaussian
        for criterion in (:REML, known ? :UBRE : :GCV)
            fixed(rho) = StatsAPI.fit(PenalizedLeastSquares, X, y; family, link, penalties,
                criterion, lambda=exp.(rho))
            rho = [log(0.3), log(2.0)]
            model = fixed(rho)
            h = 1e-5
            steps = [h .* (1:2 .== j) for j in 1:2]
            numeric_gradient = [(fixed(rho + s).criterion_value - fixed(rho - s).criterion_value) / 2h
                for s in steps]
            numeric_hessian = reduce(hcat,
                [(fixed(rho + s).settings.gradient - fixed(rho - s).settings.gradient) / 2h for s in steps])
            scale = 1 + abs(model.criterion_value)
            @test norm(model.settings.gradient - numeric_gradient) <= 1e-6 * scale
            @test norm(model.settings.hessian - numeric_hessian) <= 1e-6 * scale
            # Score equations of the penalized deviance at the solution.
            S = zeros(16, 16)
            for ((P, c), l) in zip(penalties, model.lambda)
                S[c, c] .+= l .* P
            end
            mu, b = StatsAPI.fitted(model), StatsAPI.coef(model)
            V = family === :binomial ? mu .* (1 .- mu) : family === :poisson ? mu : one.(mu)
            dmu = map(e -> link === :logit ? exp(-e) / (1 + exp(-e))^2 :
                link === :probit ? exp(-e^2 / 2) / sqrt(2pi) :
                link === :cloglog ? exp(e - exp(e)) : link === :log ? exp(e) :
                link === :inverse ? -1 / e^2 : link === :sqrt ? 2e : 1.0, model.linear_predictors)
            score = transpose(X) * ((y .- mu) ./ V .* dmu)
            @test norm(score - S * b) <= 1e-8 * (1 + norm(S * b))
            # Fisher weights give Vp (they equal the Newton weights for
            # canonical links, as mgcv's post-processing uses).
            H = transpose(X) * ((dmu .^ 2 ./ V) .* X) + S
            @test StatsAPI.vcov(model) ≈ inv(H) .* model.sigma2 rtol=1e-8
            @test model.edf_total ≈ tr(H \ (transpose(X) * ((dmu .^ 2 ./ V) .* X))) rtol=1e-8
        end
    end

    # The Gaussian identity link through the IRLS path is the closed-form solver.
    FDA = FunctionalDataAnalysis
    y = eta .+ 0.3 .* (u .- 0.5)
    mats = [Matrix(first(p)) for p in penalties]
    cols = [collect(last(p)) for p in penalties]
    groups, unpenalized = FDA._penalty_groups(mats, cols, 16)
    family = FDA._glm_family(:gaussian, nothing)
    problem = FDA._PIRLSProblem{Float64}(X, y, ones(80), family, 80, 16, mats, cols, groups,
        unpenalized, FDA._pls_rank_tol(nothing, Float64), FDA._glm_saturated(family, y, ones(80)),
        zeros(16), Ref(false), 1e-12, 200)
    for criterion in (:REML, :GCV)
        closed = StatsAPI.fit(PenalizedLeastSquares, X, y; penalties, criterion, lambda=[0.3, 2.0])
        irls = FDA._pls_evaluate(problem, [0.3, 2.0], criterion)
        @test irls.value ≈ closed.criterion_value rtol=1e-12
        @test irls.gradient ≈ closed.settings.gradient rtol=1e-8
        @test irls.hessian ≈ closed.settings.hessian rtol=1e-8
        @test only(irls.coefficients) ≈ StatsAPI.coef(closed) rtol=1e-10
    end
    # Same on Float32 inputs.
    single32 = StatsAPI.fit(PenalizedLeastSquares, Float32.(X), Float32.(bernoulli);
        family=:binomial, penalties=[Float32.(first(p)) => last(p) for p in penalties])
    @test eltype(StatsAPI.coef(single32)) == Float32
    inferred = @inferred StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; family=:binomial, penalties)
    @test inferred isa PenalizedLeastSquares{Float64}
    @test occursin("family=binomial, link=logit", sprint(show, inferred))
    @test_throws ArgumentError StatsAPI.predict(inferred, X; type=:mean)

    # Input validation.
    good = (; penalties)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; good..., family=:gamma)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; good..., family=:binomial,
        link=:log)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; good..., family=:binomial,
        criterion=:GCV)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., criterion=:UBRE)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., family=:gaussian,
        link=:log, criterion=:UBRE)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, 2 .* bernoulli; good..., family=:binomial)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli ./ 3; good..., family=:binomial)
    @test StatsAPI.fit(PenalizedLeastSquares, X, bernoulli ./ 3; good..., family=:binomial,
        weights=fill(3.0, 80), lambda=[1.0, 1.0]) isa PenalizedLeastSquares
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, counts .+ 0.5; good..., family=:poisson)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, counts .- 10; good..., family=:poisson)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y; good..., family=:gaussian, link=:log)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, y .* 0; good..., family=:gaussian,
        link=:inverse)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; good..., family=:binomial,
        irls_tol=0)
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; good..., family=:binomial,
        irls_maxiter=0)
    # Classes separated by an unpenalized direction: IRLS diverges and throws.
    t = collect(range(-1, 1; length=40))
    separated = Float64.(t .> 0)
    Xs = hcat(ones(40), t, sinpi.(t), cospi.(t))
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, Xs, separated; family=:binomial,
        penalties=[[1.0 0; 0 1] => 3:4], lambda=[1.0])

    # Undefined criteria of PIRLS fits with fixed lambda are reported as NaN.
    tb = collect(range(0.0, 1.0; length=20))
    sb = BSplineBasis(4, (0.0, 1.0), 8)
    Bs, Ps = basis_matrix(sb, tb), penalty(sb, 2)
    # n = Mₚ = 2 with an estimated scale: REML has no residual degrees of
    # freedom and GCV has n - τ = 0.
    for criterion in (:REML, :GCV)
        two = StatsAPI.fit(PenalizedLeastSquares, Bs[[1, end], :], [1.0, 3.0]; penalties=[Ps],
            lambda=[1.0], family=:gaussian, link=:log, criterion)
        @test isnan(two.criterion_value) && !two.settings.criterion_defined
        @test two.fitted ≈ [1.0, 3.0]
    end
    @test_throws ArgumentError StatsAPI.fit(PenalizedLeastSquares, Bs[[1, end], :], [1.0, 3.0];
        penalties=[Ps], family=:gaussian, link=:log)
    # Means near 1e100 overflow the weight derivatives (μ⁴): the criterion is
    # undefined but the fit is not. Scaling y by s shifts η by log s (the
    # B-splines sum to one, a direction the penalty ignores) and divides the
    # effective lambda by s².
    wiggly = 1 .+ tb .+ 0.3 .* sinpi.(7 .* tb)
    scaled = StatsAPI.fit(PenalizedLeastSquares, Bs, 1e100 .* wiggly; penalties=[Ps], lambda=[1.0],
        family=:gaussian, link=:log)
    @test isnan(scaled.criterion_value) && !scaled.settings.criterion_defined
    @test StatsAPI.coef(scaled) ≈ StatsAPI.coef(StatsAPI.fit(PenalizedLeastSquares, Bs, wiggly;
        penalties=[Ps], lambda=[1e-200], family=:gaussian, link=:log)) .+ log(1e100) rtol=1e-8
    # maxiter=0 stops at the initial smoothing parameters.
    stopped = StatsAPI.fit(PenalizedLeastSquares, X, bernoulli; family=:binomial, penalties, maxiter=0)
    @test stopped.settings.convergence === :maxiter && stopped.settings.iterations == 0
    @test stopped.lambda ≈ stopped.settings.initial_lambda rtol=1e-12

    # Where the Newton Hessian of a non-canonical link is not positive
    # definite, PIRLS takes a Fisher scoring step. Gaussian log link at μ = 1:
    # Newton weights μ(2μ - y) = [1, 1, 1, -8] sum to -5 < 0.
    ylog = [1.0, 1, 1, 10]
    flog = FDA._glm_family(:gaussian, :log)
    intercept = FDA._PIRLSProblem{Float64}(ones(4, 1), ylog, ones(4), flog, 4, 1, Matrix{Float64}[],
        Vector{Int}[], FDA._PenaltyGroup{Float64}[], [1], FDA._pls_rank_tol(nothing, Float64),
        FDA._glm_saturated(flog, ylog, ones(4)), zeros(1), Ref(false), 1e-12, 200)
    start = FDA._pirls_state(intercept, zeros(4), zeros(1), zeros(1, 1))
    @test FDA._pirls_values(start, Val(:wn)) ≈ [1, 1, 1, -8]
    newton = FDA._pirls_fit(intercept, ones(4, 1), zeros(1, 1), zeros(1); warm=true)
    @test only(newton.state.beta) ≈ log(3.25) rtol=1e-12
end

@testset "Penalized functional GLM against refund 0.1-40" begin
    G0 = golden_read("m7_refund_pfr.csv")
    G = golden_read("m7g_refund_pfr.csv")
    v0(q) = vec(golden_matrix4(G0, q))
    v(q) = vec(golden_matrix4(G, q))
    g1, X1, z = v0("grid1"), golden_matrix4(G0, "X1"), v0("z")
    train, test = 1:50, 51:56
    x, xnew = GridFunctionalData(X1[train, :], g1), GridFunctionalData(X1[test, :], g1)
    for (case, family, link, criterion) in (("binomial", :binomial, nothing, :REML),
            ("poisson", :poisson, nothing, :REML), ("gaussian_log", :gaussian, :log, :REML),
            ("binomial_ubre", :binomial, nothing, :UBRE))
        @testset "$case" begin
            y = v("y_" * (case == "binomial_ubre" ? "binomial" : case))[train]
            free = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z[train],
                family, link, criterion)
            lambda = v("$(case)_sp") ./ v("$(case)_S_scale") .* free.penalty_scale
            model = StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z[train],
                family, link, criterion, lambda)
            @test model.settings.family === family && model.settings.criterion === criterion
            @test _glm_rel(StatsAPI.coef(model)[1:2], v("$(case)_coefficients")[1:2]) < 1e-8
            @test _glm_rel(StatsAPI.fitted(model), v("$(case)_fitted")) < 1e-8
            @test _glm_rel(model.pls.linear_predictors, v("$(case)_eta")) < 1e-8
            @test _glm_rel([model.sigma2], v("$(case)_sigma2")) < 1e-8
            @test _glm_rel([model.pls.criterion_value], v("$(case)_score")) < 1e-8
            @test _glm_rel([StatsAPI.dof(model)], v("$(case)_edf_total")) < 1e-8
            @test _glm_rel([model.pls.deviance], v("$(case)_deviance")) < 1e-8
            # r2 is mgcv's deviance explained.
            @test StatsAPI.r2(model) ≈ 1 - only(v("$(case)_deviance")) / only(v("$(case)_null_deviance")) rtol=1e-8
            @test free.pls.criterion_value <= model.pls.criterion_value + 1e-10 * abs(model.pls.criterion_value)
            link_scale = StatsAPI.predict(model, xnew; scalar_covariates=z[test], type=:link,
                interval=:confidence)
            @test _glm_rel(link_scale.estimate, v("$(case)_predict_link")) < 1e-8
            @test _glm_rel(link_scale.se, v("$(case)_predict_se")) < 1e-8
            @test _glm_rel(StatsAPI.predict(model, xnew; scalar_covariates=z[test]),
                v("$(case)_predict_response")) < 1e-8
            band = only(coefficient_bands(model; covariance=:bayesian))
            @test _glm_rel(band.grid, v("$(case)_beta_grid")) < 1e-12
            @test _glm_rel(band.estimate, v("$(case)_beta")) < 1e-8
            @test _glm_rel(band.se, v("$(case)_se_vp")) < 1e-8
            if criterion === :REML
                @test free.settings.corrected_covariance === :mgcv_Vc
                # mgcv's Vc at its selected smoothing parameters.
                penalties = [t.penalty ./ t.penalty_scale for t in model.terms]
                pls, final = FunctionalDataAnalysis._pls_fit(model.design, y;
                    penalties=[S => c for (S, c) in zip(penalties, model.pls.penalty_columns)],
                    weights=nothing, family, link, criterion, lambda=model.pls.lambda, lower=nothing,
                    upper=nothing, rank_tol=nothing, tol=1e-8, maxiter=200, irls_tol=1e-12, irls_maxiter=200)
                corrected, _ = FunctionalDataAnalysis._pfr_glm_corrected_covariance(model.design, pls,
                    final, penalties, model.pls.penalty_columns)
                index = 2 .+ (1:model.terms[1].k)
                B = basis_matrix(model.terms[1].basis, band.grid)
                se = sqrt.(vec(sum((B * corrected[index, index]) .* B; dims=2)))
                @test _glm_rel(se, v("$(case)_se")) < 1e-8
            else
                @test free.settings.corrected_covariance === :unavailable
            end
        end
    end

    # Scales, intervals and validation.
    y = v("y_binomial")[train]
    f = @inferred StatsAPI.fit(PenalizedFunctionalRegression, x, y; scalar_covariates=z[train],
        family=:binomial)
    eta = StatsAPI.predict(f, xnew; scalar_covariates=z[test], type=:link)
    @test StatsAPI.predict(f, xnew; scalar_covariates=z[test]) ≈ 1 ./ (1 .+ exp.(-eta))
    @test StatsAPI.fitted(f) ≈ 1 ./ (1 .+ exp.(-f.pls.linear_predictors))
    on_link = StatsAPI.predict(f, xnew; scalar_covariates=z[test], type=:link, interval=:confidence)
    on_mean = StatsAPI.predict(f, xnew; scalar_covariates=z[test], interval=:confidence)
    @test on_link.settings.scale === :link && on_mean.settings.scale === :response
    @test on_mean.lower ≈ 1 ./ (1 .+ exp.(-on_link.lower))
    @test on_mean.upper ≈ 1 ./ (1 .+ exp.(-on_link.upper))
    @test on_mean.se == on_link.se
    @test on_link.upper ≈ on_link.estimate .+ 1.959963984540054 .* on_link.se
    @test on_mean.settings.covariance === :bayesian
    @test_throws ArgumentError StatsAPI.predict(f, xnew; scalar_covariates=z[test], interval=:prediction)
    @test_throws ArgumentError StatsAPI.predict(f, xnew; scalar_covariates=z[test], type=:mean)
    bands = fitted_bands(f; type=:link)
    @test bands.estimate ≈ f.pls.linear_predictors
    @test fitted_bands(f).estimate ≈ StatsAPI.fitted(f)
    @test_throws ArgumentError fitted_bands(f; interval=:prediction)
    @test_throws ArgumentError fitted_bands(f; type=:mean)
    @test coefficient_bands(f)[1].settings.covariance === :corrected
    @test occursin("terms=1, family=binomial, link=logit", sprint(show, f))
    @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, x, 2 .* y; family=:binomial)
    @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, x, y; family=:binomial,
        criterion=:GCV)
    @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, x, y; family=:binomial,
        link=:identity)
    @test_throws ArgumentError StatsAPI.fit(PenalizedFunctionalRegression, x, y .- 0.5;
        family=:gaussian, link=:log)
end

@testset "FPC GLM regression against fda.usc 2.2.0 and glm" begin
    rd(name) = readdlm(joinpath(@__DIR__, "data", "m7_$(name).csv"), ','; comments=true)
    golden = readdlm(joinpath(@__DIR__, "data", "m7g_fdausc.csv"), ',', Any; comments=true)
    ref(case, q) = Float64.(golden[(golden[:, 1] .== case) .& (golden[:, 2] .== q), 4])
    grid = vec(rd("dense_grid"))
    x1 = GridFunctionalData(rd("dense_input1"), grid)
    new1 = GridFunctionalData(rd("dense_heldout1"), grid)
    z = rd("dense_response")[:, 2]
    znew = vec(rd("dense_heldout_scalar"))
    for (case, family, link, scalar) in (("binomial", :binomial, nothing, false),
            ("poisson", :poisson, nothing, true), ("gaussian_log", :gaussian, :log, false))
        @testset "$case" begin
            y = ref(case, "y")
            zs, zn = scalar ? (z, znew) : (nothing, nothing)
            f = StatsAPI.fit(FPCRegression, x1, y; scalar_covariates=zs, ncomponents=3,
                fpca_options=(; method=:dense, max_k=3), family, link)
            @test f.family === family
            # Both packages center the scores, so the intercepts agree.
            @test _glm_rel([StatsAPI.coef(f)[1]], ref(case, "intercept")) < 1e-8
            @test _glm_rel(StatsAPI.fitted(f), ref(case, "fitted")) < 1e-8
            @test StatsAPI.fitted(f) + StatsAPI.residuals(f) ≈ y
            @test _glm_rel([f.deviance], ref(case, "deviance")) < 1e-8
            @test _glm_rel([f.null_deviance], ref(case, "null_deviance")) < 1e-8
            @test StatsAPI.r2(f) ≈ 1 - f.deviance / f.null_deviance
            @test _glm_rel([f.sigma2], ref(case, "dispersion")) < 1e-8
            band = only(coefficient_bands(f))
            @test _glm_rel(band.estimate, ref(case, "beta")) < 1e-8
            @test _glm_rel(band.se, ref(case, "beta_se")) < 1e-8
            prediction = StatsAPI.predict(f, new1; scalar_covariates=zn, type=:link, interval=:confidence)
            @test _glm_rel(prediction.estimate, ref(case, "predict_link")) < 1e-8
            @test _glm_rel(prediction.se, ref(case, "predict_se")) < 1e-8
            @test _glm_rel(StatsAPI.predict(f, new1; scalar_covariates=zn), ref(case, "predict_response")) < 1e-8
            scalar && @test _glm_rel([StatsAPI.coef(f)[2]], ref(case, "scalar_coefficient")) < 1e-8
            # Known scale: normal (Wald) intervals; estimated Gaussian scale: t, as summary.glm.
            if family === :gaussian
                @test band.settings.distribution === :student_t
                @test f.settings.residual_variance === :pearson_over_residual_dof
            else
                @test band.settings.distribution === :normal
                @test band.critical_value ≈ 1.959963984540054 rtol=1e-12
                @test f.settings.residual_variance === :known_one
            end
            @test f.settings.covariance === :conditional_fisher_information
            # R's AIC()/BIC() for glm, at the three-component candidate.
            for (selector, name) in ((AIC(), "AIC"), (BIC(), "BIC"))
                selected = StatsAPI.fit(FPCRegression, x1, y; scalar_covariates=zs, selection=selector,
                    fpca_options=(; max_k=3), family, link)
                @test abs(selected.selection.values[3] / only(ref(case, name)) - 1) < 1e-8
            end
        end
    end
    y = ref("binomial", "y")
    for (selector, name) in ((AIC(), "AIC"), (BIC(), "BIC"))
        selected = StatsAPI.fit(FPCRegression, x1, y; selection=selector, fpca_options=(; max_k=3),
            family=:binomial)
        @test selected.selection.candidates == [[1], [2], [3]]
        @test _glm_rel(selected.selection.values, ref("selection", name)) < 1e-8
        @test only(ncomponents(selected)) == argmin(ref("selection", name))
    end

    # Scales, intervals and validation.
    f = @inferred StatsAPI.fit(FPCRegression, x1, y; ncomponents=3, fpca_options=(; max_k=3),
        family=:binomial)
    eta = StatsAPI.predict(f, new1; type=:link)
    @test StatsAPI.predict(f, new1) ≈ 1 ./ (1 .+ exp.(-eta))
    on_mean = StatsAPI.predict(f, new1; interval=:confidence)
    on_link = StatsAPI.predict(f, new1; type=:link, interval=:confidence)
    @test on_mean.lower ≈ 1 ./ (1 .+ exp.(-on_link.lower))
    @test on_mean.settings.scale === :response && on_link.settings.scale === :link
    @test fitted_bands(f; type=:link).estimate ≈ f.design * StatsAPI.coef(f)
    @test fitted_bands(f).estimate ≈ StatsAPI.fitted(f)
    @test_throws ArgumentError StatsAPI.predict(f, new1; interval=:prediction)
    @test_throws ArgumentError StatsAPI.predict(f, new1; type=:mean)
    @test_throws ArgumentError fitted_bands(f; type=:mean)
    @test occursin("family=binomial, link=logit", sprint(show, f))
    @test f.settings.irls_tol == 1e-12 && f.settings.irls_maxiter == 200
    @test 0 < f.settings.irls_iterations <= 200
    @test occursin("irls_iterations = $(f.settings.irls_iterations)", sprint(show, f))
    @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; selection=:loocv, family=:binomial)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, 2 .* y; ncomponents=3, family=:binomial)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; family=:binomial, link=:log)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, x1; family=:poisson)
    # Gaussian identity fits are unchanged and report RSS as their deviance.
    ysingle = rd("dense_response")[:, 3]
    least = StatsAPI.fit(FPCRegression, x1, ysingle; ncomponents=3, fpca_options=(; max_k=3))
    @test least.family === :gaussian && least.link === :identity
    @test least.settings.irls_iterations == 0
    @test least.deviance ≈ sum(abs2, StatsAPI.residuals(least))
    @test StatsAPI.predict(least, new1; type=:link) == StatsAPI.predict(least, new1)
    # Classes separated by the scores: glm warns, the package throws.
    linear = StatsAPI.fitted(least)
    separated = Float64.(linear .> sort(linear)[12])
    @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, separated; ncomponents=3,
        fpca_options=(; max_k=3), family=:binomial)
    # Bootstrap intervals refit the GLM in every replicate.
    counts = ref("poisson", "y")
    poisson = StatsAPI.fit(FPCRegression, x1, counts; ncomponents=2, fpca_options=(; max_k=2),
        family=:poisson)
    boot = only(coefficient_bands(poisson; method=:bootstrap, B=20, rng=Random.Xoshiro(3)))
    @test all(boot.lower .<= boot.upper)
    @test boot.settings.method === :bootstrap
end
