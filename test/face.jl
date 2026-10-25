import StatsAPI
using DelimitedFiles
import RecipesBase

@testset "FACE: refund 0.1-40 and L2 conversion" begin
    FDA = FunctionalDataAnalysis
    readface(name) = readdlm(joinpath(@__DIR__, "data", "m6_$(name).csv"), ','; comments=true)
    input = readface("face_input")
    grid = input[:, 1]
    Y = permutedims(input[:, 2:end])
    n, J = size(Y)
    fd = GridFunctionalData(Y, grid; ids=["face-$i" for i in axes(Y, 1)])
    relerr(x, y) = norm(x-y)/norm(y)
    reference(case) = readdlm(joinpath(@__DIR__, "data", "m6_face_$(case).csv"), ',', Any; comments=true)
    function quantity(golden, name)
        rows = golden[golden[:, 1] .== name, :]
        isempty(rows) && return zeros(0, 0)
        reshape(Float64.(rows[:, 4]), Int(maximum(rows[:, 2])), Int(maximum(rows[:, 3])))
    end

    ps = readdlm(joinpath(@__DIR__, "data", "m6_pspline.csv"), ',', Any; comments=true)
    setting = FDA._pspline_setting(grid, 35, 3, 2)
    @test relerr(setting.B, quantity(ps, "basis")) < 1e-8
    @test setting.P == quantity(ps, "penalty")
    @test setting.knots ≈ vec(quantity(ps, "knots")) rtol=1e-14
    @test nbasis(setting.basis) == 38
    @test basis_matrix(setting.basis, grid) ≈ setting.B
    @test domain(setting.basis) == (0.0, 2.0)
    @test FDA._difference_penalty(5, 0) == Matrix{Float64}(I, 5, 5)
    @test FDA._difference_penalty(5, 1) ≈ diff(Matrix{Float64}(I, 5, 5); dims=1)' * diff(Matrix{Float64}(I, 5, 5); dims=1)
    constant_splines = FDA._pspline_setting(grid, 4, 0, 0)
    @test vec(sum(constant_splines.B; dims=2)) ≈ ones(length(grid))
    @test constant_splines.P == Matrix{Float64}(I, 4, 4)

    for case in ("default", "fixed", "missing", "package_mean")
        golden = reference(case)
        q(name) = quantity(golden, name)
        data = fd
        if case == "missing"
            missing_input = Int.(readface("face_missing_input"))
            missingY = Matrix{Union{Missing,Float64}}(undef, size(Y))
            missingY .= Y
            for (i, j) in eachrow(missing_input)
                missingY[i, j] = missing
            end
            data = GridFunctionalData(missingY, grid; ids=fd.ids)
        end
        supplied = (; grid, values=vec(q("mean")))
        options = case == "fixed" ? (; lambda=2.5) : (;)
        model = case == "package_mean" ?
            (@inferred ((d)->StatsAPI.fit(FPCA, d; method=:face))(data)) :
            StatsAPI.fit(FPCA, data; method=:face, mean=supplied, face=options)
        @test model.method == :face
        _test_prediction_contract(model, data)
        @test nobs(model) == n
        @test relerr(vec(model.mean.values), vec(q("mean"))) < 1e-8
        if case != "package_mean"
            @test model.settings.mean.input.values == supplied.values
            @test model.settings.mean.input.values !== supplied.values
        end
        @test relerr(eigvals(model), vec(q("eigenvalue"))) < 1e-8
        @test relerr(model.basis.values, q("eigenfunction")) < 1e-8
        @test relerr(scores(model), q("score")) < 1e-8
        @test noise_variance(model) ≈ only(q("sigma2")) rtol=1e-8
        @test model.settings.lambda ≈ only(q("lambda")) rtol=1e-8
        @test relerr(model.settings.gcv_scores, vec(q("gcv"))) < 1e-8
        fitted_curves = Int.(vec(q("fitted_curves")))
        @test relerr(reconstruct(model).values[fitted_curves, :], q("fitted")) < 1e-8
        @test model.basis.values' * (model.weights .* model.basis.values) ≈ I atol=2e-12
        @test model.settings.imputation.iterations == only(q("iterations"))
        @test model.settings.imputation.converged
        completed = data.values .- transpose(vec(q("mean")))
        if case == "missing"
            completed[.!data.observed] = vec(q("imputed"))
        end
        @test relerr(model.settings.imputation.completed_centered, completed) < 1e-8
        prediction = StatsAPI.predict(model, data)
        @test prediction.scores == scores(model)
        @test prediction.covariances == score_covariances(model)
        @test ids(reconstruct(model)) == ids(data)
        @test model.covariance isa FDA._LowRankCovariance
        @test size(model.covariance.factor) == (J, 38)
        if case != "missing"
            # Compare the compressed factor with the uncompressed smoothed
            # observations, independently of any eigenspectrum truncation.
            h = inv.(1 .+ model.settings.lambda .* setting.s)
            uncompressed = setting.A * (h .* (setting.A' * completed')) / sqrt(nobs(model))
            @test Matrix(model.covariance) ≈ uncompressed * uncompressed' rtol=1e-12
        end
        @test covariance_data(model).values ≈ Matrix(model.covariance)
        @test fitted_curve_data(model, 1).method == :face
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol,Any}(), FDA._plot_payload(Val(:scree), scree_data(model))))
        @test mean_data(model).values ≈ vec(q("mean"))
        @test scree_data(model).eigenvalues ≈ model.all_eigenvalues

        bands = fitted_bands(model; curves=[1, 3])
        @test relerr(vec(bands.variance[1, :]), vec(q("posterior_variance"))) < 1e-8
        @test norm(q("posterior_variance")-q("refund_variance")) > 1e-4
        # Independent observation-space posterior, without the precision solve.
        Phi = model.basis.values
        Lambda = Diagonal(eigvals(model))
        sigma2 = noise_variance(model)
        posterior = Lambda - Lambda*Phi' * ((Phi*Lambda*Phi' + sigma2*I) \ (Phi*Lambda))
        @test score_covariances(model)[:, :, 1] ≈ posterior rtol=1e-8
        @test bands.variance[1,:] ≈ diag(Phi*posterior*Phi') rtol=1e-8
        @test_throws ArgumentError fitted_bands(model; coverage=:band)

        # The rectangular compatibility calculation isolates refund's raw
        # scaling. Public constructors always use stored trapezoidal weights.
        if case != "package_mean"
            compatibility = deepcopy(data)
            delta = (last(grid)-first(grid))/length(grid)
            compatibility.weights .= delta
            raw = StatsAPI.fit(FPCA, compatibility; method=:face, mean=supplied, face=options)
            reference_phi = q("refund_eigenfunction")/sqrt(delta)
            reference_score = q("refund_score")*sqrt(delta)
            for k in axes(reference_phi, 2)
                if dot(raw.basis.values[:,k], reference_phi[:,k]) < 0
                    reference_phi[:,k] .*= -1
                    reference_score[:,k] .*= -1
                end
            end
            @test relerr(eigvals(raw), vec(q("refund_eigenvalue"))*delta) < 1e-8
            @test relerr(raw.basis.values, reference_phi) < 1e-8
            @test relerr(scores(raw), reference_score) < 1e-8
            @test relerr(reconstruct(raw).values[fitted_curves, :], q("refund_fitted")) < 1e-8
            PhiR = raw.basis.values*sqrt(delta)
            SigmaR = eigvals(raw)/(delta*length(grid))
            refund_cov = noise_variance(raw)*PhiR *
                ((PhiR'PhiR + noise_variance(raw)*Diagonal(SigmaR)) \ PhiR')
            @test relerr(diag(refund_cov), vec(q("refund_diag_var"))) < 1e-8
            @test norm(fitted_bands(raw; curves=[1]).variance[1,:]-diag(refund_cov)) > 1e-4
        end
    end

    @testset "Prediction, multivariate reuse, copying and errors" begin
        model = @inferred ((d)->StatsAPI.fit(FPCA, d; method=:face,
            ncomponents=3, face=(; lambda=2.5)))(fd)
        newdata = GridFunctionalData(Y[1:2, :] .+ 0.1, grid)
        prediction = StatsAPI.predict(model, newdata)
        Phi = model.basis.values
        precision = Phi'Phi + Diagonal(noise_variance(model) ./ eigvals(model))
        @test prediction.scores ≈ ((newdata.values .- model.mean.values)*Phi)/precision
        @test score_covariances(model)[:,:,1] ≈ noise_variance(model)*inv(precision)
        # Exercise the shared PACE observation-space solve after factoring its
        # per-curve algebra. Both routes describe this identical selected model.
        selected_covariance = Phi * Diagonal(eigvals(model)) * Phi'
        pace = StatsAPI.fit(FPCA, newdata; method=:pace, grid,
            mean=(; grid, values=vec(model.mean.values)),
            covariance=(; grid, surface=selected_covariance),
            noise_variance=noise_variance(model), ncomponents=3, max_k=3)
        @test scores(pace) ≈ prediction.scores rtol=1e-10
        @test score_covariances(pace) ≈ prediction.covariances rtol=1e-10
        @test StatsAPI.predict(pace, newdata).scores ≈ scores(pace)
        masked = Matrix{Union{Missing,Float64}}(newdata.values)
        masked[1, 1:8] .= missing
        maskeddata = GridFunctionalData(masked, grid)
        maskedprediction = StatsAPI.predict(model, maskeddata)
        keep = 9:J
        localPhi = Phi[keep, :]
        localPrecision = localPhi'localPhi + Diagonal(noise_variance(model) ./ eigvals(model))
        @test maskedprediction.scores[1, :] ≈ localPrecision \ (localPhi'*(newdata.values[1,keep]-vec(model.mean.values)[keep]))
        @test maskedprediction.covariances[:,:,1] ≈ noise_variance(model)*inv(localPrecision)
        masked[1,:] .= missing
        @test_throws ArgumentError StatsAPI.predict(model, GridFunctionalData(masked, grid))
        @test_throws ArgumentError StatsAPI.predict(model, to_irregular(fd))
        @test_throws ArgumentError StatsAPI.predict(model, GridFunctionalData(Y, grid .+ .01))
        mfd = MultivariateFunctionalData([fd, GridFunctionalData(Y .* 1.5, grid; ids=fd.ids)])
        marginal = StatsAPI.fit(FPCA, mfd; method=:face, ncomponents=3, face=(; lambda=2.5))
        @test all(f.method == :face for f in marginal.fits)
        joint = StatsAPI.fit(MFPCA, mfd; ncomponents=2,
            univariate=(; method=:face, ncomponents=3, face=(; lambda=2.5)))
        @test component(joint, 1).method == :face
        _test_prediction_contract(marginal, mfd)
        _test_prediction_contract(joint, mfd)
        @test size(scores(joint)) == (n, 2)
        @test StatsAPI.predict(joint, mfd) ≈ scores(joint)

        uncentered = StatsAPI.fit(FPCA, fd; method=:face, face=(; center=false, lambda=0.0))
        @test all(iszero, uncentered.mean.values)
        nonuniform = collect(range(0, 1; length=90)).^1.3
        curves = [sin(i*.2)*(1+t)+cos(i*.7)*sinpi(t) for i in 1:20, t in nonuniform]
        unequal = StatsAPI.fit(FPCA, GridFunctionalData(curves, nonuniform);
            method=:face, face=(; knots=12, lambda=.5))
        @test unequal.basis.values'*(unequal.weights .* unequal.basis.values) ≈ I atol=1e-12
        # n < basis dimension and alternative degree/difference order.
        few = StatsAPI.fit(FPCA, fd[1:8]; method=:face, ncomponents=2,
            face=(; p=2, m=1, lambda=1.0))
        @test size(scores(few)) == (8, 2)
        @test size(few.covariance.factor) == (J, 8)
        all_components = StatsAPI.fit(FPCA, fd; method=:face, ncomponents=FVE(1),
            face=(; lambda=2.5))
        @test ncomponents(all_components) == length(all_components.all_eigenvalues)
        f32 = @inferred ((d)->StatsAPI.fit(FPCA, d; method=:face, ncomponents=2,
            face=(; lambda=2.5)))(GridFunctionalData(Float32.(Y), Float32.(grid)))
        @test eltype(scores(f32)) == Float32
        @test eltype(eigvals(f32)) == Float32
        # Analytic noiseless rank-one covariance: no optimizer or RNG involved.
        analytic = StatsAPI.fit(FPCA, GridFunctionalData([a*(1+t) for a in [-2.,-1,1,2], t in grid], grid);
            method=:face, ncomponents=1, face=(; lambda=0.0))
        @test only(eigvals(analytic)) ≈ 2.5*sum(fd.weights .* (1 .+ grid).^2) rtol=1e-12
        @test noise_variance(analytic) < 1e-12
        @test reconstruct(analytic).values ≈ analytic.training_data.values rtol=1e-12
        dense = StatsAPI.fit(FPCA, analytic.training_data; method=:dense, ncomponents=1)
        @test eigvals(dense) ≈ eigvals(analytic)*4/3 rtol=1e-12
        # Roundoff-sized fitted means may legitimately select opposite signs.
        orientation = dot(vec(dense.basis.values), vec(analytic.basis.values)) < 0 ? -1 : 1
        @test dense.basis.values ≈ orientation .* analytic.basis.values rtol=1e-12
        @test scores(dense) ≈ orientation .* scores(analytic) rtol=1e-12
        for fitted in (dense, analytic)
            phi = vec(fitted.basis.values)
            mu = vec(fitted.mean.values)
            alignment = dot(phi, mu)
            threshold = fitted.sign_rule.tolerance * norm(phi) * norm(mu)
            if abs(alignment) <= threshold
                @test phi[argmax(abs.(phi))] > 0
            else
                @test alignment > 0
            end
        end
        @test reconstruct(dense).values ≈ analytic.training_data.values rtol=1e-12
        custom_smoother = PenalizedSpline(BSplineBasis(4, (0., 2.), 8); lambda=.1)
        custom = StatsAPI.fit(FPCA, fd; method=:face,
            face=(; mean_smoother=custom_smoother, lambda=2.5))
        direct_mean = smooth_curves(GridFunctionalData(sum(Y; dims=1)/size(Y,1), grid), custom_smoother)
        @test custom.mean.values ≈ evaluate(direct_mean, grid)
        @test custom.settings.mean.smoothing.lambda == .1

        snapshot = deepcopy(model)
        fd.values[1,1] += 1
        @test model.training_data.values == snapshot.training_data.values
        @test model.basis.values == snapshot.basis.values
        for text in ("lambda", "knots", "alpha", "center", "converged", "mean", "quadrature")
            @test occursin(text, sprint(show, model))
        end
        for options in ((; knots=0), (; knots=J-2), (; p=-1), (; m=-1),
                (; m=38), (; alpha=0), (; lambda=-1), (; lambda=Inf),
                (; loglambda=[1.,1.]), (; loglambda=[1000.]), (; loglambda=Float64[]),
                (; center=1), (; unknown=true))
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, face=options)
        end
        for selector in (AIC(), BIC())
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, ncomponents=selector)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, max_k=39)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, ncomponents=100)
        for bad in (0, -1, 1.5, :invalid, true)
            @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, ncomponents=bad)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; face=(; lambda=2.5))
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, presmooth=:bad)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, noise_variance=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face, face=[1])
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face,
            mean=(; grid, values=zeros(J)), face=(; center=false))
        @test_throws DimensionMismatch StatsAPI.fit(FPCA, fd; method=:face,
            mean=(; grid, values=zeros(J-1)))
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face,
            mean=(; grid, values=zeros(J)), face=(; mean_smoother=custom_smoother))
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face,
            face=(; mean_smoother=:invalid))
        @test_throws ArgumentError StatsAPI.fit(FPCA, fd; method=:face,
            face=(; mean_smoother=PenalizedSpline(custom_smoother.basis; lambda=.1, order=1)))
        absent = Matrix{Union{Missing,Float64}}(Y)
        absent[1,:] .= missing
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(absent, grid); method=:face)
        absent = Matrix{Union{Missing,Float64}}(Y)
        absent[:,1] .= missing
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(absent, grid); method=:face)
        covered = StatsAPI.fit(FPCA, GridFunctionalData(absent[1:8, :], grid); method=:face,
            mean=(; grid, values=vec(model.mean.values)))
        @test isfinite(noise_variance(covered))
        absent[1,4:end] .= missing
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(absent, grid); method=:face)
        # Strictly increasing times can still produce a rank-deficient spline
        # design when almost all points lie in one knot interval.
        clustered = vcat(collect(range(0, 1e-5; length=J-1)), 2.)
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(Y, clustered); method=:face)
        @test_throws ArgumentError StatsAPI.fit(FPCA, GridFunctionalData(zeros(5,J), grid); method=:face)
    end
end

@testset "FACE derivative scores use the full observation covariance" begin
    grid = 2 .* collect(range(0, 1; length=75)).^1.2
    mu = 1 .+ grid.^2
    Y = [mu[j] + cos(0.4i)*(1+t) + sin(0.7i)*sinpi(t) +
         sin(1.1i)*cospi(2t) + 0.1cos(2i+11t) for i in 1:12, (j,t) in enumerate(grid)]
    for missing_values in (false, true)
        data = Matrix{Union{Missing,Float64}}(Y)
        if missing_values
            data[1, [1, 3, 5]] .= missing
            data[2, 20:24] .= missing
        end
        fd = GridFunctionalData(data, grid)
        f = StatsAPI.fit(FPCA, fd; method=:face, ncomponents=1, max_k=4,
            mean=(; grid, values=mu), face=(; knots=8, lambda=0.5))
        @test length(f.all_eigenvalues) == 4 > ncomponents(f)
        @test noise_variance(f) > 0
        d = derivative(f; method=:DPC, bandwidth=0.4, ncomponents=2)
        # Observation-space matrices provide an independent check of both the
        # cross-covariance numerator and the full max_k covariance denominator.
        G = Matrix(f.settings.full_projected_covariance)
        L = FunctionalDataAnalysis.KernelSmoothing.smoother_matrix(grid, grid;
            bandwidth=0.4, derivative=1, degree=2,
            kernel=FunctionalDataAnalysis.KernelSmoothing.GaussianKernel())
        cross = transpose(L * G) * (f.weights .* d.basis.values)
        @test d.settings.derivative_covariance ≈ L * G * L' rtol=1e-11
        for i in 1:nobs(fd)
            keep = fd.observed === nothing ? collect(eachindex(grid)) : findall(fd.observed[i, :])
            covariance = G[keep, keep] + noise_variance(f)*I
            numerator = cross[keep, :]
            residual = Y[i, keep] - mu[keep]
            expected_scores = numerator' * (covariance \ residual)
            expected_covariance = Diagonal(eigvals(d)) - numerator' * (covariance \ numerator)
            @test scores(d)[i, :] ≈ expected_scores rtol=1e-10
            @test score_covariances(d)[:, :, i] ≈ expected_covariance rtol=1e-10
        end
        @test scores(derivative(f; bandwidth=0.4)) == scores(f)
    end
end

@testset "FACE complete-grid posterior batches" begin
    FDA = FunctionalDataAnalysis
    for (T,U) in ((Float32,Float32),(Float64,Float64),(Float32,Float64),(Float64,Float32))
        # Dyadic knots also match exactly when the model and input types differ.
        grid = collect(range(T(0),T(1);length=17))
        phi = hcat(ones(T,17),grid .- T(.5))
        values = T[.8,.3]
        mu = T.(sinpi.(grid))
        data = U[mu[j]+(i-2)*phi[j,1]+T(.3)*i*phi[j,2]+T(.05)*sin(T(i)*t)
            for i in 1:3,(j,t) in enumerate(grid)]
        fd = GridFunctionalData(data,U.(grid);ids=[:a,:b,:c])
        saved = copy(fd.values)
        # Include an unselected covariance direction: FACE's selected-model
        # solver must not accidentally adopt DPC's full-spectrum denominator.
        full = FDA._LowRankCovariance(hcat(phi .* transpose(sqrt.(values)),T(4).*sinpi.(2grid)))
        tolerance = T === Float32 || U === Float32 ? 3e-5 : 1e-10
        for sigma2 in (zero(T),T(.2))
            batch = @inferred FDA._face_scores(fd,grid,mu,phi,values,full,sigma2)
            rowwise = FDA._pace_ce(fd,grid,mu,phi,values,full,sigma2;curve_ce=FDA._face_curve_ce)
            @test eltype(batch.scores) === promote_type(T,U)
            @test batch.scores ≈ rowwise.scores rtol=tolerance atol=tolerance
            @test batch.covariances == rowwise.covariances
            @test fd.values == saved
            if sigma2 > 0
                R = promote_type(T,U)
                Phi, Lambda = Matrix{R}(phi),Diagonal(Vector{R}(values))
                observation_covariance = Phi*Lambda*transpose(Phi)+R(sigma2)*I
                posterior = Lambda-Lambda*transpose(Phi)*(observation_covariance \ (Phi*Lambda))
                @test batch.covariances[:,:,1] ≈ posterior rtol=tolerance atol=tolerance
            else
                @test all(iszero,batch.covariances)
            end
            before = copy(batch.covariances[:,:,2])
            batch.covariances[1,1,1] += 1
            @test batch.covariances[:,:,2] == before
        end
        # Irregular, masked, and changed work grids retain per-curve geometry.
        masked = Matrix{Union{Missing,U}}(data);masked[1,3:5] .= missing
        for other in (to_irregular(fd),GridFunctionalData(masked,U.(grid)),
                GridFunctionalData(data[:,2:2:16],U.(grid[2:2:16])))
            actual = FDA._face_scores(other,grid,mu,phi,values,full,T(.2))
            expected = FDA._pace_ce(other,grid,mu,phi,values,full,T(.2);curve_ce=FDA._face_curve_ce)
            @test actual.scores == expected.scores
            @test actual.covariances == expected.covariances
        end
        deficient = hcat(phi[:,1],phi[:,1])
        rank_error = try
            FDA._face_scores(fd,grid,mu,deficient,values,full,zero(T))
        catch err
            err
        end
        @test rank_error isa ArgumentError
        @test occursin("noiseless score design for curve 1 is rank deficient",sprint(showerror,rank_error))
    end
    # Preserve the finite-score contract when new observations overflow after
    # subtracting a finite mean, in both the batched and per-curve routes.
    grid = collect(0.:.25:1.)
    fd = GridFunctionalData(fill(floatmax(Float64),2,5),grid)
    mu,phi = fill(-floatmax(Float64),5),ones(5,1)
    covariance = FDA._LowRankCovariance(phi)
    @test_throws ArgumentError FDA._face_scores(fd,grid,mu,phi,[1.],covariance,.2)
    @test_throws ArgumentError FDA._pace_ce(fd,grid,mu,phi,[1.],covariance,.2;curve_ce=FDA._face_curve_ce)
end
