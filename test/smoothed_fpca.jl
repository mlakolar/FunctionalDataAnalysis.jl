using DelimitedFiles
import StatsAPI
import RecipesBase

function fda_m6_table(path)
    raw = readdlm(path, ',', Any; comments=true, comment_char='#')
    names = unique(String.(raw[:, 1]))
    Dict(name => begin
        entries = raw[raw[:, 1] .== name, :]
        out = zeros(Float64, maximum(Int.(entries[:, 2])), maximum(Int.(entries[:, 3])))
        for row in eachrow(entries)
            out[Int(row[2]), Int(row[3])] = Float64(row[4])
        end
        out
    end for name in names)
end

@testset "Smoothed basis FPCA and functional CCA" begin
    @testset "Harmonic acceleration" begin
        b = FourierBasis((0, 2), 7)
        P = @inferred penalty(b, :harmonic_acceleration)
        @test diag(P) ≈ [0, 0, 0, 36pi^6, 36pi^6, 576pi^6, 576pi^6]
        @test penalty(FourierBasis((0, 2), 1), :harmonic_acceleration) == zeros(1,1)
        @test (@inferred penalty(FourierBasis((0.0f0, 2.0f0), 5), :harmonic_acceleration)) isa Matrix{Float32}
        points = collect(range(0, 2; length=401))
        L = basis_matrix(b, points; derivative=3) + pi^2*basis_matrix(b, points; derivative=1)
        @test transpose(L)*(quadrature_weights(points).*L) ≈ P atol=1e-9 rtol=1e-12
    end

    @testset "Analytic eigenproblem and exact evaluation" begin
        b = FourierBasis((0, 1), 5)
        C = zeros(10, 5)
        for j in 1:5
            C[2j-1, j] = 6-j
            C[2j, j] = -(6-j)
        end
        fd = BasisFunctionalData(C, b; ids=string.(1:10))
        model = @inferred StatsAPI.fit(FPCA, fd)
        @test eigvals(model) ≈ [5, 3.2, 1.8, .8, .2]
        @test model.settings.penalty === :harmonic_acceleration
        @test model.settings.covariance_denominator == 10
        @test model.method == :basis
        _test_prediction_contract(model, fd)
        @test gram(eigenfunctions(model)) ≈ Matrix{Float64}(I, 5, 5)
        @test (@inferred StatsAPI.predict(model, fd)) ≈ scores(model)
        @test (@inferred reconstruct(model)).values ≈ evaluate(fd, model.grid)
        @test ids(reconstruct(model)) == ids(fd)
        @test noise_variance(model) === score_covariances(model) === nothing
        @test_throws ArgumentError fitted_bands(model)
        points = [0.0, .073, .2, .497, .799, 1.0]
        @test reconstruct(model; grid=points).values ≈ evaluate(fd, points)
        @test basis_matrix(eigenfunctions(model), points; derivative=1) ≈
            basis_matrix(b, points; derivative=1)*eigenfunctions(model).coefs
        @test basis_matrix(DerivativeBasis(eigenfunctions(model)), points) ≈
            basis_matrix(eigenfunctions(model), points; derivative=1)
        @test_throws ArgumentError derivative(model)
        @test penalty(eigenfunctions(model), 2) ≈ penalty(b, 2)
        @test domain(eigenfunctions(model)) == domain(fd)
        @test length(mean_data(model).values) == 201
        @test size(modes_of_variation(model, 1).values) == (201, 4)
        @test length(scree_data(model).individual) == 5
        @test size(covariance_data(model).values) == (201, 201)
        @test fitted_curve_data(model, 1).observed.values ≈ vec(evaluate(fd[1], model.grid))
        @test size(RecipesBase.apply_recipe(Dict{Symbol,Any}(), FunctionalDataAnalysis._FDAPlot{:modes}(modes_of_variation(model,1))))[1] > 0
        @test model.training_data.coefs == fd.coefs
        fd.coefs[1,1] = 999
        @test model.training_data.coefs[1,1] == 5
        for lambda in (0.001, .1)
            f = StatsAPI.fit(FPCA, BasisFunctionalData(C, b); lambda, penalty=2, ncomponents=5)
            analytic = diag(transpose(C)*C/10) ./ (1 .+ lambda*diag(penalty(b,2)))
            @test eigvals(f) ≈ sort(analytic; rev=true)
            @test diag(gram(eigenfunctions(f))) ≈ ones(5)
            @test f.settings.lambda == lambda
        end
        shifted = BasisFunctionalData(C .+ transpose([1.,2,3,4,5]), b)
        uncentered = StatsAPI.fit(FPCA, shifted; center=false, ncomponents=5)
        @test uncentered.mean.coefs == zeros(1,5)
        @test reconstruct(uncentered).values ≈ evaluate(shifted, uncentered.grid)
        small = BasisFunctionalData(Float32.(C[:,1:3]), FourierBasis((0.0f0,1.0f0),3))
        f32 = @inferred StatsAPI.fit(FPCA, small)
        @test (@inferred project_scores(f32, small)) isa Matrix{Float32}
        @test (@inferred predict_scores(f32, small)).scores isa Matrix{Float32}
        @test predict_scores(f32, small).covariances === nothing
        @test (@inferred StatsAPI.residuals(f32)) isa BasisFunctionalData{Float32}
        @test (@inferred StatsAPI.predict(f32, small)) isa Matrix{Float32}
        @test (@inferred reconstruct(f32)) isa GridFunctionalData{Float32}
        @test_throws ArgumentError StatsAPI.fit(FPCA,small;lambda=1e-60)
        @test_throws ArgumentError StatsAPI.fit(FPCA,small;
            lambda=LambdaCV([1e-60]),ncomponents=2)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,small;lambda=1e-60)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,small;lambda=LambdaCV([1e-60]))
        uncentered_cv = @inferred StatsAPI.fit(FPCA,shifted;
            center=false,lambda=LambdaCV([0.,1.]),ncomponents=2)
        @test uncentered_cv.settings.lambda_selection.centering == :none
        @test uncentered_cv.settings.lambda_selection.ncomponents == 2
        @test occursin("lambda", sprint(show, model))
    end

    @testset "Analytic canonical correlation and pairing" begin
        b = MonomialBasis((0,1),0)
        x = [-3.,-2,-1,0,0,1,2,3]
        y = [1.,-3,-1,2,-2,0,4,3]
        left, right = BasisFunctionalData(reshape(x,:,1), b), BasisFunctionalData(reshape(y,:,1), b)
        f = @inferred StatsAPI.fit(FunctionalCCA, left, right; ncan=1)
        alias = @inferred StatsAPI.fit(FunctionalCCA, left, right; ncomponents=1)
        @test alias.weights1.coefs == f.weights1.coefs
        @test alias.weights2.coefs == f.weights2.coefs
        @test canonical_variates(alias) == canonical_variates(f)
        @test canonical_correlations(alias) == canonical_correlations(f)
        @test squared_canonical_correlations(alias) == squared_canonical_correlations(f)
        @test alias.settings == f.settings
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA, left, right; ncan=1, ncomponents=1)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA, left, right; ncomponents=true)
        weights = @inferred canonical_weights(f)
        variates = @inferred canonical_variates(f)
        @test keys(weights) == keys(variates) == (:left, :right)
        @test weights.left === f.weights1
        @test weights.right === f.weights2
        @test variates.left === f.variates1
        @test variates.right === f.variates2
        @test (@inferred canonical_correlations(f)) === f.correlations
        @test (@inferred squared_canonical_correlations(f)) === f.squared_correlations
        independent_weights = deepcopy(weights)
        independent_weights.left.coefs[1,1] += 1
        @test independent_weights.left.coefs != weights.left.coefs
        independent_variates = map(copy, variates)
        independent_variates.right[1,1] += 1
        @test independent_variates.right != variates.right
        yc = y .- sum(y)/length(y)
        rho = dot(x,yc)/sqrt(dot(x,x)*dot(yc,yc))
        @test f.correlations ≈ [rho]
        @test f.squared_correlations ≈ [rho^2]
        @test f.weights1.coefs ≈ ones(1,1)
        @test f.variates1 ≈ reshape(x,:,1)
        @test (@inferred StatsAPI.predict(f,left,right)).variates2 ≈ f.variates2
        @test size(StatsAPI.predict(f,left[1],right[1]).variates1) == (1,1)
        @test nobs(f) == 8
        @test ncomponents(f) == 1
        @test ids(f) === nothing
        @test occursin("lambda", sprint(show,f))
        @test occursin("center", sprint(show,f))
        @test f.settings.lambda == (1e-10,1e-10)
        @test f.settings.penalty == (2,2)
        @test StatsAPI.fit(FunctionalCCA,left,right; ncan=1,center=false).mean1.coefs == zeros(1,1)
        labeled = BasisFunctionalData(reshape(x,:,1),b; ids=1:8)
        labeled2 = BasisFunctionalData(reshape(y,:,1),b; ids=1:8)
        @test ids(StatsAPI.fit(FunctionalCCA,labeled,labeled2; ncan=1)) == collect(1:8)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,labeled,right; ncan=1)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,labeled,labeled2[8:-1:1]; ncan=1)
        @test_throws DimensionMismatch StatsAPI.fit(FunctionalCCA,left,right[1:7]; ncan=1)
    end

    @testset "fda 6.3.0 golden values" begin
        for dataset in ("growth", "weather"), kind in ("bspline", "fourier")
            interval = dataset == "growth" ? (1.,18.) : (0.,365.)
            b = kind == "bspline" ? BSplineBasis(4, interval, 5) : FourierBasis(interval,5)
            prefix = joinpath(@__DIR__, "data", "m6_fda_$(dataset)_$(kind)")
            left = BasisFunctionalData(readdlm(prefix*"_left.csv", ','; comments=true), b)
            right = BasisFunctionalData(readdlm(prefix*"_right.csv", ','; comments=true), b)
            for index in 1:3
                ref = fda_m6_table(prefix*"_$(index).csv")
                lambda, points = only(ref["lambda"]), vec(ref["grid"])
                # Orient using the default display grid, as in the R script.
                f = @inferred StatsAPI.fit(FPCA, left; lambda, ncomponents=2)
                @test eigvals(f) ≈ vec(ref["eigenvalues"])[1:2] rtol=1e-8
                @test f.all_eigenvalues ≈ vec(ref["eigenvalues"]) rtol=1e-8
                @test basis_matrix(eigenfunctions(f), points) ≈ ref["harmonics"] rtol=1e-8
                @test scores(f) ≈ ref["scores"] rtol=1e-8
                @test scree_data(f).individual[1:2] ≈ vec(ref["varprop"]) rtol=1e-8
                @test f.settings.original_harmonic_norms[1:2] ≈ vec(ref["harmonic_norms"]) rtol=1e-8
                @test diag(gram(eigenfunctions(f))) ≈ ones(2)
                @test vec(sum(abs2, scores(f); dims=1))/nobs(left) ≈
                    eigvals(f) ./ f.settings.original_harmonic_norms[1:2].^2 rtol=1e-8
                @test StatsAPI.predict(f,left) ≈ scores(f)
                @test reconstruct(f; grid=points).values ≈ evaluate(f.mean,points) .+ scores(f)*transpose(basis_matrix(eigenfunctions(f),points))
                cc = @inferred StatsAPI.fit(FunctionalCCA, left, right; lambda)
                @test cc.correlations ≈ vec(ref["correlations"]) rtol=1e-8
                @test cc.squared_correlations ≈ vec(ref["correlations"]).^2 rtol=1e-8
                @test transpose(evaluate(cc.weights1,points)) ≈ ref["weights1"] rtol=1e-8
                @test transpose(evaluate(cc.weights2,points)) ≈ ref["weights2"] rtol=1e-8
                @test cc.variates1 ≈ ref["variates1"] rtol=1e-8
                @test cc.variates2 ≈ ref["variates2"] rtol=1e-8
                @test vec(sum(abs2,cc.weights1.coefs; dims=2)) ≈ ones(2)
                @test StatsAPI.predict(cc,left,right).variates1 ≈ cc.variates1
            end
            refcv = readdlm(prefix*"_cv.csv", ','; comments=true)
            candidates = refcv[:,1]
            cv = LambdaCV(candidates)
            tuned = @inferred StatsAPI.fit(FPCA,left; lambda=cv,ncomponents=2)
            @test tuned.settings.lambda_selection.scores ≈ refcv[:,2] rtol=1e-8
            @test tuned.settings.lambda == candidates[argmin(refcv[:,2])]
            @test tuned.settings.lambda_selection.candidates == candidates
            tunedcc = @inferred StatsAPI.fit(FunctionalCCA,left,right; lambda=cv)
            @test tunedcc.settings.lambda_selection.scores ≈ refcv[:,3] rtol=1e-8
            @test tunedcc.settings.lambda == (candidates[argmax(refcv[:,3])],candidates[argmax(refcv[:,3])])
            @test tunedcc.settings.lambda_selection.candidates == candidates
            candidates[1] = 999
            @test cv.lambdas[1] == 0
        end
    end

    @testset "Existing FPCA routes and documented examples" begin
        grid = collect(range(0.,1.;length=21))
        latent = [2. 0; -2 0; 0 1; 0 -1; 1 1; -1 -1]
        functions = hcat(ones(21), grid .- .5)
        values = 1 .+ transpose(grid) .+ latent*transpose(functions)
        fd = GridFunctionalData(values,grid)
        dense = @inferred (data -> StatsAPI.fit(FPCA,data;
            method=:dense,ncomponents=2,max_k=2))(fd)
        @test (@inferred StatsAPI.predict(dense,fd)) ≈ scores(dense)
        @test (@inferred reconstruct(dense)).values ≈ values
        @test fitted_curve_data(dense,1).observed.values == values[1,:]
        @test eigenfunctions(dense) isa FPCBasis
        face = @inferred (data -> StatsAPI.fit(FPCA,data;method=:face,ncomponents=2,max_k=2,
            mean=(;grid,values=1 .+ grid),face=(;knots=4,lambda=1.0)))(fd)
        @test StatsAPI.predict(face,fd).scores ≈ scores(face)
        @test eigenfunctions(face) isa FPCBasis
        @test size(reconstruct(face).values) == size(values)
        irregular = to_irregular(fd)
        surface = functions*Diagonal([2.,.5])*transpose(functions)
        pace = @inferred StatsAPI.fit(FPCA,irregular;grid,ncomponents=2,max_k=2,
            mean=(;grid,values=1 .+ grid), covariance=(;grid,surface),noise_variance=.1)
        @test StatsAPI.predict(pace,irregular).scores ≈ scores(pace)
        @test eigenfunctions(pace) isa FPCBasis
        @test size(reconstruct(pace).values) == size(values)
        # Basis FPCA must not widen the established multivariate fit APIs.
        multivariate = MultivariateFunctionalData([fd, fd])
        marginal = @inferred (data -> StatsAPI.fit(FPCA,data;
            ncomponents=2,max_k=2))(multivariate)
        @test scores(marginal) == [scores(dense), scores(dense)]
        joint = @inferred StatsAPI.fit(MFPCA,multivariate;ncomponents=2,
            univariate=(;ncomponents=2,max_k=2))
        @test (@inferred StatsAPI.predict(joint,multivariate)) ≈ scores(joint)
        @test length((@inferred modes_of_variation(joint,1)).nodes) == 2
        growth = load_dataset(:growth;sample=true).data
        b = BSplineBasis(4,domain(growth),5)
        fitted = StatsAPI.fit(FPCA,to_basis(growth,b);lambda=.1,ncomponents=2)
        @test ncomponents(fitted) == 2
        weather = load_dataset(:canadian_weather;sample=true).data
        t, p = component(weather,1), component(weather,2)
        b = FourierBasis(domain(t),3)
        canonical = StatsAPI.fit(FunctionalCCA,to_basis(t,b),to_basis(p,b);lambda=1e4)
        @test length(canonical.squared_correlations) == 2
        @test all(isfinite,canonical.squared_correlations)
        f32 = @inferred StatsAPI.fit(FunctionalCCA,
            BasisFunctionalData(Float32.(latent),MonomialBasis((0.0f0,1.0f0),1)); ncan=1)
        @test f32.correlations isa Vector{Float32}
        x = BasisFunctionalData(reshape([-2.,-1,0,1,2],:,1),MonomialBasis((0.,1.),0))
        y = BasisFunctionalData(reshape([-1.,-2,0,2,1],:,1),FourierBasis((2.,4.),1))
        different = @inferred StatsAPI.fit(FunctionalCCA,x,y;ncan=1,lambda=(.1,.2),penalty=(0,0))
        @test domain(different.weights1) != domain(different.weights2)
        @test StatsAPI.predict(different,x,y).variates2 ≈ different.variates2
    end

    @testset "Validation" begin
        b = BSplineBasis(4,(0,1),4)
        C = [1. 0 2 3; 3 1 5 4; 7 2 4 1; 6 3 0 2; 9 1 7 1; 3 4 3 3]
        fd = BasisFunctionalData(C,b)
        f = StatsAPI.fit(FPCA,fd;ncomponents=2)
        @test f.settings.penalty == 2
        @test StatsAPI.fit(FPCA,BasisFunctionalData(C[:,1:2],BSplineBasis(1,(0,1),2))).settings.penalty == 0
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd[1])
        @test_throws ArgumentError StatsAPI.fit(FPCA,BasisFunctionalData(ones(3,4),b))
        for lambda in (-1, Inf, NaN, :auto)
            @test_throws ArgumentError StatsAPI.fit(FPCA,fd;lambda)
            @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;lambda)
        end
        for lambdas in (Float64[], [-1.], [NaN], [Inf], [0.,0.])
            @test_throws ArgumentError LambdaCV(lambdas)
        end
        for P in (-Matrix{Float64}(I,4,4), fill(NaN,4,4), [1. 2 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
            @test_throws ArgumentError StatsAPI.fit(FPCA,fd;penalty=P)
            @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;penalty=P)
        end
        @test_throws DimensionMismatch StatsAPI.fit(FPCA,fd;penalty=ones(3,3))
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;penalty=-1)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;center=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;max_k=0)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;max_k=5)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;ncomponents=0)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;ncomponents=AIC())
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;eigen_tol=NaN)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;grid=[0.,1.,.5])
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;grid=[0.,2.])
        cv = LambdaCV([0.,1.])
        @test_throws r"FPCA LambdaCV requires a positive integer ncomponents" StatsAPI.fit(FPCA,fd;lambda=cv)
        for rule in (FVE(0.9), AIC(), 0, true)
            @test_throws r"FPCA LambdaCV requires a positive integer ncomponents" StatsAPI.fit(FPCA,fd;lambda=cv,ncomponents=rule)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd[1:2];lambda=cv,ncomponents=1)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd[1:2];lambda=LambdaCV([0.,1.]))
        @test sprint(show, cv) == "LambdaCV([0.0, 1.0])"
        @test eltype(f.basis) == Float64
        @test nbasis(f.basis) == 2
        mono = BasisFunctionalData(C[:,1:2], MonomialBasis((0.,1.),1))
        @test StatsAPI.fit(FPCA,mono;ncomponents=1).settings.penalty == 0
        @test_throws r"explicit penalty" FunctionalDataAnalysis._default_fpca_penalty(f.basis)
        @test_throws r"LambdaCV candidate 0.5 failed in held-out curve 3" FunctionalDataAnalysis._basis_cv_error(
            PosDefException(1), 0.5, 3)
        @test_throws KeyError try
            throw(KeyError(:x))
        catch err
            FunctionalDataAnalysis._basis_cv_error(err, 0.5, 3)
        end
        cca = StatsAPI.fit(FunctionalCCA,fd;ncan=1)
        @test sprint(show, MIME"text/plain"(), cca) == sprint(show, cca)
        @test_throws ArgumentError StatsAPI.predict(f,to_grid(fd,f.grid))
        @test_throws ArgumentError StatsAPI.predict(f,BasisFunctionalData(C,BSplineBasis(3,(0,1),4)))
        @test_throws ArgumentError StatsAPI.predict(StatsAPI.fit(FPCA,to_grid(fd,f.grid);method=:dense),fd)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;ncan=0)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;ncan=5)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;center=:yes)
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,fd;lambda=(1.,2.,3.))
        @test_throws ArgumentError StatsAPI.fit(FunctionalCCA,BasisFunctionalData(ones(4,4),b);lambda=0)
        cc = StatsAPI.fit(FunctionalCCA,fd)
        @test cc.settings.ncan == 2
        @test canonical_correlations(StatsAPI.fit(FunctionalCCA,fd;ncomponents=2)) == canonical_correlations(cc)
        @test_throws ArgumentError StatsAPI.predict(cc,fd,BasisFunctionalData(C,BSplineBasis(3,(0,1),4)))
        P = penalty(b,2)
        stored = StatsAPI.fit(FPCA,fd;penalty=P)
        P[1,1] = 999
        @test stored.settings.penalty[1,1] != 999
        fd.basis.breakpoints[1] = -1
        @test domain(stored.mean) == (0.,1.)
    end
end
