import StatsAPI
import RecipesBase
using DelimitedFiles

@testset "Sparse FACE: face 0.1-8 and weighted L2 conversion" begin
    FDA = FunctionalDataAnalysis
    readinput(name) = readdlm(joinpath(@__DIR__, "data", name), ','; comments=true)
    function irregular(x)
        IrregularFunctionalData((id=Int.(x[:,1]),time=x[:,2],value=x[:,3]);
            id=:id,time=:time,value=:value)
    end
    function quantity(golden, name)
        x = golden[golden[:,1] .== name,:]
        reshape(Float64.(x[:,4]), Int(maximum(x[:,2])), Int(maximum(x[:,3])))
    end
    relerr(x,y) = norm(x-y)/norm(y)
    for dataset in ("sparse", "cd4"), case in ("default", "fixed")
        input = readinput(dataset=="sparse" ? "m3_sparse_input.csv" : "m6_sparse_face_cd4_input.csv")
        fd = irregular(input)
        if dataset == "cd4"
            sample = load_dataset(:cd4;sample=true).data
            @test all(fd[i] == sample[i] for i in 1:nobs(sample))
        end
        golden = readdlm(joinpath(@__DIR__, "data", "m6_sparse_face_$(dataset)_$(case).csv"), ',', Any; comments=true)
        q(name) = quantity(golden,name)
        options = case=="default" ? (;) : (; lambda=2.5,lambda_mean=.75)
        f = StatsAPI.fit(FPCA,fd;method=:face,face=options)
        subjects = Int.(vec(q("subjects")))
        @test f.method == :face
        @test f.grid ≈ vec(q("grid")) rtol=1e-14
        @test relerr(vec(f.mean.values),vec(q("mean"))) < 1e-8
        @test f.settings.lambda ≈ only(q("lambda")) rtol=1e-8
        @test f.settings.mean.lambda ≈ only(q("lambda_mean")) rtol=1e-8
        @test relerr(f.settings.gcv_scores,vec(q("gcv"))) < 1e-8
        @test relerr(f.settings.mean.gcv_scores,vec(q("mean_gcv"))) < 1e-8
        @test noise_variance(f) ≈ only(q("sigma2")) rtol=1e-8
        @test !f.settings.noise_floor_applied
        @test f.settings.raw_noise_variance ≈ only(q("raw_sigma2")) rtol=1e-8
        @test relerr(f.all_eigenvalues,vec(q("eigenvalue"))) < 1e-8
        @test relerr(f.basis.values,q("eigenfunction")) < 1e-8
        @test relerr(scores(f)[subjects,:],q("score")) < 1e-8
        @test relerr(reshape(score_covariances(f)[:,:,subjects],ncomponents(f)^2,:),q("score_covariance")) < 1e-8
        @test relerr(reconstruct(f).values[subjects[1:3],:],q("fitted")) < 1e-8
        @test relerr(fitted_bands(f;curves=subjects[1:3]).variance,q("variance")) < 1e-8
        @test f.basis.values'*(f.weights .* f.basis.values) ≈ I atol=1e-12
        @test f.settings.knot_sequence ≈ vec(q("knots")) rtol=1e-14
        @test f.covariance.coefficient_root*f.covariance.coefficient_root' ≈ q("theta") rtol=1e-8
        @test Matrix(f.covariance) ≈ covariance_data(f).values
        @test StatsAPI.predict(f,fd).scores == scores(f)
        @test StatsAPI.predict(f,fd).covariances == score_covariances(f)
        @test f.settings.gcv_scores[argmin(f.settings.gcv_scores)] <=
            f.settings.gcv_scores[argmin(abs.(f.settings.gcv_lambdas .- only(q("lambda"))))]
        # R's native prediction uses every positive direction, irrespective of
        # pve. Retaining all functional PCs reproduces it at observation times.
        full = StatsAPI.fit(FPCA,fd;method=:face,ncomponents=FVE(1),face=options)
        predicted, variance = Float64[], Float64[]
        for i in subjects
            t,_ = fd[i]
            phi = basismatrix(full.covariance.basis,t)*full.settings.eigenfunction_coefficients
            append!(predicted, FDA._face_sparse_mean_values(full.settings.mean,t)+phi*scores(full)[i,:])
            append!(variance,diag(phi*score_covariances(full)[:,:,i]*phi'))
        end
        @test relerr(predicted,vec(q("source_prediction"))) < 1e-8
        @test relerr(variance,vec(q("source_variance"))) < 1e-8
        # Plot data and recipes consume ordinary FPCA fields without a new path.
        @test fitted_curve_data(f,1).method == :face
        @test mean_data(f).values ≈ vec(f.mean.values)
        @test scree_data(f).eigenvalues == f.all_eigenvalues
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol,Any}(), FDA._plot_payload(Val(:scree),scree_data(f))))
    end

    @testset "Prediction, settings, copying, validation" begin
        x = readinput("m3_sparse_input.csv")
        fd = irregular(x)
        f = @inferred ((d)->StatsAPI.fit(FPCA,d;method=:face,face=(;lambda=2.5,lambda_mean=.75)))(fd)
        heldout = irregular(readinput("m3_sparse_heldout.csv"))
        predicted = StatsAPI.predict(f,heldout)
        # Independent observation-space BLUP, with exact splines at new times.
        for i in 1:nobs(heldout)
            t,y = heldout[i]
            B = basismatrix(f.covariance.basis,t)
            Phi = B*f.settings.eigenfunction_coefficients
            theta = f.covariance.coefficient_root*f.covariance.coefficient_root'
            V = B*theta*B' + noise_variance(f)*I
            residual = y-FDA._face_sparse_mean_values(f.settings.mean,t)
            cross = Phi*Diagonal(eigvals(f))
            @test predicted.scores[i,:] ≈ cross'*(V\residual) rtol=1e-10
            @test predicted.covariances[:,:,i] ≈ Diagonal(eigvals(f))-cross'*(V\cross) rtol=1e-10
        end
        griddata = GridFunctionalData(reconstruct(f).values[1:2,:],f.grid)
        @test StatsAPI.predict(f,griddata).scores ≈ StatsAPI.predict(f,to_irregular(griddata)).scores
        supplied = (; grid=[0.,1.],values=[2.,2.3])
        withmean = StatsAPI.fit(FPCA,fd;method=:face,mean=supplied,face=(;lambda=2.5))
        @test vec(withmean.mean.values) ≈ 2 .+ .3 .* withmean.grid
        @test withmean.settings.mean.source == :supplied
        uncentered = StatsAPI.fit(FPCA,fd;method=:face,face=(;center=false,lambda=2.5))
        @test all(iszero,uncentered.mean.values)
        # Nonuniform work-grid weights determine normalization and eigenvalues.
        uneven = StatsAPI.fit(FPCA,fd;method=:face,grid=collect(range(0,1;length=101)).^1.2,
            ncomponents=2,face=(;lambda=2.5))
        @test uneven.basis.values'*(uneven.weights .* uneven.basis.values) ≈ I atol=1e-12
        @test uneven.settings.grid_source == :supplied
        fd32 = IrregularFunctionalData(Float32.(fd.times),Float32.(fd.values),fd.offsets)
        f32 = @inferred ((d)->StatsAPI.fit(FPCA,d;method=:face,ncomponents=2,
            face=(;lambda=2.5,lambda_mean=.75)))(fd32)
        @test eltype(scores(f32)) == Float32
        @test eltype(eigvals(f32)) == Float32
        @test f32.basis.values'*(f32.weights .* f32.basis.values) ≈ I atol=1e-5
        # A deterministic example with a negative joint diagonal coefficient:
        # the positive floor must survive fitting and prediction.
        times = [collect(range(0,1;length=5)) for i in 1:8]
        values = [[sin(.4i)*(1+t)+cos(.9i)*cospi(t)+.1sin(i+5t) for t in ti]
            for (i,ti) in enumerate(times)]
        noiseless = StatsAPI.fit(FPCA,IrregularFunctionalData(times,values);method=:face,
            ncomponents=1,face=(;knots=3,center=false,lambda=.01))
        @test noiseless.settings.raw_noise_variance < 0
        @test noise_variance(noiseless) == noiseless.settings.noise_floor
        @test noise_variance(noiseless) == 0.025 * noiseless.settings.noise_floor_scale
        @test noiseless.settings.noise_floor_scale > 0
        @test noiseless.settings.noise_floor_relative == 0.025
        @test noiseless.settings.noise_floor_scale_source == :maximum_fitted_pointwise_variance
        @test noiseless.settings.noise_floor_applied
        @test noiseless.settings.noise_truncation == :positive_floor
        for value_scale in (1e-6, 1e6)
            scaled = StatsAPI.fit(FPCA,
                IrregularFunctionalData(times, [value_scale .* y for y in values]);
                method=:face, ncomponents=1, face=(;knots=3,center=false,lambda=.01))
            @test scaled.settings.noise_floor_applied
            @test scaled.settings.raw_noise_variance ≈ value_scale^2 * noiseless.settings.raw_noise_variance rtol=1e-8
            @test noise_variance(scaled) ≈ value_scale^2 * noise_variance(noiseless) rtol=1e-8
            @test reconstruct(scaled).values ≈ value_scale .* reconstruct(noiseless).values rtol=1e-8
            @test score_covariances(scaled) ≈ value_scale^2 .* score_covariances(noiseless) rtol=1e-8
        end
        single = IrregularFunctionalData([[.37]],[[1.]];domain=(0.,1.))
        @test all(isfinite,StatsAPI.predict(noiseless,single).scores)
        rank_deficient = IrregularFunctionalData([collect(range(0,1;length=20))],[ones(20)])
        predicted = StatsAPI.predict(noiseless,rank_deficient)
        @test all(isfinite, predicted.scores)
        @test all(isfinite, predicted.covariances)
        # Irregular margins pass through the existing multivariate interfaces.
        multivariate = MultivariateFunctionalData([fd,irregular(x)])
        marginal = StatsAPI.fit(FPCA,multivariate;method=:face,ncomponents=2,
            face=(;lambda=2.5,lambda_mean=.75))
        @test all(f.settings.route == :sparse for f in marginal.fits)
        @test all(p.scores ≈ s for (p,s) in zip(StatsAPI.predict(marginal,multivariate),scores(marginal)))
        snapshot = deepcopy(f)
        fd.values[1] += 10
        fd.times[1] += .001
        @test f.training_data.values == snapshot.training_data.values
        @test f.training_data.times == snapshot.training_data.times
        @test f.settings == snapshot.settings
        fd = irregular(x)
        for name in ("lambda", "knots", "center", "mean", "criterion", "ridge", "two_step", "quadrature")
            @test occursin(name,sprint(show,f))
        end
        for options in ((;knots=0),(;p=-1),(;m=-1),(;m=20),(;center=1),
                (;lambda=-1),(;lambda=Inf),(;lambda=true),(;lambda_mean=-1),
                (;loglambda=[1.,1.]),(;loglambda=[1000.]),(;loglambda=Float64[]),
                (;mean_loglambda=[NaN]),(;two_step=true),(;alpha=1),(;unknown=true))
            @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,face=options)
        end
        for selector in (AIC(),BIC(),0,true,:invalid)
            @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,ncomponents=selector)
        end
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,maxK=11)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,ncomponents=11)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,noise_variance=1)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,presmooth=:invalid)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,mean_bandwidth=.1)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,grid=[.1,.9])
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,grid=[0.,1.])
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,face=[1])
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;face=(;lambda=1))
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,mean=supplied,face=(;center=false))
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,mean=supplied,face=(;lambda_mean=1))
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,mean=supplied,face=(;mean_loglambda=[1.]))
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd;method=:face,face=(;center=false,mean_loglambda=[1.]))
        @test_throws DimensionMismatch StatsAPI.fit(FPCA,fd;method=:face,mean=(;grid=[0.,1.],values=[1.]))
        outside = IrregularFunctionalData([[-.01,.5]],[ones(2)];domain=(-.1,1.))
        @test_throws ArgumentError StatsAPI.predict(f,outside)
        @test_throws ArgumentError fitted_bands(f;coverage=:band)
        @test_throws ArgumentError StatsAPI.fit(FPCA,fd[1:1];method=:face)
    end
end

@testset "Low-noise sparse regression: cell 17 observation design" begin
    # Like scaling cell 17: 100 curves, 2:10 observations and a 101-point grid.
    # Deterministic smooth curves avoid reliance on version-specific RNGs.
    # Before the floor, sparse FACE failed on curve 7 with this design.
    curve(i, t) = 1+t+sin(.4i)*(1+t)+cos(.9i)*cospi(t)+.1sin(i+5t)
    times = [collect(range(0.0, 1.0; length=2+mod(i-1, 9))) for i in 1:100]
    data = IrregularFunctionalData(times,
        [[curve(i, t) for t in ti] for (i, ti) in enumerate(times)])
    grid = collect(range(0.0, 1.0; length=101))
    truth = [curve(i, t) for i in 1:100, t in grid]
    observed_range = maximum(data.values) - minimum(data.values)
    for method in (:pace, :face)
        model = StatsAPI.fit(FPCA, data; method, grid)
        @test model.settings.raw_noise_variance < 0
        @test model.settings.noise_floor_applied
        @test noise_variance(model) == model.settings.noise_floor
        @test noise_variance(model) == 0.025 * model.settings.noise_floor_scale
        reconstructed = reconstruct(model).values
        @test all(isfinite, reconstructed)
        @test all(isfinite, score_covariances(model))
        @test sum(abs2.(reconstructed-truth) .* transpose(model.weights))/100 < 0.1
        # Control individual spikes as well as the integrated average error.
        @test maximum(abs, reconstructed-truth) < observed_range
        @test minimum(reconstructed) >= minimum(data.values) - observed_range/2
        @test maximum(reconstructed) <= maximum(data.values) + observed_range/2
        prediction = StatsAPI.predict(model, data)
        @test prediction.scores == scores(model)
        @test prediction.covariances == score_covariances(model)
        displayed = sprint(show, model)
        @test occursin("noise_floor_applied = true", displayed)
        @test occursin("raw_noise_variance", displayed)
        for choice in ("noise_floor_relative", "noise_floor_scale", "noise_floor_scale_source")
            @test occursin(choice, displayed)
        end
    end
end
