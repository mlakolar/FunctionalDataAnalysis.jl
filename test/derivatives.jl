using Test, LinearAlgebra, Random, DelimitedFiles
using FunctionalDataAnalysis
import StatsAPI
import KernelSmoothing
import RecipesBase

@testset "Exact basis derivatives" begin
    t = [0.0, .04, .2, .37, .71, 1.0]
    b = MonomialBasis((0., 1.), 3)
    coefs = [1. 2 -3 4; 0 0 0 1]
    fd = BasisFunctionalData(coefs, b; ids=[:a,:b], metadata=(; source=[1,2]))
    d = @inferred derivative(fd)
    expected = permutedims(hcat(2 .- 6t .+ 12t.^2, 3t.^2))
    @test (@inferred evaluate(d,t)) ≈ expected atol=1e-13
    @test d.basis isa DerivativeBasis
    @test d.basis.order == 1 && nbasis(d.basis) == 4
    @test ids(d) == ids(fd) && d.metadata == fd.metadata
    @test d.coefs !== fd.coefs && d.metadata.source !== fd.metadata.source
    @test derivative(fd,0).coefs == fd.coefs
    @test evaluate(derivative(d),t)[2,:] ≈ 6t
    @test derivative(d).basis == DerivativeBasis(b,2)
    @test_throws ArgumentError DerivativeBasis(DerivativeBasis(b,typemax(Int)),1)
    @test gram(d.basis)[4,4] ≈ 9/5
    @test penalty(d.basis,1)[4,4] ≈ 12
    @test penalty(d.basis,3) == zeros(4,4)
    @test basis_matrix(d.basis,t;derivative=1) ≈ basis_matrix(b,t;derivative=2)
    @test domain(d.basis) == (0.,1.)
    @test d.basis == deepcopy(d.basis)
    @test isequal(d.basis,deepcopy(d.basis))
    @test hash(d.basis) == hash(deepcopy(d.basis))
    @test occursin("order=1",sprint(show,d.basis))
    # Independent analytic Fourier derivatives and their integrated squares.
    f = FourierBasis((0.,1.),3)
    ffd = BasisFunctionalData([1. 2 3],f)
    for nu in 1:3
        df = derivative(ffd,nu)
        phi = nu==1 ? sqrt(2)*2pi*(2cos.(2pi*t)-3sin.(2pi*t)) :
            nu==2 ? -sqrt(2)*(2pi)^2*(2sin.(2pi*t)+3cos.(2pi*t)) :
            -sqrt(2)*(2pi)^3*(2cos.(2pi*t)-3sin.(2pi*t))
        @test vec(evaluate(df,t)) ≈ phi atol=1e-10
        @test gram(df.basis) ≈ Diagonal([0.,(2pi)^(2nu),(2pi)^(2nu)])
    end
    # x^3 is in the cubic spline span; coefficients come from an independent fit.
    spline = BSplineBasis(4,[0.,.17,.6,1.])
    dense = collect(range(0.,1.;length=31))
    sfd = to_basis(GridFunctionalData(reshape(dense.^3,1,:),dense),spline)
    for nu in 1:4
        ds = derivative(sfd,nu)
        expected = nu==1 ? 3t.^2 : nu==2 ? 6t : nu==3 ? fill(6.,length(t)) : zeros(length(t))
        @test vec(evaluate(ds,t)) ≈ expected atol=1e-10
    end
    ds = derivative(sfd)
    @test only(sfd.coefs*gram(ds.basis)*transpose(sfd.coefs)) ≈ 9/5 atol=1e-12
    @test only(sfd.coefs*penalty(ds.basis,1)*transpose(sfd.coefs)) ≈ 12 atol=1e-11
    @test curve_data(ds;grid=t).values[1] ≈ 3t.^2
    for nu in (-1,1.5,NaN,big(typemax(Int))+1)
        @test_throws ArgumentError derivative(fd,nu)
    end
    @test_throws ArgumentError basis_matrix(DerivativeBasis(b,typemax(Int)),t;derivative=1)
    @test_throws ArgumentError evaluate(d,[-.1])
end

@testset "Polynomial curve and mean derivatives" begin
    t = [0.,.01,.04,.1,.17,.25,.37,.5,.65,.78,.9,1.]
    y = 1 .+ 2t .- 3t.^2 .+ 4t.^3
    fd = GridFunctionalData(permutedims(hcat(y,y)),t;ids=[:a,:b])
    ir = to_irregular(fd)
    for raw in (fd,ir)
        @test_throws ArgumentError derivative(raw)
        for nu in 1:2
            method = @inferred LocalPolynomial(bandwidth=1.2,derivative=nu,degree=3)
            d = @inferred smooth_curves(raw,method;grid=t)
            truth = nu==1 ? 2 .- 6t .+ 12t.^2 : -6 .+ 24t
            @test d.values ≈ permutedims(hcat(truth,truth)) atol=1e-10
            @test d.metadata.derivative == nu && d.metadata.degree == 3
            @test ids(d) == ids(fd)
            m = @inferred StatsAPI.fit(SmoothedMean,raw;grid=t,bandwidth=1.2,derivative=nu,degree=3)
            @test m.values ≈ truth atol=1e-10
            @test m.derivative == nu && m.degree == 3
            @test mean_data(m).values ≈ truth atol=1e-10
            @test evaluate(m,t) ≈ truth atol=1e-10
            @test occursin("derivative=$nu",sprint(show,m))
            @test occursin("derivative=$nu",sprint(show,method))
        end
    end
    @test LocalPolynomial(bandwidth=.5).derivative == 0
    @test LocalPolynomial(bandwidth=.5,derivative=2).degree == 3
    @test StatsAPI.fit(SmoothedMean,fd;grid=t,bandwidth=1.,derivative=1).degree == 2
    for nu in (-1,1.5,NaN,nothing,"first",big(typemax(Int))+1)
        @test_throws ArgumentError LocalPolynomial(bandwidth=.5,derivative=nu)
        @test_throws ArgumentError StatsAPI.fit(SmoothedMean,fd;grid=t,bandwidth=.5,derivative=nu)
    end
    @test_throws ArgumentError LocalPolynomial(bandwidth=.5,derivative=2,degree=1)
    @test_throws ArgumentError StatsAPI.fit(SmoothedMean,fd;grid=t,bandwidth=CurveCV([.5,1.];folds=2),derivative=1)
    @test_throws ArgumentError smooth_curves(fd,LocalPolynomial(bandwidth=.0001,derivative=1);grid=t)
    @test_throws ArgumentError StatsAPI.fit(SmoothedMean,fd;grid=t,bandwidth=.0001,derivative=1)
    # No extrapolation and no silent removal of masked observations.
    masked = GridFunctionalData([missing 1. 2 3 missing; 0 1 2 3 4.],[0.,.25,.5,.75,1.])
    dm = smooth_curves(masked,LocalPolynomial(bandwidth=1.,derivative=1);grid=masked.grid)
    @test !dm.observed[1,1] && !dm.observed[1,end]
    @test dm.values[1,2:4] ≈ fill(4.,3)
end

@testset "Derivative means cannot center curves" begin
    grid = [0., .1, .25, .45, .7, 1.]
    fd = GridFunctionalData([i + t + t^2 for i in 1:3, t in grid], grid)
    for raw in (fd, to_irregular(fd))
        m = StatsAPI.fit(SmoothedMean, raw; grid, bandwidth=2.)
        covariance = (; grid, surface=ones(length(grid), length(grid)))
        mfd = MultivariateFunctionalData([raw, raw])
        @test (@inferred StatsAPI.fit(FPCA, raw; method=:pace, grid, mean=m,
            covariance, noise_variance=.1, ncomponents=1, max_k=1)).method === :pace
        @test (@inferred StatsAPI.fit(SmoothedCovariance, raw, m;
            grid, bandwidth=2.)) isa SmoothedCovariance
        @test (@inferred StatsAPI.fit(SmoothedCrossCovariance, raw, raw, m, m;
            grid1=grid, bandwidth=2.)) isa SmoothedCrossCovariance
        for nu in 1:2
            dm = StatsAPI.fit(SmoothedMean, raw;
                grid, bandwidth=2., derivative=nu, degree=3)
            err = ArgumentError("mean has derivative=$nu; a centering mean must have derivative=0")
            @test_throws err StatsAPI.fit(FPCA, raw; method=:pace, grid, mean=dm,
                covariance, noise_variance=.1, ncomponents=1, max_k=1)
            @test_throws err StatsAPI.fit(FPCA, raw; method=:pace, grid, mean=dm,
                covariance_bandwidth=2., noise_variance=.1, ncomponents=1, max_k=1)
            @test_throws err StatsAPI.fit(SmoothedCovariance, raw, dm;
                grid, bandwidth=2.)
            for means in ([dm, m], [m, dm])
                @test_throws err StatsAPI.fit(SmoothedCrossCovariance, raw, raw,
                    means...; grid1=grid, bandwidth=2.)
                @test_throws err cross_covariances(mfd; means, bandwidth=2.)
            end
            @test_throws err cross_covariances(MultivariateFunctionalData([raw]);
                means=[dm], bandwidth=2.)
            @test_throws err FunctionalDataAnalysis._pace_mean_values(dm, grid)
            @test_throws err FunctionalDataAnalysis._covariance_mean_values(dm, grid)
        end
    end
end

function derivative_fixture(design)
    read(name) = readdlm(joinpath(@__DIR__,"data",name),',';comments=true)
    data = read("m3_$(design)_input.csv")
    n = Int(maximum(data[:,1]))
    times = [Float64.(data[data[:,1].==i,2]) for i in 1:n]
    values = [Float64.(data[data[:,1].==i,3]) for i in 1:n]
    if design=="dense"
        fd = GridFunctionalData(permutedims(hcat(values...)),first(times);ids=collect(1:n))
        f = StatsAPI.fit(FPCA,fd;ncomponents=3,max_k=5)
    else
        m = read("m3_sparse_mean.csv"); grid=Float64.(m[:,1])
        fd = IrregularFunctionalData(times,values;ids=collect(1:n))
        f = StatsAPI.fit(FPCA,fd;grid,mean=(;grid,values=m[:,2]+.2grid.^2),
            covariance=(;grid,surface=read("m3_sparse_surface.csv")),
            noise_variance=.12,ncomponents=3,max_k=5)
    end
    fd,f
end

@testset "FPCA derivative reference and interface" begin
    golden = readdlm(joinpath(@__DIR__,"data","m5_fdapace_der.csv"),',',Any;comments=true)
    relative(actual,expected) = norm(actual-expected)/norm(expected)
    for design in ("dense","sparse")
        fd,f = derivative_fixture(design)
        ref(method,quantity) = Float64.(golden[(golden[:,1].==design).&(golden[:,2].==method).&(golden[:,3].==quantity),6])
        # Native DPC re-estimates the pooled mean even with a supplied mean.
        # Compare that quantity directly, independently of the DPC score prior.
        native_mean = StatsAPI.fit(SmoothedMean, fd; grid=f.grid, bandwidth=.12,
            kernel=KernelSmoothing.GaussianKernel(), derivative=1, degree=2)
        @test relative(native_mean.values, ref("native_DPC", "mean")) < 1e-8
        for (method,nu,key) in ((:FPC,1,"FPC_order1"),(:FPC,2,"FPC_order2"),(:DPC,1,"DPC"))
            d = method===:FPC ? derivative(f,nu;mean_bandwidth=.12,covariance_bandwidth=.15) :
                derivative(f,nu;method,mean_bandwidth=.12,covariance_bandwidth=.15,ncomponents=3)
            @test d isa FPCADerivative && d.method==method && d.order==nu
            @test relative(vec(d.mean.values),ref(key,"mean")) < 1e-8
            @test relative(d.basis.values,reshape(ref(key,"functions"),length(f.grid),:)) < 1e-8
            @test relative(eigvals(d),ref(key,"eigenvalues")) < 1e-8
            @test relative(scores(d),reshape(ref(key,"scores"),nobs(f),:)) < 1e-8
            @test relative(reconstruct(d).values,reshape(ref(key,"curves"),nobs(f),:)) < 1e-8
            @test ids(d)==ids(f) && ids(reconstruct(d))==ids(f)
            @test d.scores !== f.scores && d.basis.values !== f.basis.values
            @test ids(reconstruct(d,copy(scores(d)))) === nothing
            @test (@inferred mean_data(d)).values == vec(d.mean.values)
            cd = @inferred curve_data(d;curves=[2,1])
            @test cd.values[1] == reconstruct(d).values[2,:]
            @test cd.representation==:derivative && cd.curves==[2,1]
            @test reconstruct(d;grid=[.05,.45,.85]).grid == [.05,.45,.85]
            if method===:FPC
                @test scores(d)==scores(f) && eigvals(d)==eigvals(f)
                @test score_covariances(d)==score_covariances(f)
            else
                @test relative(d.settings.derivative_covariance,
                    reshape(ref(key,"covariance"),length(f.grid),:)) < 1e-8
                @test transpose(d.basis.values)*(d.weights.*d.basis.values) ≈ I atol=1e-12
                if design=="sparse"
                    for i in 1:nobs(f)
                        @test relative(score_covariances(d)[:,:,i],
                            reshape(ref(key,"score_covariance_$i"),3,3)) < 1e-8
                        @test minimum(eigvals(Symmetric(score_covariances(d)[:,:,i]))) > -1e-10
                    end
                    # Native DPC is a different, recorded quantity, not the aligned oracle.
                    @test relative(d.eigenvalues,ref("native_DPC","eigenvalues")[1:3]) > 1e-3
                else
                    # On a common grid, differentiating the stored sample mean
                    # also equals native DPC's pooled derivative mean.
                    @test relative(vec(d.mean.values), ref("native_DPC", "mean")) < 1e-8
                    @test score_covariances(d) === nothing
                    @test eigvals(d) ≈ vec(sum(abs2,scores(d);dims=1))/(nobs(d)-1) rtol=1e-10
                end
            end
            @test occursin("method=$method",sprint(show,d))
            @test occursin(r"degree\s*=",sprint(show,d))
            @test_throws DimensionMismatch reconstruct(d,zeros(2,2))
            @test_throws ArgumentError reconstruct(d,fill(NaN,2,3))
            @test_throws ArgumentError reconstruct(d;grid=[-.1,.5])
            @test_throws ArgumentError curve_data(d;curves=[1,1])
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(),d)
            @test !isempty(recipes)
        end
        d = @inferred derivative(f)
        @test d.method===:FPC && d.settings.degree==1
        @test reconstruct(derivative(f; eigen_tol=1e-10, sign_tol=1e-12)).values ==
            reconstruct(d).values
        dpc = derivative(f; method=:DPC, eigen_tol=1e-9, sign_tol=1e-11)
        @test dpc.settings.eigen_tol == 1e-9 && dpc.settings.sign_tol == 1e-11
        expected_h = design=="dense" ? .05 : .1
        @test d.mean_bandwidth ≈ expected_h && d.covariance_bandwidth ≈ expected_h
        @test derivative(f,2).mean_bandwidth ≈ 2expected_h
        @test derivative(f;method=:DPC).settings.selection.threshold == .9999
        @test derivative(f;bandwidth=.2).mean_bandwidth==.2
        for opts in ((;method=:FPC1),(;method=:bad),(;method=:DPC,degree=0),
                (;mean_bandwidth=.1),(;bandwidth=.1,covariance_bandwidth=.1),
                (;ncomponents=2),(;eigen_tol=1e-9),(;sign_tol=1e-11),
                (;method=:DPC,eigen_tol=-1.),
                (;method=:DPC,ncomponents=100),(;bandwidth=NaN))
            @test_throws ArgumentError derivative(f;opts...)
        end
        @test_throws ArgumentError derivative(f,0)
        @test_throws ArgumentError derivative(f,3)
        @test_throws ArgumentError derivative(f,2;method=:DPC)
    end
end

@testset "Node derivatives and nonuniform DPC" begin
    t = [0.,.03,.1,.2,.4,.65,.8,.9,1.]
    v = collect(-3.:3.) .* transpose(t.^2 .+ t) .+ transpose(1 .+ t)
    fd = GridFunctionalData(v,t;ids=collect(1:7))
    f = StatsAPI.fit(FPCA,fd;ncomponents=1,max_k=1)
    d = @inferred (f -> derivative(f;method=:DPC,bandwidth=2.,ncomponents=1))(f)
    # The quadratic local derivative is exact on this nonuniform grid.
    @test reconstruct(d).values ≈ collect(-3.:3.) .* transpose(2t .+ 1) .+ 1 atol=1e-10
    mfd = MultivariateFunctionalData([fd,fd])
    for mode in (:marginal,:pooled)
        mf = StatsAPI.fit(FPCA,mfd;basis=mode,ncomponents=1,max_k=1)
        md = derivative(mf;bandwidth=[.2,.3])
        @test md isa MultivariateFPCADerivative
        @test nobs(md)==7 && ncomponents(md)==[1,1]
        @test component(md,1).mean_bandwidth==.2 && component(md,2).mean_bandwidth==.3
        @test ids(reconstruct(md))==ids(mfd)
        @test mean_data(md).nodes[1] == mean_data(component(md,1))
        @test length(curve_data(md;curves=2).nodes)==2
        @test component(reconstruct(md;grid=[.1,.5,.9]),1).grid == [.1,.5,.9]
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol,Any}(),md))
        @test_throws DimensionMismatch derivative(mf;bandwidth=[.1])
        @test_throws ArgumentError component(md,3)
        dd = derivative(mf;method=:DPC,bandwidth=2.)
        @test all(node -> node.method===:DPC,dd.nodes)
        @test component(reconstruct(dd),1).values ≈ reconstruct(d).values atol=1e-10
        @test scores(md)[1] === scores(component(md,1))
        @test_throws DimensionMismatch reconstruct(md;grid=[[.1,.5]])
        @test_throws ArgumentError reconstruct(md;grid=[nothing,nothing])
        if mode===:pooled
            @test mf.training_data !== mfd
            @test component(mf.training_data,1).values !== component(mfd,1).values
            @test component(dd,1).settings.original_settings.source_basis===:pooled
        end
    end
    # Distinct node covariances must survive a one-component pooled truncation.
    fd2 = GridFunctionalData(collect(-3.:3.) .* transpose(t.^3 .- .5t) .+
        transpose(1 .+ t.^2), t; ids=collect(1:7))
    mixed = MultivariateFunctionalData([fd, fd2])
    pooled = StatsAPI.fit(FPCA, mixed; basis=:pooled, ncomponents=1, max_k=2)
    dd = derivative(pooled; method=:DPC, bandwidth=2., degree=3, ncomponents=1)
    truth2 = collect(-3.:3.) .* transpose(3t.^2 .- .5) .+ transpose(2t)
    @test component(reconstruct(dd), 2).values ≈ truth2 atol=1e-10
    @test component(reconstruct(dd), 1).values ≈ reconstruct(d).values atol=1e-10
    component(mixed, 2).values[1, 1] = 999
    @test component(reconstruct(derivative(pooled; method=:DPC,
        bandwidth=2., degree=3, ncomponents=1)), 2).values ≈ truth2 atol=1e-10
    # Different designs and methods retain each node's CE uncertainty.
    densefd, _ = derivative_fixture("dense")
    sparsefd, sparsefit = derivative_fixture("sparse")
    mixed_design = MultivariateFunctionalData([densefd, sparsefd[1:nobs(densefd)]])
    sg = sparsefit.grid
    mf = StatsAPI.fit(FPCA, mixed_design; ncomponents=3, max_k=5,
        grid=[nothing, sg], mean=[nothing, (; grid=sg, values=vec(sparsefit.mean.values))],
        covariance=[nothing, (; grid=sg, surface=sparsefit.covariance)],
        noise_variance=[nothing, .12])
    md = derivative(mf; method=[:FPC, :DPC], bandwidth=[.12, .15], ncomponents=[nothing, 3])
    @test component(md, 1).method===:FPC && component(md, 2).method===:DPC
    @test score_covariances(md)[1] === nothing
    @test score_covariances(md)[2] ≈ score_covariances(derivative(component(mf, 2);
        method=:DPC, bandwidth=.15, ncomponents=3))
    @test ids(reconstruct(md)) == ids(mixed_design)
end

@testset "Fourier derivative recovery" begin
    grid = collect(range(0.,1.;length=101))
    funcs = fourier_eigenfunctions((0.,1.),3)[2:3]
    lambda = [2.,.5]
    for design in (:dense,:sparse)
        sampling = design===:dense ? RegularSampling(grid) : SparseSampling((0.,1.),6:10)
        sim = simulate_kl(MersenneTwister(512),250;mean=t->2+t,eigenfunctions=funcs,eigenvalues=lambda,
            sampling,noise_sd=design===:dense ? 0. : .05)
        if design===:dense
            f = StatsAPI.fit(FPCA,sim.data;ncomponents=2,max_k=2)
        else
            phi = hcat([fun.(grid) for fun in funcs]...)
            # Supplied covariance isolates derivative/CE recovery from smoothing bias.
            f = StatsAPI.fit(FPCA,sim.data;grid,ncomponents=2,max_k=2,
                mean=(;grid,values=2 .+ grid),covariance=(;grid,surface=phi*Diagonal(lambda)*transpose(phi)),
                noise_variance=.05^2)
        end
        d = derivative(f;bandwidth=.015,degree=2)
        analytic = hcat(sqrt(2)*2pi*cos.(2pi*grid),-sqrt(2)*2pi*sin.(2pi*grid))
        for k in 1:2
            estimated = d.basis.values[:,k]
            @test min(norm(sqrt.(d.weights).*(estimated-analytic[:,k])),
                norm(sqrt.(d.weights).*(estimated+analytic[:,k]))) / (2pi) < .12
        end
        truth = sim.scores*transpose(analytic) .+ 1
        recovered = reconstruct(d).values
        @test norm(recovered-truth)/norm(truth) < .12
        dpc = derivative(f; method=:DPC, bandwidth=.015)
        @test norm(reconstruct(dpc).values-truth)/norm(truth) < .12
    end
end
