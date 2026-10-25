import StatsAPI
import KernelSmoothing
using DelimitedFiles

_m6_read(name) = readdlm(joinpath(@__DIR__, "data", "m6_$(name).csv"), ','; comments=true)
_m6_relative(a, b) = norm(a-b)/norm(b)

@testset "FSVD dense reference and quadrature" begin
    l, r = _m6_read("dense_left"), _m6_read("dense_right")
    x = GridFunctionalData(_m6_read("dense_input1"), l[:,1]; ids=1:18)
    y = GridFunctionalData(_m6_read("dense_input2"), r[:,1]; ids=1:18)
    f = @inferred StatsAPI.fit(FSVD, x, y; ncomponents=2, max_k=2)
    @test (@inferred nobs(f)) == nobs(x)
    @test (@inferred ncomponents(f)) == 2
    @test (@inferred svdvals(f)) === f.singular_values
    @test (@inferred left_functions(f)) === f.left_functions
    @test (@inferred right_functions(f)) === f.right_functions
    @test (@inferred left_scores(f)) === f.left_scores
    @test (@inferred right_scores(f)) === f.right_scores
    @test (@inferred fve(f)) === f.fve
    independent_scores = copy(left_scores(f))
    independent_scores[1,1] += 1
    @test independent_scores[1,1] != left_scores(f)[1,1]
    expected = _m6_read("dense_values")
    @test _m6_relative(f.singular_values, expected[:,1]) < 1e-8
    @test _m6_relative(f.left_functions, l[:,2:end]) < 1e-8
    @test _m6_relative(f.right_functions, r[:,2:end]) < 1e-8
    @test _m6_relative(f.left_scores, _m6_read("dense_scores1")) < 1e-8
    @test _m6_relative(f.right_scores, _m6_read("dense_scores2")) < 1e-8
    @test _m6_relative(f.fve, expected[:,2]) < 1e-8
    @test f.cross_covariance.denominator == 16
    @test f.cross_covariance.surface ≈ (x.values .- sum(x.values;dims=1)/18)' *
        (y.values .- sum(y.values;dims=1)/18)/16
    @test f.left_functions' * Diagonal(x.weights) * f.left_functions ≈ I
    @test f.right_functions' * Diagonal(y.weights) * f.right_functions ≈ I
    @test f.settings.effective_fpca_options1 == f.settings.effective_fpca_options2 == (;method=:auto)
    @test StatsAPI.fit(FSVD,x,y).selection.max_k == 16
    selected = StatsAPI.fit(FSVD,x,y;ncomponents=FVE(.5),max_k=2)
    @test selected.selection.chosen == 1
    @test ncomponents(selected) == length(svdvals(selected)) == 1
    @test length(fve(selected)) == 2
    @test occursin("denominator=16", sprint(show, f))
    @test f.cross_covariance.ids == x.ids
    @test f.cross_covariance.ids !== x.ids
    @test f.cross_covariance.grid1 !== x.grid
    # Analytic rank-one covariance on unequal, nonuniform domains.
    g1, g2 = [0.,.1,.6,1.], [-2.,-.5,2.]
    a, b, z = 1 .+ g1, 2 .- .2g2, [-2.,-1.,1.,2.]
    xx = GridFunctionalData(z*a',g1)
    yy = GridFunctionalData(z*b',g2)
    h = StatsAPI.fit(FSVD,xx,yy;ncomponents=1)
    na, nb = sqrt(dot(a,xx.weights.*a)), sqrt(dot(b,yy.weights.*b))
    @test h.singular_values ≈ [sum(abs2,z)/2*na*nb]
    @test h.left_functions[:,1] ≈ a/na
    @test h.right_functions[:,1] ≈ b/nb
    @test h.left_scores[:,1] ≈ z*na
    @test h.right_scores[:,1] ≈ z*nb
    @test h.fve == [1.]
    small = GridFunctionalData(Float32.(xx.values),Float32.(xx.grid))
    @test eltype((@inferred StatsAPI.fit(FSVD,small,small;ncomponents=1)).singular_values) == Float32
    @test eltype((@inferred StatsAPI.fit(FSVD,xx,small;ncomponents=1)).singular_values) == Float64
    @test_throws ArgumentError StatsAPI.fit(FSVD,xx,yy;
        fpca_options1=(;presmooth=LocalPolynomial(bandwidth=1.)))
    for kw in ((;max_k=0), (;ncomponents=0), (;max_k=1,ncomponents=2), (;max_k=true), (;ncomponents=true),
            (;bandwidth=.5), (;fpca_options1=[]), (;kernel=:gauss))
        @test_throws ArgumentError StatsAPI.fit(FSVD,xx,yy;kw...)
    end
    @test_throws ArgumentError StatsAPI.fit(FSVD,xx[1:2],yy[1:2])
    @test_throws DimensionMismatch StatsAPI.fit(FSVD,xx[1:3],yy)
    wrong = GridFunctionalData(yy.values, yy.grid; ids=[2,1,3,4])
    tagged = GridFunctionalData(xx.values, xx.grid; ids=1:4)
    @test_throws ArgumentError StatsAPI.fit(FSVD,tagged,wrong)
    zero = GridFunctionalData(ones(4,4),g1)
    @test_throws ArgumentError StatsAPI.fit(FSVD,zero,zero)
end

@testset "FSVD sparse weighted reference" begin
    input = readdlm(joinpath(@__DIR__, "data", "m4_cross_input.csv"), ','; comments=true)
    nodes = map(1:2) do j
        rows = input[input[:,1] .== j,:]
        IrregularFunctionalData([rows[rows[:,2] .== i,3] for i in 1:12],
            [rows[rows[:,2] .== i,4] for i in 1:12]; ids=1:12)
    end
    g = collect(range(0.,1.;length=51))
    phi = hcat(ones(51),g .- .5)
    cov1, cov2 = phi*Diagonal([.8,.3])*phi', phi*Diagonal([.7,.2])*phi'
    o1 = (;grid=g, mean=(;grid=g,values=1 .+ g), covariance=(;grid=g,surface=cov1),
        noise_variance=.2, ncomponents=2,max_k=2)
    o2 = (;grid=g, mean=(;grid=g,values=-.2 .+ .3g), covariance=(;grid=g,surface=cov2),
        noise_variance=.2,ncomponents=2,max_k=2)
    f = @inferred StatsAPI.fit(FSVD,nodes...;ncomponents=2,max_k=2,bandwidth=(.55,.8),
        kernel=KernelSmoothing.EpanechnikovKernel(),fpca_options1=o1,fpca_options2=o2)
    @test _m6_relative(f.cross_covariance.surface,_m6_read("sparse_cross")) < 1e-8
    @test _m6_relative(f.singular_values,_m6_read("sparse_values")[:,1]) < 1e-8
    @test _m6_relative(f.left_functions,_m6_read("sparse_left")[:,2:end]) < 1e-8
    @test _m6_relative(f.right_functions,_m6_read("sparse_right")[:,2:end]) < 1e-8
    @test _m6_relative(f.left_scores,_m6_read("sparse_scores1")) < 1e-8
    @test _m6_relative(f.right_scores,_m6_read("sparse_scores2")) < 1e-8
    @test _m6_relative(f.fve,_m6_read("sparse_values")[:,2]) < 1e-8
    @test_throws ArgumentError StatsAPI.fit(FSVD,nodes...)
    @test occursin("joint_conditional_expectation",sprint(show,f))
    # Joint conditioning must use the second node, even for the left scores.
    changed = IrregularFunctionalData([nodes[2][i][1] for i in 1:12],
        [nodes[2][i][2] .+ .1 for i in 1:12])
    ls, _ = FunctionalDataAnalysis._fsvd_joint_scores(nodes[1],changed,f.fpca1,f.fpca2,
        f.cross_covariance,f.left_functions,f.right_functions,f.singular_values)
    @test norm(ls-f.left_scores) > .01
end

@testset "FSVD mixed complete-grid and irregular marginals" begin
    rng = MersenneTwister(2026102501)
    g = _m6_read("dense_left")[:,1]
    values = _m6_read("dense_input1")
    x = GridFunctionalData(values + .4randn(rng, size(values)),g; ids=1:18)
    input = _m6_read("concurrent_input")
    times = [input[input[:,1] .== i,2] for i in 1:18]
    observations = [input[input[:,1] .== i,4] for i in 1:18]
    y = IrregularFunctionalData(times,
        [v + .4randn(rng, length(v)) for v in observations]; ids=1:18)
    # The complete grid has more points than the sample covariance's rank.
    @test length(g) > nobs(x)-1
    f = @inferred StatsAPI.fit(FSVD,x,y;bandwidth=.6,ncomponents=2,max_k=2)
    @test f.fpca1.method == f.fpca2.method == :pace
    @test f.fpca1.noise_variance > 0 && f.fpca2.noise_variance > 0
    @test length(f.fpca1.grid) == length(f.fpca2.grid) == 51
    @test all(isfinite,f.left_scores) && all(isfinite,f.right_scores)
    @test size(f.left_scores) == size(f.right_scores) == (18,2)
    @test f.settings.fpca_options1 == f.settings.fpca_options2 == (;)
    @test f.settings.effective_fpca_options1 == f.settings.effective_fpca_options2 == (;method=:pace)
    @test occursin("effective_fpca_options1 = (method = :pace,)",sprint(show,f))
    # Changing only the representation must preserve the conditional scores.
    irregular = StatsAPI.fit(FSVD,to_irregular(x),y;bandwidth=.6,ncomponents=2,max_k=2)
    @test f.singular_values ≈ irregular.singular_values
    @test f.left_functions ≈ irregular.left_functions
    @test f.right_functions ≈ irregular.right_functions
    @test f.left_scores ≈ irregular.left_scores
    @test f.right_scores ≈ irregular.right_scores
    reversed = @inferred StatsAPI.fit(FSVD,y,x;bandwidth=.6,ncomponents=2,max_k=2)
    @test reversed.fpca1.method == reversed.fpca2.method == :pace
    @test reversed.singular_values ≈ f.singular_values
    @test all(isfinite,reversed.left_scores) && all(isfinite,reversed.right_scores)
    masked = StatsAPI.fit(FSVD,x,to_grid(y,g,NoInterpolation());
        bandwidth=.6,ncomponents=2,max_k=2)
    @test masked.fpca1.method == masked.fpca2.method == :pace
    @test masked.left_scores ≈ f.left_scores
    @test masked.right_scores ≈ f.right_scores
    # Supplying noise alone must work without also supplying method=:pace.
    options = (;method=:auto,noise_variance=.25)
    explicit = StatsAPI.fit(FSVD,x,y;bandwidth=.6,ncomponents=2,max_k=2,
        fpca_options1=options)
    @test explicit.fpca1.noise_variance == .25
    @test explicit.settings.fpca_options1 == options
    @test explicit.settings.effective_fpca_options1 == (;method=:pace,noise_variance=.25)
    for (a,b,name) in ((x,y,"fpca_options1"), (y,x,"fpca_options2"))
        kw = NamedTuple{(Symbol(name),)}(((;method=:dense),))
        @test_throws ArgumentError(
            "FSVD with sparse or incomplete nodes requires PACE marginals; use $name=(;method=:pace, ...)") StatsAPI.fit(
                FSVD,a,b;bandwidth=.6,kw...)
    end
end

@testset "Concurrent correlation reference and errors" begin
    input = _m6_read("concurrent_input")
    times = [input[input[:,1] .== i,2] for i in 1:18]
    x = IrregularFunctionalData(times,[input[input[:,1] .== i,3] for i in 1:18])
    y = IrregularFunctionalData(times,[input[input[:,1] .== i,4] for i in 1:18])
    expected = _m6_read("concurrent")
    c = concurrent_correlation(x,y; bandwidth=[.5,.6,.55,.65,.7],grid=expected[:,1])
    @test _m6_relative(c.correlation,expected[:,2]) < 1e-8
    @test c.bandwidth == [.5,.6,.55,.65,.7]
    @test c.degree == 1
    @test !any(c.affected1) && !any(c.affected2)
    @test concurrent_correlation(x,y;bandwidth=.6).grid == sort(unique(vcat(times...)))
    @test concurrent_correlation(x,y;bandwidth=[.6]).correlation ≈
        concurrent_correlation(x,y;bandwidth=.6).correlation
    for bandwidth in ([.5,.6], [-1.], [NaN], :bad)
        @test_throws ArgumentError concurrent_correlation(x,y;bandwidth)
    end
    @test_throws ArgumentError concurrent_correlation(x,y;bandwidth=.5,grid=[-1.,1.])
    @test_throws DimensionMismatch concurrent_correlation(x[1:2],y;bandwidth=.5)
    mismatched = IrregularFunctionalData([t .+ .01 for t in times],
        [y[i][2] for i in 1:18])
    @test_throws ArgumentError concurrent_correlation(x,mismatched;bandwidth=.5)
    g = [0.,.5,1.]
    zero = GridFunctionalData(zeros(4,3),g)
    @test_logs (:warn,r"variance estimate is nonpositive") begin
        c = concurrent_correlation(zero,zero;bandwidth=2.)
        @test all(isnan,c.correlation)
        @test all(c.affected1) && all(c.affected2)
    end
end

@testset "Dynamic correlation reference and errors" begin
    g = _m6_read("dense_left")[:,1]
    x = GridFunctionalData(_m6_read("dense_input1"),g)
    y = GridFunctionalData(_m6_read("dense_input2"),g)
    d = dynamic_correlation(x,y)
    @test _m6_relative(d.individual,vec(_m6_read("dynamic"))) < 1e-8
    @test d.correlation ≈ sum(_m6_read("dynamic"))/18
    @test dynamic_correlation(x,x).individual ≈ ones(18)
    @test dynamic_correlation(x,GridFunctionalData(-x.values,g)).individual ≈ -ones(18)
    @test d.grid !== x.grid && d.weights !== x.weights
    @test d.standardized1*(x.weights/sum(x.weights)) ≈ zeros(18) atol=1e-14
    @test (d.standardized1.^2)*(x.weights/sum(x.weights)) ≈ ones(18)
    # Domain rescaling and nonuniform-grid integrals preserve affine correlation.
    t = [0.,.1,.7,2.]
    xx = GridFunctionalData([-2.,-1.,1.,2.]*t',t)
    @test dynamic_correlation(xx,xx).correlation ≈ 1
    @test_throws ArgumentError dynamic_correlation(x[1:1],y[1:1])
    @test_throws DimensionMismatch dynamic_correlation(x[1:2],y)
    @test_throws ArgumentError dynamic_correlation(x,GridFunctionalData(y.values,g .+ 1))
    @test_throws ArgumentError dynamic_correlation(GridFunctionalData(ones(4,4),t),
        GridFunctionalData(ones(4,4),t))
    masked = Matrix{Union{Missing,Float64}}(x.values); masked[1,1] = missing
    @test_throws ArgumentError dynamic_correlation(GridFunctionalData(masked,g),y)
end
