import StatsAPI
import KernelSmoothing

# A minimal offset-indexed input verifies rejection without adding a dependency.
struct _OffsetAuditVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(v::_OffsetAuditVector) = size(v.data)
Base.axes(v::_OffsetAuditVector) = (0:length(v.data)-1,)
Base.getindex(v::_OffsetAuditVector, i::Int) = v.data[i+1]

@testset "Count and order validation" begin
    FDA = FunctionalDataAnalysis
    grid = collect(range(0., 1.; length=17))
    rng = MersenneTwister(3901)
    fd = GridFunctionalData([i/3 + sinpi(t) + .2randn(rng) for i in 1:12, t in grid], grid)
    mfd = MultivariateFunctionalData([fd, fd])
    irregular = to_irregular(fd)
    basis = MonomialBasis((0.,1.), 1)
    bfd = BasisFunctionalData(randn(rng,12,2), basis)
    dense = StatsAPI.fit(FPCA, fd; ncomponents=1)
    mean = StatsAPI.fit(SmoothedMean, fd; grid, bandwidth=1.2)
    covariance = (;grid, surface=ones(length(grid),length(grid)))
    pace(;kwargs...) = StatsAPI.fit(FPCA, fd; method=:pace, grid,
        mean=(;grid, values=ones(length(grid))), covariance, noise_variance=.2, kwargs...)
    cases = [
        "B-spline order" => x -> BSplineBasis(x, [0.,.5,1.]),
        "B-spline dimension" => x -> BSplineBasis(1, (0.,1.), x),
        "Fourier dimension" => x -> FourierBasis((0.,1.), x),
        "monomial degree" => x -> MonomialBasis((0.,1.), x),
        "basis derivative" => x -> basis_matrix(basis, grid; derivative=x),
        "basis penalty" => x -> penalty(basis, x),
        "derivative basis" => x -> DerivativeBasis(basis, x),
        "local degree" => x -> LocalPolynomial(bandwidth=1.,degree=x),
        "local derivative" => x -> LocalPolynomial(bandwidth=1.,derivative=x),
        "exact basis derivative" => x -> derivative(bfd,x),
        "basis conversion order" => x -> to_basis(fd,basis;order=x),
        "spline penalty order" => x -> PenalizedSpline(basis;lambda=.1,order=x),
        "pooled mean derivative" => x -> StatsAPI.fit(SmoothedMean, fd;grid,bandwidth=1.2,derivative=x),
        "curve folds" => x -> CurveCV([1.,2.];folds=x),
        "dense selection" => x -> StatsAPI.fit(FPCA,fd;ncomponents=x),
        "dense cap" => x -> StatsAPI.fit(FPCA,fd;max_k=x),
        "legacy dense cap" => x -> StatsAPI.fit(FPCA,fd;maxK=x),
        "PACE selection" => x -> pace(ncomponents=x,max_k=1),
        "PACE cap" => x -> pace(max_k=x,ncomponents=1),
        "pooled cap" => x -> StatsAPI.fit(FPCA,mfd;basis=:pooled,max_k=x),
        "basis cap" => x -> StatsAPI.fit(FPCA,bfd;max_k=x),
        "basis selection" => x -> StatsAPI.fit(FPCA,bfd;ncomponents=x),
        "basis CV count" => x -> StatsAPI.fit(FPCA,bfd;ncomponents=x,lambda=LambdaCV([.1,1.])),
        "MFPCA selection" => x -> StatsAPI.fit(MFPCA,mfd;ncomponents=x),
        "FSVD selection" => x -> StatsAPI.fit(FSVD,fd,fd;ncomponents=x),
        "FSVD cap" => x -> StatsAPI.fit(FSVD,fd,fd;max_k=x),
        "CCA count" => x -> StatsAPI.fit(FunctionalCCA,bfd;ncan=x),
        "CCA count alias" => x -> StatsAPI.fit(FunctionalCCA,bfd;ncomponents=x),
        "FACE selection" => x -> StatsAPI.fit(FPCA,fd;method=:face,ncomponents=x,face=(;knots=4)),
        "FACE cap" => x -> StatsAPI.fit(FPCA,fd;method=:face,max_k=x,face=(;knots=4)),
        "sparse FACE selection" => x -> StatsAPI.fit(FPCA,irregular;method=:face,ncomponents=x),
        "sparse FACE cap" => x -> StatsAPI.fit(FPCA,irregular;method=:face,max_k=x),
        "FACE intervals" => x -> StatsAPI.fit(FPCA,fd;method=:face,face=(;knots=x)),
        "difference dimension" => x -> FDA._difference_penalty(x,0),
        "difference order" => x -> FDA._difference_penalty(5,x),
        "FACE degree" => x -> FDA._pspline_design(grid,4,x,2),
        "FACE difference order" => x -> FDA._pspline_design(grid,4,3,x),
        "FPCA derivative order" => x -> derivative(dense,x),
        "covariance bins" => x -> StatsAPI.fit(SmoothedCovariance,fd,mean;grid,bandwidth=1.2,binning=x),
        "cross-covariance bins" => x -> StatsAPI.fit(SmoothedCrossCovariance,fd,fd,mean,mean;grid1=grid,bandwidth=1.2,binning=x),
        "PACE bins" => x -> FDA._pace_binning(fd,x),
        "bootstrap count" => x -> mean_bands(fd,mean;B=x,rng=MersenneTwister(1)),
        "bootstrap count alias" => x -> mean_bands(fd,mean;bootstrap_replicates=x,rng=MersenneTwister(1)),
        "Fourier simulation count" => x -> fourier_eigenfunctions((0.,1.),x),
        "Legendre simulation count" => x -> legendre_eigenfunctions((0.,1.),x),
        "simulation subjects" => x -> simulate_kl(MersenneTwister(1),x;mean=t->0.,
            eigenfunctions=[t->1.],eigenvalues=[1.],sampling=RegularSampling(grid))
    ]
    for (label, call) in cases
        @testset "$label" begin
            @test_throws ArgumentError call(true)
            @test_throws ArgumentError call(false)
            @test_throws ArgumentError call(big(typemax(Int))+1)
        end
    end
    @test_throws ArgumentError SparseSampling((0.,1.), true:true)
    @test_throws ArgumentError SparseSampling((0.,1.), big(typemax(Int))+1:big(typemax(Int))+2)
    @test length(fourier_eigenfunctions((0.,1.),0)) == 0
    @test nbasis(MonomialBasis((0.,1.),0)) == 1
    @test LocalPolynomial(bandwidth=1.,degree=0,derivative=0).degree == 0
    @test penalty(basis,0) == gram(basis)
end

@testset "Float32 and one-based array genericity" begin
    # These are Float32 smoke tolerances, not replacements for Float64 goldens.
    relative32, absolute32 = 2e-3, 2e-5
    grid = collect(0.:.125:1.)
    times = [collect(range(0.,1.;length=m)) for m in (7,8,9,10)]
    values = [1 .+ t .+ (-1)^i .* cos.(2t) for (i,t) in enumerate(times)]
    fd = IrregularFunctionalData(times,values)
    fd32 = IrregularFunctionalData(Float32.(fd.times),Float32.(fd.values),fd.offsets;
        domain=(0f0,1f0))
    reference = StatsAPI.fit(FPCA,fd; grid,ncomponents=1,max_k=2,
        mean_bandwidth=1.2,covariance_bandwidth=1.2,weighting=:subj,binning=5)
    result = StatsAPI.fit(FPCA,fd32; grid=range(0f0,1f0;length=9),ncomponents=1,max_k=2,
        mean_bandwidth=1.2f0,covariance_bandwidth=1.2f0,weighting=:subj,binning=5)
    @test result.settings.mean_source == result.settings.covariance_source == :estimated
    @test result.settings.noise_variance_source == :estimated
    @test scores(result) isa Matrix{Float32}
    @test eigvals(result) isa Vector{Float32}
    @test eigvals(result) ≈ eigvals(reference) rtol=relative32 atol=absolute32
    @test noise_variance(result) ≈ noise_variance(reference) rtol=relative32 atol=absolute32
    @test result.mean.values ≈ reference.mean.values rtol=relative32 atol=absolute32
    @test scores(result) ≈ scores(reference) rtol=relative32 atol=absolute32

    raw = Float32[i/3 + 2t for i in 1:5, t in grid]
    dense32 = GridFunctionalData(@view(raw[2:4,:]),range(0f0,1f0;length=9))
    dense64 = GridFunctionalData(Float64.(dense32.values),Float64.(dense32.grid))
    work = range(0f0,1f0;length=13)
    smooth32 = smooth_curves(dense32,LocalPolynomial(bandwidth=.7f0);grid=work)
    smooth64 = smooth_curves(dense64,LocalPolynomial(bandwidth=Float64(.7f0));grid=Float64.(work))
    @test smooth32.values isa Matrix{Float32}
    @test smooth32.values ≈ smooth64.values rtol=relative32 atol=absolute32
    @test smooth32.values ≈ Float32[i/3 + 2t for i in 2:4, t in work] rtol=relative32 atol=absolute32
    queries = @view dense32.grid[2:8]
    @test evaluate(dense32,queries,PiecewiseLinear()) == dense32.values[:,2:8]
    @test quadrature_weights(queries) isa Vector{Float32}
    b = MonomialBasis((0f0,1f0),1)
    coefficients = @view raw[1:3,1:2]
    basis_data = BasisFunctionalData(coefficients,b)
    @test basis_data.coefs == coefficients && basis_data.coefs !== coefficients
    @test evaluate(basis_data,queries) ≈ coefficients*transpose(basis_matrix(b,queries))
    offset = _OffsetAuditVector(Float32[0,.5,1])
    @test_throws ArgumentError quadrature_weights(offset)
    @test_throws ArgumentError GridFunctionalData(ones(2,3),offset)
    @test_throws ArgumentError evaluate(dense32,offset,PiecewiseLinear())
    @test_throws ArgumentError basis_matrix(b,offset)
end
