using FunctionalDataAnalysis, LinearAlgebra, Random, Test, TOML
import StatsAPI, JET

JET.JET_AVAILABLE || error("JET is unavailable on Julia $VERSION; do not skip this job")
Base.pkgversion(JET) == v"0.13.0" || error("review the baseline before changing JET")
BLAS.set_num_threads(1)

function jet_cases()
    rng = MersenneTwister(2026102551)
    grid = collect(range(0.,1.;length=41))
    a, b = randn(rng,32), randn(rng,32)
    fd = GridFunctionalData([1+t+a[i]*sinpi(2t)+.6b[i]*cospi(2t)+.1randn(rng)
        for i in 1:32,t in grid],grid)
    fd2 = GridFunctionalData([.5+t+.8a[i]*sinpi(2t)+.7b[i]*cospi(2t)+.1randn(rng)
        for i in 1:32,t in grid],grid)
    ts = [sort!(rand(rng,8)) for _ in 1:32]
    ys = [[1+t+a[i]*sinpi(2t)+.6b[i]*cospi(2t)+.1randn(rng) for t in ts[i]] for i in 1:32]
    sparse = IrregularFunctionalData(ts,ys;domain=(0.,1.))
    multi = MultivariateFunctionalData([fd,fd2])
    dense = StatsAPI.fit(FPCA,fd;ncomponents=2,max_k=4)
    basis = FourierBasis((0.,1.),7)
    coefs = randn(rng,32,7)
    basis1 = BasisFunctionalData(coefs,basis)
    basis2 = BasisFunctionalData(.8coefs+.6randn(rng,32,7),basis)
    (
        "mean" => ()->StatsAPI.fit(SmoothedMean,sparse;grid,bandwidth=.25),
        "curve_smoothing" => ()->smooth_curves(fd,LocalPolynomial(;bandwidth=.2);grid),
        "dense" => ()->StatsAPI.fit(FPCA,fd;method=:dense,ncomponents=2,max_k=4),
        "pace" => ()->StatsAPI.fit(FPCA,sparse;method=:pace,grid,ncomponents=2,max_k=4,
            mean_bandwidth=.25,covariance_bandwidth=.4,noise_variance=.04,binning=nothing),
        "face" => ()->StatsAPI.fit(FPCA,fd;method=:face,ncomponents=2,face=(;knots=8,lambda=1.)),
        "mfpca" => ()->StatsAPI.fit(MFPCA,multi;ncomponents=2,univariate=(;ncomponents=2,max_k=4)),
        "fsvd" => ()->StatsAPI.fit(FSVD,fd,fd2;ncomponents=2,max_k=4,
            fpca_options1=(;ncomponents=2,max_k=4),fpca_options2=(;ncomponents=2,max_k=4)),
        "derivative" => ()->derivative(dense,1;bandwidth=.2),
        "sparse_face" => ()->StatsAPI.fit(FPCA,sparse;method=:face,ncomponents=2),
        "basis_fpca" => ()->StatsAPI.fit(FPCA,basis1;lambda=.01,ncomponents=2),
        "basis_cca" => ()->StatsAPI.fit(FunctionalCCA,basis1,basis2;ncan=2),
    )
end

# JET is pinned because these diagnostic fields and compiler inference can change.
# A multiset of report type and innermost source location detects replacements,
# additions and stale suppressions; a count-only allowance would miss replacements.
function report_site(report)
    frame = last(report.vst)
    join((nameof(typeof(report)),basename(string(frame.file)),frame.line), ":")
end

function check_jet_baseline(cases, baseline)
    @test Set(first.(cases)) == Set(keys(baseline))
    for (name, f) in cases
        f() # Ensure each concrete workload also succeeds at runtime.
        @testset "$name" begin
            for (kind, analyze) in (("opt",JET.report_opt),("call",JET.report_call))
                result = analyze(f,Tuple{};target_modules=(FunctionalDataAnalysis,))
                sites = sort!(report_site.(JET.get_reports(result)))
                println("JET ",name," ",kind," reports=",length(sites)," sites=",sites)
                show(stdout,MIME("text/plain"),result);println()
                @test sites == baseline[name][kind]
            end
        end
    end
end

println("Julia=",VERSION," JET=",Base.pkgversion(JET)," Julia_threads=",Threads.nthreads(),
    " BLAS_threads=",BLAS.get_num_threads())
cases = jet_cases()
baseline_path = joinpath(@__DIR__,"baseline.toml")
if get(ENV,"FDA_JET_RECORD_BASELINE","false") == "true"
    # Explicit development command only. CI never records or accepts its own baseline.
    baseline = Dict(name=>Dict(kind=>begin
        result = analyze(f,Tuple{};target_modules=(FunctionalDataAnalysis,))
        show(stdout,MIME("text/plain"),result);println()
        sort!(report_site.(JET.get_reports(result)))
    end for (kind,analyze) in (("opt",JET.report_opt),("call",JET.report_call))) for (name,f) in cases)
    open(io->TOML.print(io,baseline;sorted=true),baseline_path,"w")
    println("Recorded candidate baseline; review every report before committing it")
else
    @testset "Bounded JET baseline" begin
        check_jet_baseline(cases,TOML.parsefile(baseline_path))
    end
end
