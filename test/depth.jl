using DelimitedFiles: readdlm
import StatsAPI

const _DEPTH_INPUT = readdlm(joinpath(@__DIR__, "data", "m8_depth_input.csv"), ',', Any; comments=true)
const _DEPTH_GOLDEN = readdlm(joinpath(@__DIR__, "data", "m8_depth_golden.csv"), ',', Any; comments=true)

function _depth_input(name)
    rows = _DEPTH_INPUT[_DEPTH_INPUT[:, 1] .== name, :]
    out = zeros(Int(maximum(rows[:, 2])), Int(maximum(rows[:, 3])))
    for row in eachrow(rows)
        out[Int(row[2]), Int(row[3])] = Float64(row[4])
    end
    out
end
_depth_golden(case, quantity) =
    Float64.(_DEPTH_GOLDEN[(_DEPTH_GOLDEN[:, 1] .== case) .& (_DEPTH_GOLDEN[:, 2] .== quantity), 4])
_depth_close(actual, case, quantity) =
    golden_relative_error(actual, _depth_golden(case, quantity)) < 1e-8

# Definitions by pairs, for checking the rank and bit-mask computations.
function _mbd_pairs(x, y, w)
    n0 = size(y, 1)
    [sum(w[t] * count(j < k && min(y[j, t], y[k, t]) <= x[i, t] <= max(y[j, t], y[k, t])
        for j in 1:n0, k in 1:n0) for t in axes(x, 2)) / (n0 * (n0 - 1) / 2) for i in axes(x, 1)]
end
function _bd_pairs(x, y)
    n0 = size(y, 1)
    [count(j < k && all(min.(y[j, :], y[k, :]) .<= x[i, :] .<= max.(y[j, :], y[k, :]))
        for j in 1:n0, k in 1:n0) / (n0 * (n0 - 1) / 2) for i in axes(x, 1)]
end

@testset "depth goldens against roahd 1.4.3 and fda.usc 2.2.0" begin
    grid = vec(_depth_input("grid"))
    proj = _depth_input("proj")
    for case in ("smooth", "tied")
        fd = GridFunctionalData(_depth_input(case), grid)
        new = GridFunctionalData(_depth_input(case * "_new"), grid)
        uniform_average = ModifiedBandDepth(ties=:average, integration=:uniform)
        @test _depth_close(depth(fd, BandDepth(definition=:roahd)).values, case, "BD")
        @test _depth_close(depth(new, fd, BandDepth(definition=:roahd)).values, case, "BD_relative")
        @test _depth_close(depth(fd, BandDepth()).values, case, "BD_graph")
        @test _depth_close(depth(new, fd, BandDepth()).values, case, "BD_graph_relative")
        @test _depth_close(depth(fd, uniform_average).values, case, "MBD")
        @test _depth_close(depth(fd, ModifiedBandDepth(integration=:uniform)).values, case, "MBD_ties")
        @test _depth_close(depth(new, fd, uniform_average).values, case, "MBD_relative")
        @test _depth_close(modified_epigraph_index(fd; integration=:uniform), case, "MEI")
        @test _depth_close(modified_hypograph_index(fd; integration=:uniform), case, "MHI")
        fm = depth(fd, FraimanMuniz(integration=:uniform))
        @test _depth_close(fm.values, case, "depth.FM")
        @test fm.median == Int(only(_depth_golden(case, "depth.FM_lmed")))
        @test _depth_close(depth(new, fd, FraimanMuniz(integration=:uniform)).values, case, "depth.FM_relative")
        @test _depth_close(depth(fd, ModifiedBandDepth()).values, case, "depth.MB")
        @test _depth_close(depth(new, fd, ModifiedBandDepth()).values, case, "depth.MB_relative")
        rp = RandomProjectionDepth(projections=proj)
        @test _depth_close(depth(fd, rp).values, case, "depth.RP")
        @test _depth_close(depth(new, fd, rp).values, case, "depth.RP_relative")
        mode = depth(fd, ModalDepth())
        @test _depth_close(mode.values, case, "depth.mode")
        @test _depth_close([mode.settings.bandwidth], case, "depth.mode_hq")
        @test _depth_close(depth(new, fd, ModalDepth()).values, case, "depth.mode_relative")
    end

    nu = GridFunctionalData(_depth_input("smooth"), vec(_depth_input("grid_nu")))
    @test _depth_close(depth(nu, ModifiedBandDepth()).values, "nonuniform", "depth.MB")
    @test _depth_close(depth(nu, FraimanMuniz(integration=:uniform)).values, "nonuniform", "depth.FM")
    @test _depth_close(depth(nu, ModifiedBandDepth(ties=:average, integration=:uniform)).values,
        "nonuniform", "MBD")
    nu_mode = depth(nu, ModalDepth())
    @test _depth_close(nu_mode.values, "nonuniform", "depth.mode")
    @test _depth_close([nu_mode.settings.bandwidth], "nonuniform", "depth.mode_hq")
    @test _depth_close(depth(nu, RandomProjectionDepth(projections=proj)).values, "nonuniform", "depth.RP")

    mv(names...) = MultivariateFunctionalData([GridFunctionalData(_depth_input(n), grid) for n in names])
    mfd, mnew = mv("mv1", "mv2"), mv("mv1_new", "mv2_new")
    tied = mv("mvt1", "mvt2")
    average = ModifiedBandDepth(ties=:average, integration=:uniform)
    @test _depth_close(depth(mfd, average).values, "multivariate", "multiMBD")
    weighted = depth(mfd, average; weights=[0.3, 0.7])
    @test _depth_close(weighted.values, "multivariate", "multiMBD_weighted")
    @test weighted.settings.component_weights == [0.3, 0.7]
    @test _depth_close(depth(mfd, BandDepth(definition=:roahd)).values, "multivariate", "multiBD")
    @test _depth_close(depth(tied, average).values, "multivariate", "multiMBD_tied")
    @test _depth_close(depth(tied, ModifiedBandDepth(integration=:uniform)).values,
        "multivariate", "multiMBD_tied_ties")
    fmp = FraimanMuniz(integration=:uniform)
    @test _depth_close(depth(mfd, fmp).values, "multivariate", "depth.FMp")
    @test _depth_close(depth(mnew, mfd, fmp).values, "multivariate", "depth.FMp_relative")
    rpp = depth(mfd, RandomProjectionDepth(projections=proj))
    @test rpp.settings.scale
    @test _depth_close(rpp.values, "multivariate", "depth.RPp")
    @test _depth_close(depth(mnew, mfd, RandomProjectionDepth(projections=proj)).values,
        "multivariate", "depth.RPp_relative")
    modep = depth(mfd, ModalDepth())
    @test modep.settings.quantile == 0.1
    @test _depth_close(modep.values, "multivariate", "depth.modep")
    @test _depth_close([modep.settings.bandwidth], "multivariate", "depth.modep_hq")
    @test _depth_close(depth(mnew, mfd, ModalDepth()).values, "multivariate", "depth.modep_relative")
end

@testset "rank formulas agree with pairwise definitions" begin
    rng = Random.Xoshiro(81)
    for (n, m) in ((6, 5), (9, 7), (5, 70))
        grid = sort(rand(rng, m)) .+ (0:m-1)
        values = round.(randn(rng, n, m); digits=1)
        values[end, :] = values[1, :]
        fd = GridFunctionalData(values, grid)
        new = round.(randn(rng, 3, m); digits=1)
        w = quadrature_weights(grid) ./ sum(quadrature_weights(grid))
        @test depth(fd, ModifiedBandDepth()).values ≈ _mbd_pairs(values, values, w) atol=1e-13
        @test depth(GridFunctionalData(new, grid), fd, ModifiedBandDepth()).values ≈
            _mbd_pairs(new, values, w) atol=1e-13
        @test depth(fd, BandDepth()).values ≈ _bd_pairs(values, values) atol=1e-13
        @test depth(GridFunctionalData(new, grid), fd, BandDepth()).values ≈ _bd_pairs(new, values) atol=1e-13
        # Exact methods: a sample's depth equals its depth against an identical reference.
        for method in (ModifiedBandDepth(), BandDepth(), FraimanMuniz(), ModalDepth(quantile=0.5),
                RandomProjectionDepth(projections=randn(rng, 4, m)))
            @test depth(fd, method).values ≈ depth(fd, fd, method).values atol=1e-13
        end
        @test modified_epigraph_index(fd) ≈ modified_epigraph_index(fd, fd) atol=1e-13
    end
end

@testset "depth analytic values" begin
    grid = [0.0, 0.5, 1.0]
    levels = GridFunctionalData(repeat(1.0:5.0, 1, 3), grid)
    @test depth(levels, FraimanMuniz()).values ≈ [1 - 2abs(0.5 - k / 5) for k in 1:5]
    @test depth(levels, FraimanMuniz(scale=false)).values ≈ [1 - abs(0.5 - k / 5) for k in 1:5]
    @test depth(levels, ModifiedBandDepth()).values ≈
        [(10 - (5 - r) * (4 - r) / 2 - (r - 1) * (r - 2) / 2) / 10 for r in 1:5]
    @test depth(levels, BandDepth()).values ≈ depth(levels, ModifiedBandDepth()).values
    @test modified_epigraph_index(levels) ≈ [1.0, 0.8, 0.6, 0.4, 0.2]
    @test modified_hypograph_index(levels) ≈ [0.2, 0.4, 0.6, 0.8, 1.0]
    d = depth(levels, ModifiedBandDepth())
    @test d.median == 3 && d.ordering[1] == 3
    @test d.median_curve.values == [3.0 3.0 3.0]
    @test d.reference == :sample && d.nreference == 5 && d.representation == :grid
    # Ties keep curve order; the median is the first deepest curve.
    @test depth(GridFunctionalData([0.0 0.0; 1.0 1.0; 1.0 1.0; 2.0 2.0], [0, 1]), ModifiedBandDepth()).ordering ==
        [2, 3, 1, 4]
    # Signed zeros are equal values.
    zeros_fd = GridFunctionalData([0.0 0.0; -0.0 -0.0; 1.0 1.0], [0, 1])
    @test modified_epigraph_index(zeros_fd) ≈ [1.0, 1.0, 1 / 3]
    @test depth(zeros_fd, FraimanMuniz()).values ≈ [1 - 2abs(0.5 - 2 / 3), 1 - 2abs(0.5 - 2 / 3), 0.0]
    # A curve inside a band of two crossing curves: the definition counts the
    # pair, roahd's above/below count does not.
    crossing = GridFunctionalData([0.0 0.0; 1.0 -1.0; -1.0 1.0], [0, 1])
    @test depth(crossing, BandDepth()).values[1] == 1
    @test depth(crossing, BandDepth(definition=:roahd)).values[1] ≈ 2 / 3
    @test depth(GridFunctionalData([0.0 0.0], [0, 1]), crossing[2:3], BandDepth()).values == [1.0]
    @test depth(GridFunctionalData([0.0 0.0], [0, 1]), crossing[2:3], BandDepth(definition=:roahd)).values == [0.0]
    # Modal depth: Gaussian kernel sums at an explicit bandwidth.
    two = GridFunctionalData([0.0 0.0; 1.0 1.0], [0, 1])
    @test depth(two, ModalDepth(bandwidth=2)).values ≈ fill((1 + exp(-1 / 8)) / sqrt(2pi), 2)
    # Quadrature integration weights interior grid points more than endpoints.
    step = GridFunctionalData([0.0 0.0 0.0; 1.0 1.0 -1.0; -1.0 -1.0 1.0], [0, 1, 2])
    @test modified_epigraph_index(step)[2] ≈ (1 * 0.25 + 1 * 0.5 + 3 * 0.25) / 3
    @test modified_epigraph_index(step; integration=:uniform)[2] ≈ (1 + 1 + 3) / 9
end

@testset "depth representations" begin
    grid = collect(range(0, 1; length=21))
    rng = Random.Xoshiro(82)
    values = [a + b * sin(2pi * t) for (a, b) in zip(randn(rng, 15), randn(rng, 15)), t in grid]
    fd = GridFunctionalData(values, grid)
    basis = FourierBasis((0.0, 1.0), 3)
    coefs = randn(rng, 10, 3)
    bfd = BasisFunctionalData(coefs, basis)
    bd = depth(bfd, ModifiedBandDepth(); grid)
    @test bd.representation == :basis_evaluated
    @test bd.values ≈ depth(GridFunctionalData(evaluate(bfd, grid), grid), ModifiedBandDepth()).values
    @test depth(bfd, fd, ModifiedBandDepth()).values ≈
        depth(GridFunctionalData(evaluate(bfd, grid), grid), fd, ModifiedBandDepth()).values
    @test_throws ArgumentError depth(bfd, ModifiedBandDepth())

    observed = [sort(randperm(rng, 21)[1:8]) for _ in 1:15]
    irregular = IrregularFunctionalData([grid[k] for k in observed], [values[i, observed[i]] for i in 1:15])
    d = depth(irregular, ModifiedBandDepth(); fpca=(; ncomponents=2))
    f = StatsAPI.fit(FPCA, irregular; method=:pace, ncomponents=2)
    @test d.representation == :fpca_reconstruction
    @test d.fpca isa FPCA && d.fpca.method == :pace && ncomponents(d.fpca) == 2
    @test d.values ≈ depth(reconstruct(f), ModifiedBandDepth()).values
    @test depth(f, ModifiedBandDepth()).values ≈ d.values
    newdata = irregular[1:4]
    relative = depth(newdata, fd, ModifiedBandDepth(); fpca=(; ncomponents=2))
    g = StatsAPI.fit(FPCA, fd; method=:pace, ncomponents=2)
    @test relative.representation == :fpca_reconstruction && relative.nreference == 15
    @test relative.values ≈ depth(reconstruct(g, project_scores(g, newdata)), reconstruct(g),
        ModifiedBandDepth()).values
    @test depth(newdata, g, ModifiedBandDepth()).values ≈ relative.values
    # Graph indices accept a supplied FPCA fit as the reference but never fit one.
    ghat = reconstruct(g, project_scores(g, fd))
    @test modified_epigraph_index(fd, g) ≈ modified_epigraph_index(ghat, reconstruct(g))
    @test modified_hypograph_index(newdata, g) ≈
        modified_hypograph_index(reconstruct(g, project_scores(g, newdata)), reconstruct(g))
    @test modified_epigraph_index(f) ≈ modified_epigraph_index(reconstruct(f))
    @test_throws ArgumentError modified_epigraph_index(fd, irregular)
    masked = Matrix{Union{Missing,Float64}}(values)
    masked[2, 3] = missing
    @test depth(GridFunctionalData(masked, grid), ModifiedBandDepth()).representation == :fpca_reconstruction

    mfd = MultivariateFunctionalData([fd, irregular])
    md = depth(mfd, ModifiedBandDepth(); fpca=(; ncomponents=2))
    @test md.representation == [:grid, :fpca_reconstruction]
    @test md.fpca[1] === nothing && md.fpca[2] isa FPCA
    @test md.median_curve isa MultivariateFunctionalData
    @test md.values ≈ (depth(fd, ModifiedBandDepth()).values .+ d.values) ./ 2
end

@testset "depth settings and display" begin
    grid = collect(range(0, 1; length=11))
    values = [sin(i * t) + i / 3 for i in 1:8, t in grid]
    fd = GridFunctionalData(values, grid)
    d = depth(fd, FraimanMuniz())
    @test d.settings.scale === true && d.settings.integration == :quadrature && d.settings.grid == grid
    @test nobs(d) == 8
    shown = sprint(show, d)
    @test occursin("FraimanMuniz", shown) && occursin("median=", shown) && occursin("integration", shown)
    @test occursin("ModalDepth", sprint(show, depth(fd, ModalDepth())))
    # Generated projections are reproducible, unit-norm and recorded.
    rp1 = depth(fd, RandomProjectionDepth(nproj=7, rng=Random.Xoshiro(3)))
    rp2 = depth(fd, RandomProjectionDepth(nproj=7, rng=Random.Xoshiro(3)))
    @test rp1.values == rp2.values && rp1.settings.generated && !rp1.settings.scale
    z = rp1.settings.projections
    @test size(z) == (7, 11)
    @test [sum(quadrature_weights(grid) .* abs2.(z[j, :])) for j in 1:7] ≈ ones(7)
    @test all(v -> 0 <= v <= 0.5, rp1.values)
    # Their square root reproduces the fda.usc rproc2fdata covariance exp(-|s - t|/θ), θ = 0.2.
    uneven = [0.0, 0.05, 0.2, 0.3, 0.55, 0.9, 1.0]
    root = FunctionalDataAnalysis._projection_root(uneven)
    @test root * root' ≈ [exp(-abs(s - t) / 0.2) for s in uneven, t in uneven] atol=1e-12
    root2 = FunctionalDataAnalysis._projection_root(2 .* uneven)
    @test root2 * root2' ≈ [exp(-abs(s - t) / 0.4) for s in 2 .* uneven, t in 2 .* uneven] atol=1e-12
    @test occursin("generated", sprint(show, rp1.method))
    ids_fd = GridFunctionalData(values, grid; ids=string.('a':'h'))
    @test depth(ids_fd, ModifiedBandDepth()).ids == string.('a':'h')
end

@testset "depth error paths" begin
    grid = [0.0, 0.5, 1.0]
    fd = GridFunctionalData([0.0 1.0 2.0; 1.0 2.0 0.0; 2.0 0.0 1.0], grid)
    other = GridFunctionalData([0.0 1.0; 1.0 0.0], [0.0, 1.0])
    single = GridFunctionalData([0.0 1.0 2.0], grid)
    @test_throws ArgumentError depth(single, ModifiedBandDepth())
    @test_throws ArgumentError depth(fd, single, ModalDepth())
    @test_throws ArgumentError depth(other, fd, ModifiedBandDepth())
    @test_throws ArgumentError depth(fd, ModifiedBandDepth(); grid=[0.0, 1.0])
    @test_throws ArgumentError depth(fd, ModifiedBandDepth(); fpca=(; ncomponents=1))
    @test_throws ArgumentError depth(fd, ModifiedBandDepth(); weights=[1.0])
    @test_throws ArgumentError depth(1.0, ModifiedBandDepth())
    @test_throws ArgumentError ModifiedBandDepth(ties=:min)
    @test_throws ArgumentError ModifiedBandDepth(integration=:simpson)
    @test_throws ArgumentError BandDepth(definition=:strict)
    @test_throws ArgumentError FraimanMuniz(integration=:mean)
    @test_throws ArgumentError ModalDepth(bandwidth=1, quantile=0.2)
    @test_throws ArgumentError ModalDepth(bandwidth=0)
    @test_throws ArgumentError ModalDepth(quantile=1.5)
    @test_throws ArgumentError RandomProjectionDepth(nproj=0)
    @test_throws ArgumentError RandomProjectionDepth(projections=[0.0 0.0 0.0])
    @test_throws ArgumentError RandomProjectionDepth(projections=[NaN 0.0 1.0])
    @test_throws DimensionMismatch RandomProjectionDepth(nproj=2, projections=[1.0 0.0 1.0])
    @test_throws DimensionMismatch depth(fd, RandomProjectionDepth(projections=[1.0 0.0]))
    flat = GridFunctionalData(zeros(3, 3), grid)
    @test_throws ArgumentError depth(flat, ModalDepth())
    mfd = MultivariateFunctionalData([fd, fd])
    @test_throws DimensionMismatch depth(mfd, ModifiedBandDepth(); weights=[1.0])
    @test_throws ArgumentError depth(mfd, ModifiedBandDepth(); weights=[0.6, 0.6])
    @test_throws ArgumentError depth(mfd, ModifiedBandDepth(); weights=[1.5, -0.5])
    @test_throws ArgumentError depth(mfd, FraimanMuniz(); weights=[0.5, 0.5])
    @test_throws ArgumentError depth(mfd, FraimanMuniz(scale=true))
    @test_throws ArgumentError depth(mfd, FraimanMuniz())  # identical components: singular covariance
    coarse = GridFunctionalData([0.0 1.0; 1.0 0.0; 2.0 2.0], [0.0, 1.0])
    @test_throws ArgumentError depth(MultivariateFunctionalData([fd, coarse]), RandomProjectionDepth())
    @test_throws ArgumentError depth(mfd, ModifiedBandDepth(); fpca=(; ncomponents=1))
    @test_throws DimensionMismatch depth(mfd, MultivariateFunctionalData([fd]), ModifiedBandDepth())
    @test_throws ArgumentError depth(fd, mfd, ModifiedBandDepth())
    @test_throws ArgumentError depth(mfd, fd, ModifiedBandDepth())
    @test_throws DimensionMismatch depth(mfd, ModifiedBandDepth(); grid=[grid])
    irregular = IrregularFunctionalData([[0.0, 0.5], [0.5, 1.0], [0.0, 1.0]], [[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
    @test_throws ArgumentError modified_epigraph_index(irregular)
    @test_throws ArgumentError modified_epigraph_index(fd; integration=:mean)
    @test_throws ArgumentError modified_hypograph_index(fd; weights=[1.0])
    basis = BasisFunctionalData([1.0 0.0 0.0; 0.0 1.0 0.0], FourierBasis((0.0, 1.0), 3))
    @test_throws ArgumentError depth(basis, ModifiedBandDepth(); grid=[0.0, 2.0])
    @test_throws ArgumentError depth(basis, basis, ModifiedBandDepth())
end
