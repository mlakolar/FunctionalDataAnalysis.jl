import StatsAPI

@testset "canonical names and compatibility" begin
    for (old, canonical) in ((:basismatrix, :basis_matrix), (:modesplot, :modes_plot),
                            (:screeplot, :scree_plot), (:covplot, :cov_plot),
                            (:designplot, :design_plot), (:fitplot, :fit_plot),
                            (:bandplot, :band_plot))
        @test getproperty(FunctionalDataAnalysis, old) ===
            getproperty(FunctionalDataAnalysis, canonical)
    end
    grid = collect(range(0, 1; length=7))
    values = [sin(i * t) + i * t for i in 1:6, t in grid]
    fd = GridFunctionalData(values, grid)
    canonical = StatsAPI.fit(FPCA, fd; ncomponents=1, max_k=2)
    legacy = StatsAPI.fit(FPCA, fd; ncomponents=1, maxK=2)
    @test scores(legacy) == scores(canonical)
    @test eigvals(legacy) == eigvals(canonical)
    @test legacy.settings.maxK == legacy.settings.max_k == 2
    @test legacy.selection.maxK == legacy.selection.max_k == 2
    @test_throws ArgumentError StatsAPI.fit(FPCA, fd; max_k=2, maxK=2)
    b = FourierBasis((0, 1), 3)
    basis_fd = BasisFunctionalData([(-1)^i * (i + j) for i in 1:6, j in 1:3], b)
    basis_new = StatsAPI.fit(FPCA, basis_fd; ncomponents=1, max_k=2)
    basis_old = StatsAPI.fit(FPCA, basis_fd; ncomponents=1, maxK=2)
    @test scores(basis_old) == scores(basis_new)
    @test basis_old.settings.maxK == basis_old.settings.max_k == 2
    @test_throws ArgumentError StatsAPI.fit(FPCA, basis_fd; max_k=2, maxK=2)
    cross_new = StatsAPI.fit(FSVD, fd, fd; ncomponents=1, max_k=2)
    cross_old = StatsAPI.fit(FSVD, fd, fd; ncomponents=1, maxK=2)
    @test cross_old.left_scores == cross_new.left_scores
    @test cross_old.selection.maxK == cross_old.selection.max_k == 2
    @test_throws ArgumentError StatsAPI.fit(FSVD, fd, fd; max_k=2, maxK=2)
    mfd = MultivariateFunctionalData([fd, fd])
    pooled = StatsAPI.fit(FPCA, mfd; basis=:pooled, ncomponents=1, maxK=2)
    @test pooled.settings.max_k == 2
    @test pooled.settings.maxK == 2
end
