@testset "Shared internal helpers preserve public contracts" begin
    FDA = FunctionalDataAnalysis

    @testset "Strict grids and arbitrary evaluation points" begin
        for points in (Float32[0,.25,1], [0.,.25,1.], 0:2, view([0.,.25,1.],:))
            original = copy(points)
            @test FDA._validate_strict_grid(points) === nothing
            @test points == original
        end
        # These integers are distinct, but coincide after Float64 conversion.
        collapsed = [2^53,2^53+1]
        @test_throws r"strictly increasing" quadrature_weights(collapsed)
        @test_throws r"strictly increasing" FDA._validate_strict_grid(collapsed)
        basis = MonomialBasis((0.,1.),1)
        fd = BasisFunctionalData([1. 2.],basis)
        @test evaluate(fd,[.8,.2,.8]) == [2.6 1.4 2.6]
        @test size(evaluate(fd,Float64[])) == (1,0)
        @test_throws r"strictly increasing" to_grid(fd,[.8,.2,.8])
        @test_throws r"at least two" to_grid(fd,[.2])
        custom = [.2,.3,.7]
        fpc = FPCBasis([0.,.25,1.],ones(3,1);weights=view(custom,:))
        @test fpc.weights == custom
        @test fpc.weights != quadrature_weights(fpc.grid)
        custom[1] = 9
        @test fpc.weights[1] == .2
    end

    @testset "Bilinear interpolation on unequal axes" begin
        for T in (Float32,Float64)
            g1, g2 = T[0,.2,1], T[-1,-.4,.1,2]
            polynomial(s,t) = 2+3s-4t+5s*t
            surface = [polynomial(s,t) for s in g1,t in g2]
            cross = CrossCovariance(g1,g2,surface,quadrature_weights(g1),
                quadrature_weights(g2),zeros(T,3),zeros(T,4),3,nothing)
            for s in T[0,.1,.2,.6,1], t in T[-1,-.7,.1,1,2]
                @test evaluate(cross,s,t) ≈ polynomial(s,t) rtol=5eps(T) atol=5eps(T)
            end
            for (s,t) in ((T(-.1),T(0)),(T(0),T(3)),(T(NaN),T(0)),(T(0),T(Inf)))
                @test_throws r"inside their stored grid ranges" evaluate(cross,s,t)
            end
            # Endpoints and mixed query precision retain the same result type.
            @test (@inferred FDA._bilinear_surface(g1,g2,surface,0.,2.)) isa Float64
        end
    end

    @testset "Mean and covariance use the same copied-RNG curve folds" begin
        grid = [-1.,-.3,.2,.7,1.]
        values = [sin(i*t)+.2i*t for i in 1:7,t in grid]
        fd = GridFunctionalData(values,grid)
        kernel = KernelSmoothing.GaussianKernel()
        for rng in (MersenneTwister(48),Xoshiro(48)), folds in (3,7)
            selector = CurveCV([1.,1.5];folds,rng)
            snapshot = deepcopy(selector.rng)
            expected = collect(1:7)
            if folds != 7
                for (j,i) in enumerate(randperm(deepcopy(snapshot),7))
                    expected[i] = mod1(j,folds)
                end
            end
            mean = StatsAPI.fit(SmoothedMean,fd;grid,bandwidth=selector,kernel)
            covariance = StatsAPI.fit(SmoothedCovariance,fd,mean;grid,
                bandwidth=selector,diagonal_bandwidth=1.,kernel)
            @test mean.selection.assignments == expected
            @test covariance.selection.assignments == expected
            @test mean.selection.bandwidths == covariance.selection.bandwidths == selector.bandwidths
            @test mean.bandwidth == selector.bandwidths[argmin(mean.selection.errors)]
            @test covariance.bandwidth == selector.bandwidths[argmin(covariance.selection.errors)]
            @test rand(deepcopy(selector.rng),8) == rand(deepcopy(snapshot),8)
            if folds == 3
                @test sort([count(==(j),expected) for j in 1:folds]) == [2,2,3]
            else
                @test expected == 1:7
            end
            for s in (-1.,.1,1.), t in (-1.,.6,1.)
                @test evaluate(covariance,s,t) == FDA._pace_bilinear(
                    covariance.grid,covariance.surface,s,t)
            end
        end
    end

    @testset "Reconstruction validation keeps wrapper diagnostics" begin
        grid = [0.,.2,.5,.7,1.]
        values = [sin(.5i)*(1+t)+cos(.7i)*t^2 for i in 1:7,t in grid]
        fd = GridFunctionalData(values,grid;ids=collect(1:7))
        fitted = StatsAPI.fit(FPCA,fd;ncomponents=2)
        d = derivative(fitted;bandwidth=2.)
        for (fit,label) in ((fitted,"reconstruction requires"),(d,"derivative reconstruction needs"))
            @test_throws Regex(label*" at least one score row") reconstruct(fit,zeros(0,2))
            @test ids(reconstruct(fit)) == ids(fd)
            @test ids(reconstruct(fit,view(scores(fit),:,:))) === nothing
            @test reconstruct(fit,view(scores(fit),:,:)).values == reconstruct(fit).values
        end
        restored = reconstruct(d)
        @test restored.metadata.derivative == d.order
        @test restored.metadata.settings == d.settings
        @test restored.metadata.settings !== d.settings
    end
end
