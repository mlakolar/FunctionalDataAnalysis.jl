import StatsAPI
import RecipesBase

# Orthogonal deterministic columns give independent least-squares answers.
function regression_fixture()
    signs(j) = [isodd(count_ones((i - 1) & j)) ? -1.0 : 1.0 for i in 1:16]
    a, b, c, z, error = 2signs(1), signs(2), 3signs(4), signs(8), 0.2signs(3)
    grid = [0.0, 0.1, 0.35, 0.65, 0.9, 1.0]
    weights = quadrature_weights(grid)
    slope = (grid .- 0.5) ./ sqrt(dot(weights, (grid .- 0.5).^2))
    mu = 2 .+ 3slope
    labels = string.(1:16)
    x1 = GridFunctionalData(transpose(mu) .+ a .+ b * transpose(slope), grid; ids=labels)
    x2 = GridFunctionalData(repeat(-1 .+ c, 1, length(grid)), grid; ids=labels)
    y = 4 .+ 2z .+ 1.5a .- 0.75b .+ 0.4c .+ error
    (; x1, x2, y, z, a, b, c, error, grid, weights, slope, mu, signs)
end

@testset "FPC scalar-on-function regression" begin
    @testset "Analytic multivariate least squares on a nonuniform grid" begin
        fixture = regression_fixture()
        (; x1, x2, y, z, a, b, c, error, grid, slope) = fixture
        f = @inferred StatsAPI.fit(FPCRegression, (x1, x2), y;
            scalar_covariates=z, ncomponents=(2, 1),
            fpca_options=((; method=:dense, max_k=2), (; method=:dense, max_k=1)))
        expected = y - error
        sigma2 = sum(abs2, error) / 11
        covariance = Diagonal(sigma2 ./ [16, 16, 64, 16, 144])
        @test (@inferred StatsAPI.nobs(f)) == 16
        @test StatsAPI.dof(f) == 5
        @test StatsAPI.dof_residual(f) == 11
        @test StatsAPI.coef(f) ≈ [4, 2, 1.5, -0.75, -0.4] atol=2e-13
        @test StatsAPI.fitted(f) ≈ expected atol=2e-13
        @test StatsAPI.residuals(f) ≈ error atol=2e-13
        @test f.sigma2 ≈ sigma2 rtol=2e-13
        @test StatsAPI.vcov(f) ≈ covariance atol=2e-14
        @test StatsAPI.stderror(f) ≈ sqrt.(diag(covariance)) rtol=2e-13
        @test StatsAPI.r2(f) ≈ 1 - sum(abs2, error) / sum(abs2, y .- 4)
        beta = @inferred coefficient_functions(f)
        @test beta isa Tuple
        @test length(beta) == 2
        @test evaluate(beta[1], grid) ≈ transpose(1.5 .- 0.75slope) atol=2e-13
        @test evaluate(beta[2], grid) ≈ fill(0.4, 1, length(grid)) atol=2e-13
        @test (@inferred StatsAPI.predict(f, (x1, x2); scalar_covariates=z)) ≈ expected atol=2e-13
        @test StatsAPI.predict(f, MultivariateFunctionalData([x1, x2]);
            scalar_covariates=reshape(z, :, 1)) ≈ expected atol=2e-13

        confidence = StatsAPI.predict(f, (x1, x2); scalar_covariates=z, interval=:confidence)
        prediction = StatsAPI.predict(f, (x1, x2); scalar_covariates=z, interval=:prediction)
        design = hcat(ones(16), z, a, b, -c)
        variance = diag(design * covariance * transpose(design))
        @test confidence.estimate ≈ expected atol=2e-13
        @test confidence.se.^2 ≈ variance rtol=2e-12
        @test prediction.se.^2 ≈ variance .+ sigma2 rtol=2e-12
        # R qt(0.975, df=11); conditional OLS intervals use residual degrees of freedom.
        critical = 2.200985160082949
        @test confidence.lower ≈ expected - critical .* sqrt.(variance) rtol=2e-12
        @test confidence.upper ≈ expected + critical .* sqrt.(variance) rtol=2e-12
        bands = fitted_bands(f)
        @test bands.estimate ≈ confidence.estimate
        @test bands.se ≈ confidence.se
        betabands = coefficient_bands(f)
        @test length(betabands) == 2
        @test vec(betabands[1].estimate) ≈ 1.5 .- 0.75slope
        @test vec(betabands[1].se).^2 ≈ sigma2 .* (1 / 64 .+ slope.^2 / 16)
        @test vec(betabands[2].se).^2 ≈ fill(sigma2 / 144, length(grid))
        for (band, beta_j) in zip(betabands, beta)
            @test band.lower ≈ band.estimate - critical .* band.se
            @test band.upper ≈ band.estimate + critical .* band.se
            @test band.grid == grid
        end

        # The model owns its inputs, including supplied scores and harmonics.
        fitted_before = copy(StatsAPI.fitted(f))
        covariance_before = copy(StatsAPI.vcov(f))
        y .= 999
        z .= 999
        x1.values .= 999
        x2.values .= 999
        @test StatsAPI.fitted(f) == fitted_before
        @test StatsAPI.vcov(f) == covariance_before
        @test evaluate(coefficient_functions(f)[1], grid) ≈ transpose(1.5 .- 0.75slope)
        displayed = sprint(show, f)
        for choice in ("FPCRegression", "selection", "sigma2")
            @test occursin(choice, displayed)
        end
    end

    @testset "Supplied FPCA, subject projection and explicit interpolation" begin
        fixture = regression_fixture()
        (; x1, y, a, b, error, grid, slope, mu) = fixture
        response = 3 .+ 1.5a .- 0.75b .+ error
        fpca = StatsAPI.fit(FPCA, x1; ncomponents=2, max_k=2)
        f = @inferred StatsAPI.fit(FPCRegression, x1, response; fpca)
        @test StatsAPI.predict(f, x1) ≈ response - error atol=2e-13
        fresh = StatsAPI.fit(FPCRegression, x1, response; ncomponents=2,
            fpca_options=(; method=:dense, max_k=2))
        @test StatsAPI.coef(f) ≈ StatsAPI.coef(fresh)

        # A supplied fit can project different curves; it must not reuse training scores.
        shifted = GridFunctionalData(x1.values .+ 0.5, grid; ids=ids(x1))
        projected = StatsAPI.fit(FPCRegression, shifted, response; fpca)
        @test StatsAPI.coef(projected)[1] ≈ 3 - 1.5 * 0.5 atol=2e-13
        @test StatsAPI.predict(projected, shifted) ≈ response - error atol=2e-13

        indices = [3, 1, 7]
        @test StatsAPI.predict(f, x1[indices]) ≈ (response - error)[indices] atol=2e-13
        irregular = to_irregular(x1)
        @test_throws ArgumentError StatsAPI.predict(f, irregular)
        @test StatsAPI.predict(f, irregular; evaluator=PiecewiseLinear()) ≈ response - error atol=2e-13
        @test StatsAPI.predict(f, irregular; evaluator=NoInterpolation()) ≈ response - error atol=2e-13
        fine = collect(range(0, 1; length=21))
        scale = sqrt(dot(fixture.weights, (grid .- 0.5).^2))
        newa, newb = [0.5, -1.0], [0.2, 0.4]
        values = transpose(2 .+ 3 .* (fine .- 0.5) ./ scale) .+
            newa .+ newb * transpose((fine .- 0.5) ./ scale)
        newdata = GridFunctionalData(values, fine)
        @test_throws DimensionMismatch StatsAPI.predict(f, newdata)
        @test StatsAPI.predict(f, newdata; evaluator=PiecewiseLinear()) ≈
            3 .+ 1.5newa .- 0.75newb atol=2e-13
        fpca.scores .= 999
        fpca.basis.values .= 999
        @test StatsAPI.predict(f, x1) ≈ response - error atol=2e-13
    end

    @testset "Exact basis covariates" begin
        signs = regression_fixture().signs
        basis = FourierBasis((0, 1), 3)
        latent = hcat(2signs(1), signs(2), 0.5signs(4))
        coefs = latent .+ transpose([1.0, 2, -3])
        x = BasisFunctionalData(coefs, basis)
        error = 0.1signs(8)
        response = 3 .+ latent * [1.2, -0.8, 0.4] .+ error
        f = @inferred StatsAPI.fit(FPCRegression, x, response; ncomponents=3)
        @test StatsAPI.fitted(f) ≈ response - error atol=2e-13
        @test (@inferred StatsAPI.predict(f, x)) ≈ response - error atol=2e-13
        points = [0.0, 0.137, 0.45, 0.86, 1.0]
        beta = only(coefficient_functions(f))
        @test vec(evaluate(beta, points)) ≈ basis_matrix(basis, points) * [1.2, -0.8, 0.4] atol=2e-13
        @test StatsAPI.predict(f, x[[4, 1]]) ≈ (response - error)[[4, 1]] atol=2e-13
        # Raw curves against a basis fit: explicit numerical projection on the work grid.
        fine = collect(range(0.0, 1.0; length=2001))
        raw = GridFunctionalData(evaluate(x[[4, 1]], fine), fine)
        @test StatsAPI.predict(f, raw; evaluator=PiecewiseLinear()) ≈
            (response - error)[[4, 1]] rtol=1e-3
        # Basis curves against a raw-data fit are evaluated exactly on its grid.
        grid = collect(range(0.0, 1.0; length=101))
        dense = StatsAPI.fit(FPCRegression, GridFunctionalData(evaluate(x, grid), grid), response;
            ncomponents=3, fpca_options=(; method=:dense))
        @test StatsAPI.predict(dense, x) ≈ StatsAPI.fitted(dense) atol=1e-10
        @test sprint(show, MIME"text/plain"(), dense) == sprint(show, dense)
    end

    @testset "Sparse PACE scores preserve the fitted conditional expectation" begin
        fixture = regression_fixture()
        (; a, error, grid) = fixture
        times = [[0.0, 0.4, 1.0] for _ in eachindex(a)]
        values = [fill(score, 3) for score in a]
        x = IrregularFunctionalData(times, values; domain=(0.0, 1.0))
        fpca = StatsAPI.fit(FPCA, x; grid, ncomponents=1, max_k=1,
            mean=(; grid, values=zeros(length(grid))),
            covariance=(; grid, surface=fill(2.0, length(grid), length(grid))),
            noise_variance=0.3)
        response = 1 .+ 0.7a .+ error
        shrinkage = 20 / 21
        @test vec(scores(fpca)) ≈ shrinkage .* a
        f = @inferred StatsAPI.fit(FPCRegression, x, response; fpca)
        @test StatsAPI.coef(f) ≈ [1, 0.7 / shrinkage] atol=2e-13
        @test f.sigma2 ≈ sum(abs2, error) / 14 rtol=2e-13
        @test vec(evaluate(only(coefficient_functions(f)), grid)) ≈ fill(0.7 / shrinkage, length(grid))
        @test StatsAPI.predict(f, x) ≈ response - error atol=2e-13
        newx = IrregularFunctionalData([[0.2, 0.8], [0.1, 0.6]],
            [[0.5, 0.5], [-1.0, -1.0]]; domain=(0.0, 1.0))
        expected = 1 .+ (0.7 / shrinkage) .* (4 / 4.3) .* [0.5, -1.0]
        @test StatsAPI.predict(f, newx) ≈ expected atol=2e-13
    end

    @testset "Invalid response, scalar design and prediction inputs" begin
        (; x1, x2, y, z, grid) = regression_fixture()
        options = (; ncomponents=2, fpca_options=(; max_k=2))
        @test_throws DimensionMismatch StatsAPI.fit(FPCRegression, x1, y[1:15]; options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, [NaN; y[2:end]]; options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, [Inf; y[2:end]]; options...)
        @test_throws DimensionMismatch StatsAPI.fit(FPCRegression, x1, y;
            scalar_covariates=z[1:15], options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y;
            scalar_covariates=fill(NaN, 16), options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y;
            scalar_covariates=ones(16), options...)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y;
            scalar_covariates=hcat(z, z), options...)
        @test_throws DimensionMismatch StatsAPI.fit(FPCRegression, (x1, x2[1:15]), y)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, (), y)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; ncomponents=0)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; ncomponents=true)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; selection=:unknown)
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, x1, y; ncomponents=3,
            fpca_options=(; max_k=2))
        @test_throws DimensionMismatch StatsAPI.fit(FPCRegression, (x1, x2), y;
            ncomponents=(1,))
        reversed_ids = GridFunctionalData(x2.values, grid; ids=reverse(ids(x2)))
        @test_throws ArgumentError StatsAPI.fit(FPCRegression, (x1, reversed_ids), y)

        f = StatsAPI.fit(FPCRegression, x1, y; scalar_covariates=z, options...)
        @test_throws DimensionMismatch StatsAPI.predict(f, x1)
        @test_throws DimensionMismatch StatsAPI.predict(f, x1; scalar_covariates=z[1:15])
        @test_throws DimensionMismatch StatsAPI.predict(f, x1; scalar_covariates=hcat(z, z))
        @test_throws ArgumentError StatsAPI.predict(f, x1; scalar_covariates=fill(Inf, 16))
        @test_throws DimensionMismatch StatsAPI.predict(f, (x1, x2); scalar_covariates=z)
        @test_throws ArgumentError StatsAPI.predict(f, x1; scalar_covariates=z, interval=:unknown)
        @test_throws ArgumentError StatsAPI.predict(f, x1; scalar_covariates=z,
            interval=:confidence, alpha=0)
        @test_throws ArgumentError coefficient_bands(f; alpha=1)
        @test_throws ArgumentError fitted_bands(f; alpha=NaN)
    end
end

using DelimitedFiles
_m7_read(name) = readdlm(joinpath(@__DIR__, "data", "m7_$(name).csv"), ','; comments=true)
_m7_golden(name) = readdlm(joinpath(@__DIR__, "data", "m7_$(name).csv"), ',', Any; comments=true)
_m7_ref(table, quantity) = Float64.(table[table[:,1] .== quantity,4])
_m7_relative(actual, expected) = norm(actual-expected)/norm(expected)

function regression_golden_checks(f, newx, golden; scalar_covariates=nothing)
    ref(q) = _m7_ref(golden,q)
    for (actual,q) in ((StatsAPI.fitted(f),"fitted"),(StatsAPI.residuals(f),"residual"),
            ([f.sigma2],"sigma2"),([StatsAPI.r2(f)],"r2"),
            ([StatsAPI.coef(f)[1]],"centered_intercept"))
        @test _m7_relative(actual,ref(q)) < 1e-8
    end
    @test StatsAPI.dof_residual(f) == only(ref("df_residual"))
    for (j,beta) in enumerate(coefficient_functions(f))
        @test _m7_relative(vec(evaluate(beta,f.fpca[j].grid)),ref("beta$j")) < 1e-8
        band = coefficient_bands(f)[j]
        for (actual,q) in ((band.se,"beta$(j)_se"),(band.lower,"beta$(j)_lower"),
                (band.upper,"beta$(j)_upper"))
            @test _m7_relative(actual,ref(q)) < 1e-8
        end
    end
    @test _m7_relative(predict(f,newx;scalar_covariates),ref("prediction")) < 1e-8
    confidence = predict(f,newx;scalar_covariates,interval=:confidence)
    prediction = predict(f,newx;scalar_covariates,interval=:prediction)
    for (actual,q) in ((confidence.se,"prediction_se"),
            (confidence.lower,"confidence_lower"),(confidence.upper,"confidence_upper"),
            (prediction.lower,"prediction_lower"),(prediction.upper,"prediction_upper"),
            ([confidence.critical_value],"critical_value"))
        @test _m7_relative(actual,ref(q)) < 1e-8
    end
end

@testset "FPC regression R golden values" begin
    grid = vec(_m7_read("dense_grid"))
    x1 = GridFunctionalData(_m7_read("dense_input1"),grid)
    x2 = GridFunctionalData(_m7_read("dense_input2"),grid)
    new1 = GridFunctionalData(_m7_read("dense_heldout1"),grid)
    new2 = GridFunctionalData(_m7_read("dense_heldout2"),grid)
    response = _m7_read("dense_response")
    y,z,ysingle = eachcol(response)
    znew = vec(_m7_read("dense_heldout_scalar"))
    # fdapace FLM uses the dense route; FPCRegression's default is FACE, so the
    # goldens name the reference route explicitly.
    multi = StatsAPI.fit(FPCRegression,(x1,x2),y;scalar_covariates=z,
        ncomponents=(3,2),fpca_options=((;method=:dense,max_k=3),(;method=:dense,max_k=2)))
    # This grid is shorter than 35 + 3 points, so the default FACE fit uses
    # length(grid) - p - 1 knot intervals.
    @test length(grid) <= 38
    automatic = @inferred StatsAPI.fit(FPCRegression,(x1,x2),y;scalar_covariates=z)
    @test automatic.settings.fpca_method == (:face,:face)
    @test automatic.settings.fpca_method_source == (:short_grid_knots,:short_grid_knots)
    @test automatic.settings.fpca_knots == [length(grid)-4,length(grid)-4]
    @test StatsAPI.coef(automatic) == StatsAPI.coef(StatsAPI.fit(FPCRegression,(x1,x2),y;
        scalar_covariates=z,fpca_options=(;method=:face,face=(;knots=length(grid)-4))))
    golden = _m7_golden("fdapace_dense")
    regression_golden_checks(multi,(new1,new2),golden;scalar_covariates=znew)
    @test _m7_relative([StatsAPI.coef(multi)[2]],_m7_ref(golden,"scalar_coefficient")) < 1e-8
    # R's score order is intercept, functional scores, scalar; Julia's is
    # intercept, scalar, functional scores. Compare the same parameterization.
    permutation = [1,7,2,3,4,5,6]
    rcoef = _m7_ref(golden,"score_coefficient")[permutation]
    rcov = reshape(_m7_ref(golden,"coefficient_covariance"),7,7)[permutation,permutation]
    @test _m7_relative(StatsAPI.coef(multi),rcoef) < 1e-8
    @test _m7_relative(StatsAPI.vcov(multi),rcov) < 1e-8
    single = StatsAPI.fit(FPCRegression,x1,ysingle;ncomponents=3,fpca_options=(;method=:dense,max_k=3))
    regression_golden_checks(single,new1,_m7_golden("fdausc_dense"))

    read_m3(name) = readdlm(joinpath(@__DIR__,"data","m3_sparse_$name.csv"),',';comments=true)
    input,mu,surface,held = read_m3("input"),read_m3("mean"),read_m3("surface"),read_m3("heldout")
    make_sparse(rows) = IrregularFunctionalData(
        [rows[rows[:,1].==i,2] for i in 1:Int(maximum(rows[:,1]))],
        [rows[rows[:,1].==i,3] for i in 1:Int(maximum(rows[:,1]))])
    sparse,new_sparse = make_sparse(input),make_sparse(held)
    options = (;grid=mu[:,1],mean=(;grid=mu[:,1],values=mu[:,2]),
        covariance=(;grid=mu[:,1],surface),noise_variance=only(read_m3("noise")),max_k=5)
    sparse_fit = @inferred StatsAPI.fit(FPCRegression,sparse,vec(_m7_read("sparse_response"));
        ncomponents=3,fpca_options=options)
    regression_golden_checks(sparse_fit,new_sparse,_m7_golden("fdapace_sparse"))

    selection_ref = _m7_golden("selection")
    for (selector,q) in ((AIC(),"AIC"),(BIC(),"BIC"),(:loocv,"loocv"))
        selected = StatsAPI.fit(FPCRegression,x1,ysingle;selection=selector,
            fpca_options=(;method=:dense,max_k=3))
        @test selected.selection.candidates == [[1],[2],[3]]
        @test selected.selection.cap == [3]
        @test _m7_relative(selected.selection.values,_m7_ref(selection_ref,q)) < 1e-8
        @test only(ncomponents(selected)) == argmin(_m7_ref(selection_ref,q))
        @test all(isempty,selected.selection.failures)
    end
end

@testset "FPC regression selection, precision and validation" begin
    (;x1,x2,y,z,grid) = regression_fixture()
    for selector in (AIC(),BIC(),:loocv)
        chosen = @inferred StatsAPI.fit(FPCRegression,(x1,x2),y;scalar_covariates=z,
            selection=selector,fpca_options=((;max_k=2),(;max_k=1)))
        @test chosen.counts == [2,1]
        @test chosen.selection.candidates == [[1,1],[2,1]]
    end
    supplied = StatsAPI.fit(FPCA,x1;ncomponents=2,max_k=2)
    truncated = @inferred StatsAPI.fit(FPCRegression,x1,y;fpca=supplied,ncomponents=1)
    @test ncomponents(truncated) == [1]
    @test truncated.settings.fpca_source == (:supplied,)
    @test StatsAPI.fit(FPCRegression,x1,y;fpca=supplied,ncomponents=FVE(.5)).counts == [1]
    @test StatsAPI.fit(FPCRegression,x1,y;fpca_options=(;max_k=2)).settings.fpca_source == (:fitted,)
    for kw in ((;selection=:loocv,fpca=supplied),
            (;selection=AIC(),ncomponents=1),
            (;fpca=supplied,fpca_options=(;max_k=2)),
            (;fpca=42), (;fpca_options=42),
            (;fpca_options=(;ncomponents=true)), (;fpca_options=(;max_k=true)),
            (;ncomponents=1,fpca_options=(;ncomponents=1)),
            (;fpca=supplied,ncomponents=AIC()))
        @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,y;kw...)
    end
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,ones(16);ncomponents=1)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,y;scalar_covariates="bad")
    nullable = Vector{Union{Missing,Float64}}(y)
    nullable_z = Vector{Union{Missing,Float64}}(z)
    @test StatsAPI.fitted(@inferred StatsAPI.fit(FPCRegression,x1,nullable;scalar_covariates=nullable_z,
        ncomponents=2)) ≈ StatsAPI.fitted(StatsAPI.fit(FPCRegression,x1,y;scalar_covariates=z,ncomponents=2))
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,fill(missing,16))
    @test_throws r"response.*missing.*index 3" StatsAPI.fit(FPCRegression,x1,
        Union{Missing,Float64}[y[1:2]; missing; y[4:end]])
    nullable[1] = missing
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,nullable;ncomponents=2)
    nullable_z[1] = missing
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,y;scalar_covariates=nullable_z,ncomponents=2)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,(x1,x1),y;selection=AIC(),fpca_options=(;max_k=2))
    partial = StatsAPI.fit(FPCRegression,x1[1:4],y[1:4];selection=:loocv,
        fpca_options=(;max_k=2))
    @test partial.counts == [1]
    @test partial.selection.values[2] == Inf
    @test occursin("degrees of freedom",partial.selection.failures[2])
    f = StatsAPI.fit(FPCRegression,x1,y;ncomponents=1)
    coefficient_interval = only(coefficient_bands(f))
    coordinates = @inferred band_data(coefficient_interval)
    @test vec(coordinates.values) == coefficient_interval.estimate
    @test coordinates.grid == coefficient_interval.grid && coordinates.grid !== coefficient_interval.grid
    @test coordinates.target == :coefficient
    plotted = RecipesBase.apply_recipe(Dict{Symbol,Any}(),
        FunctionalDataAnalysis._plot_payload(Val(:band),coordinates))
    @test any(series -> get(series.plotattributes,:label,nothing) == "Coefficient (interval)",plotted)
    @test_throws ArgumentError predict(f,x1;evaluator=:linear)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,1.0,y)
    @test_throws ArgumentError StatsAPI.fit(FPCRegression,x1,"response")
    onecomponent = StatsAPI.fit(FPCA,x1;ncomponents=1,max_k=2)
    @test_throws r"FVE threshold" StatsAPI.fit(FPCRegression,x1,y;fpca=onecomponent,ncomponents=FVE(0.9))
    @test_throws ArgumentError coefficient_bands(f;grid="bad")
    @test_throws ArgumentError fitted_bands(f;interval=:band)
    @test_throws ArgumentError predict(f,x1;interval=:band)
    @test_throws DimensionMismatch coefficient_bands(f;grid=(grid,grid))
    small = GridFunctionalData(Float32.(x1.values),Float32.(grid))
    f32 = @inferred StatsAPI.fit(FPCRegression,small,Float32.(y);ncomponents=2,
        fpca_options=(;method=:dense,max_k=2))
    @test eltype(StatsAPI.coef(f32)) == Float32
    @test eltype(StatsAPI.vcov(f32)) == Float32
    @test (@inferred predict(f32,small)) ≈ Float32.(StatsAPI.fitted(
        StatsAPI.fit(FPCRegression,x1,y;ncomponents=2,fpca_options=(;max_k=2)))) rtol=1e-5
    for row in eachrow(_m7_read("t_quantiles"))
        @test abs(FunctionalDataAnalysis._regression_tcritical(Int(row[1]),row[2])/row[3]-1) < 1e-8
    end
    # Analytic Cauchy tail, without squaring a huge quantile and overflowing.
    @test FunctionalDataAnalysis._regression_tcritical(1,1e-12) ≈ 2e12/pi rtol=1e-10
    tecator = load_dataset(:tecator;sample=true)
    example = StatsAPI.fit(FPCRegression,tecator.data,tecator.covariates.Fat;ncomponents=1)
    @test all(isfinite,predict(example,tecator.data;interval=:prediction).lower)
end

# Smooth deterministic curves with rough deterministic noise on a grid long
# enough for FACE's default 35-interval cubic basis.
function regression_noisy_dense(n=24, m=51)
    grid = collect(range(0.0, 1.0; length=m))
    a, b, c = cos.(2pi .* (1:n) ./ n), sin.(4pi .* (1:n) ./ n), cos.(6pi .* (1:n) ./ n .+ 0.3)
    noise = 0.3 .* ((sin.(12.9898 .* (1:n) .+ 78.233 .* transpose(1:m)) .* 43758.5453) .% 1)
    x = GridFunctionalData(1 .+ 2a .* transpose(sqrt(2) .* sinpi.(grid)) .+
        b .* transpose(sqrt(2) .* cospi.(2grid)) .+ 0.5c .* transpose(sqrt(2) .* sinpi.(3grid)) .+ noise, grid)
    y = 1 .+ 2a .- b .+ 0.3c .+ 0.1 .* sin.(7.0 .* (1:n))
    (; x, y, grid)
end

@testset "FPC regression FACE default and selection cap" begin
    (; x, y) = regression_noisy_dense()
    automatic = @inferred StatsAPI.fit(FPCRegression, x, y)
    face = StatsAPI.fit(FPCRegression, x, y; fpca_options=(; method=:face))
    dense = StatsAPI.fit(FPCRegression, x, y; fpca_options=(; method=:dense))
    @test automatic.settings.fpca_method == (:face,)
    @test automatic.settings.fpca_method_source == (:regression_default,)
    @test automatic.settings.fpca_options == ((;),)
    @test only(automatic.fpca).method === :face
    @test StatsAPI.coef(automatic) == StatsAPI.coef(face)
    @test automatic.counts == face.counts
    @test face.settings.fpca_method_source == (:fpca_options,)
    # The documented override restores the previous dense route exactly.
    @test dense.settings.fpca_method == (:dense,)
    @test StatsAPI.coef(dense) == StatsAPI.coef(StatsAPI.fit(FPCRegression, x, y;
        fpca=StatsAPI.fit(FPCA, x; method=:dense)))
    @test occursin("fpca_method = (:face,)", sprint(show, automatic))
    # Options for other routes without a method name throw instead of switching route.
    @test_throws r"method=:face by default.*presmooth" StatsAPI.fit(FPCRegression, x, y;
        fpca_options=(; presmooth=LocalPolynomial(; bandwidth=0.1)))
    @test_throws r"method=:face by default.*mean_bandwidth" StatsAPI.fit(FPCRegression, x, y;
        fpca_options=(; mean_bandwidth=0.1))
    @test StatsAPI.fit(FPCRegression, x, y; fpca_options=(; method=:pace, mean_bandwidth=0.2,
        covariance_bandwidth=0.2)).settings.fpca_method == (:pace,)
    # A grid too short for FACE's default 35 knot intervals uses
    # length(grid) - p - 1 intervals; explicit knots, p or a method win.
    short = GridFunctionalData(x.values[:, 1:2:end], x.grid[1:2:end])
    @test length(short.grid) == 26
    reduced = @inferred StatsAPI.fit(FPCRegression, short, y)
    @test reduced.settings.fpca_method == (:face,)
    @test reduced.settings.fpca_method_source == (:short_grid_knots,)
    @test reduced.settings.fpca_knots == [22]
    @test only(reduced.fpca).settings.knots == 22
    @test reduced.settings.fpca_options == ((;),)
    @test StatsAPI.coef(reduced) == StatsAPI.coef(StatsAPI.fit(FPCRegression, short, y;
        fpca_options=(; method=:face, face=(; knots=22))))
    shown = sprint(show, reduced)
    @test occursin("fpca_method_source = (:short_grid_knots,)", shown)
    @test occursin("fpca_knots = $(repr(Union{Nothing,Int}[22]))", shown)
    small_face = StatsAPI.fit(FPCRegression, short, y; fpca_options=(; face=(; knots=10)))
    @test small_face.settings.fpca_knots == [10]
    @test small_face.settings.fpca_method_source == (:regression_default,)
    quadratic = StatsAPI.fit(FPCRegression, short, y; fpca_options=(; face=(; p=2)))
    @test quadratic.settings.fpca_knots == [23]
    @test quadratic.settings.fpca_method_source == (:short_grid_knots,)
    @test StatsAPI.fit(FPCRegression, short, y; fpca_options=(; method=:face, face=(; knots=10))
        ).settings.fpca_method_source == (:fpca_options,)
    @test StatsAPI.fit(FPCRegression, short, y; fpca_options=(; method=:dense)).settings.fpca_knots == [nothing]
    # The boundary: 38 points use 34 intervals, 39 points FACE's default 35.
    for (J, knots, source) in ((38, 34, :short_grid_knots), (39, 35, :regression_default))
        g = collect(range(0.0, 1.0; length=J))
        boundary = StatsAPI.fit(FPCRegression, GridFunctionalData(x.values[:, 1:J], g), y)
        @test boundary.settings.fpca_knots == [knots]
        @test boundary.settings.fpca_method_source == (source,)
    end
    # The smallest smoothable grid has p + 2 points (one knot interval).
    tiny = GridFunctionalData(x.values[:, 1:5], collect(range(0.0, 1.0; length=5)))
    @test StatsAPI.fit(FPCRegression, tiny, y).settings.fpca_knots == [1]
    @test_throws r"needs at least p \+ 2 = 5 grid points, got 4.*method=:dense" StatsAPI.fit(
        FPCRegression, GridFunctionalData(x.values[:, 1:4], tiny.grid[1:4]), y)
    @test StatsAPI.fit(FPCRegression, GridFunctionalData(x.values[:, 1:4], tiny.grid[1:4]), y;
        fpca_options=(; method=:dense)).settings.fpca_method == (:dense,)
    # A FACE failure on a default fit names the default and its overrides.
    clustered = GridFunctionalData(x.values, x.grid .^ 3)
    @test_throws r"default FACE fit for complete grid data failed.*rank deficient.*method=:dense" StatsAPI.fit(
        FPCRegression, clustered, y)
    # Masked grids and irregular curves keep PACE; basis curves keep the basis route.
    masked_values = Matrix{Union{Missing,Float64}}(x.values)
    masked_values[1, 2] = missing
    masked = GridFunctionalData(masked_values, x.grid)
    @test sampling_design(masked) != :regular
    masked_fit = StatsAPI.fit(FPCRegression, masked, y; ncomponents=2)
    @test masked_fit.settings.fpca_method == (:pace,)
    @test masked_fit.settings.fpca_method_source == (:fpca_default,)
    irregular_fit = StatsAPI.fit(FPCRegression, to_irregular(x), y; ncomponents=2)
    @test irregular_fit.settings.fpca_method == (:pace,)
    @test irregular_fit.settings.fpca_method_source == (:fpca_default,)
    basis = to_basis(x, BSplineBasis(4, (0.0, 1.0), 8))
    @test StatsAPI.fit(FPCRegression, basis, y; ncomponents=2).settings.fpca_method == (:basis,)
    supplied = StatsAPI.fit(FPCA, x; method=:dense, ncomponents=3)
    supplied_fit = StatsAPI.fit(FPCRegression, x, y; fpca=supplied)
    @test supplied_fit.settings.fpca_method == (:dense,)
    @test supplied_fit.settings.fpca_method_source == (:supplied,)

    # Regression selection searches up to each FPCA's own selection.
    own = ncomponents(StatsAPI.fit(FPCA, x; method=:face))
    for selector in (AIC(), BIC())
        chosen = @inferred StatsAPI.fit(FPCRegression, x, y; selection=selector)
        @test chosen.selection.cap == [own]
        @test chosen.selection.candidates == [[k] for k in 1:own]
        @test only(chosen.counts) <= own
        @test occursin("cap = [$own]", sprint(show, chosen))
    end
    capped = StatsAPI.fit(FPCRegression, x, y; selection=BIC(), fpca_options=(; ncomponents=2))
    @test capped.selection.cap == [2]
    @test capped.selection.candidates == [[1], [2]]
    @test StatsAPI.fit(FPCRegression, x, y; selection=AIC(),
        fpca_options=(; ncomponents=FVE(0.5))).selection.cap ==
        [ncomponents(StatsAPI.fit(FPCA, x; method=:face, ncomponents=FVE(0.5)))]
    @test StatsAPI.fit(FPCRegression, x, y; selection=AIC(), fpca=supplied).selection.cap == [3]
    # The dense route keeps far more components; its own FVE(0.99) count caps it.
    dense_bic = StatsAPI.fit(FPCRegression, x, y; selection=BIC(), fpca_options=(; method=:dense))
    @test dense_bic.selection.cap == [ncomponents(StatsAPI.fit(FPCA, x; method=:dense))]
    @test only(dense_bic.selection.cap) > own
    @test_throws r"capped by each FPCA's own selection" StatsAPI.fit(FPCRegression, x, y;
        selection=BIC(), ncomponents=2)
    # LOOCV folds refit with the full-data route (FACE here), not the FPCA default.
    small = (; x=x[1:12], y=y[1:12])
    loocv = StatsAPI.fit(FPCRegression, small.x, small.y; selection=:loocv,
        fpca_options=(; ncomponents=2))
    explicit = StatsAPI.fit(FPCRegression, small.x, small.y; selection=:loocv,
        fpca_options=(; method=:face, ncomponents=2))
    @test loocv.selection.values == explicit.selection.values
    @test loocv.selection.values != StatsAPI.fit(FPCRegression, small.x, small.y;
        selection=:loocv, fpca_options=(; method=:dense, ncomponents=2)).selection.values
end

@testset "Functional-response FPC regression uses the FACE covariate default" begin
    (; x, y, grid) = regression_noisy_dense()
    response = GridFunctionalData(y .* transpose(sqrt(2) .* sinpi.(grid)) .+
        0.5 .* sin.(3.0 .* (1:length(y))) .* transpose(sqrt(2) .* cospi.(grid)), grid)
    f = @inferred StatsAPI.fit(FPCRegression, x, response)
    @test f.settings.fpca_method == (:face,)
    @test f.settings.fpca_method_source == (:regression_default,)
    @test f.selection.cap == f.counts
    # The response FPCA defaults to FACE on complete grids as well.
    @test f.response_fpca.method === :face
    @test f.settings.response_fpca_method === :face
    @test f.settings.response_fpca_method_source === :regression_default
    @test f.settings.response_fpca_knots == [35]
    explicit = StatsAPI.fit(FPCRegression, x, response; fpca_options=(; method=:face))
    @test StatsAPI.coef(f) == StatsAPI.coef(explicit)
    dense = StatsAPI.fit(FPCRegression, x, response; fpca_options=(; method=:dense))
    @test dense.settings.fpca_method == (:dense,)
    @test StatsAPI.coef(dense) == StatsAPI.coef(StatsAPI.fit(FPCRegression, x, response;
        fpca=StatsAPI.fit(FPCA, x; method=:dense)))
end
