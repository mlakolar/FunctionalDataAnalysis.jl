@testset "Caller-owned evaluation and reconstruction buffers" begin
    @test FunctionalDataAnalysis.reconstruct! === StatsAPI.reconstruct!
    for T in (Float32,Float64)
        grid = T[0,.03,.1,.2,.4,.65,.8,.9,1]
        values = T[sin(.5i)*(1+t)+cos(.7i)*t^2 for i in 1:8,t in grid]
        fd = GridFunctionalData(values,grid;ids=collect(1:8))
        irregular = to_irregular(fd)
        masked_values = Matrix{Union{Missing,T}}(values)
        masked_values[1,1] = missing
        masked = GridFunctionalData(masked_values,grid)
        points = T[1,.1,.65,.1,0]
        for data in (fd,irregular,masked), e in (PiecewiseLinear(),NoInterpolation())
            if data === masked && e isa NoInterpolation
                @test_throws ArgumentError evaluate!(zeros(T,8,5),data,points,e)
                continue
            end
            expected = evaluate(data,points,e)
            out = fill(T(99),size(expected))
            snapshot = deepcopy(data)
            @test (@inferred evaluate!(out,data,points,e)) === out
            @test isequal(out,expected)
            @test all(isequal(getfield(data,name),getfield(snapshot,name))
                for name in fieldnames(typeof(data)))
            @test evaluate!(out,data,points,e) === out
            @test isequal(out,expected)
        end
        @test_throws r"requires PiecewiseLinear" evaluate!(zeros(T,8,5),fd,points)
        @test_throws DimensionMismatch evaluate!(zeros(T,7,5),fd,points,PiecewiseLinear())
        @test_throws r"element type" evaluate!(zeros(Int,8,5),fd,points,PiecewiseLinear())
        @test_throws r"must not alias" evaluate!(fd.values,fd,grid,PiecewiseLinear())
        @test_throws r"must not alias" evaluate!(reshape(points,1,:),fd[1:1],points,PiecewiseLinear())
        @test_throws r"finite" evaluate!(zeros(T,8,1),fd,T[NaN],PiecewiseLinear())
        @test evaluate!(zeros(T,8,0),fd,T[],PiecewiseLinear()) == evaluate(fd,T[],PiecewiseLinear())

        for basis in (MonomialBasis((zero(T),one(T)),3),
                FourierBasis((zero(T),one(T)),5),BSplineBasis(4,(zero(T),one(T)),5))
            coefficients = T[sin(i+j) for i in 1:8,j in 1:nbasis(basis)]
            data = BasisFunctionalData(coefficients,basis)
            for order in 0:2, target in (points,Float64.(points))
                expected = evaluate(data,target;derivative=order)
                out = similar(expected)
                @test (@inferred evaluate!(out,data,target;derivative=order)) === out
                @test out == expected
                parent = fill(eltype(out)(99),2size(out,1),2size(out,2))
                strided = view(parent,1:2:size(parent,1),1:2:size(parent,2))
                @test evaluate!(strided,data,target;derivative=order) === strided
                @test strided == expected
                @test all(==(99),parent[2:2:end,:])
                columns = view(parent,1:size(out,1),1:2:size(parent,2))
                @test evaluate!(columns,data,target;derivative=order) === columns
                @test columns == expected
            end
            @test_throws r"must not alias" evaluate!(data.coefs,data,
                collect(range(zero(T),one(T);length=nbasis(basis))))
            @test_throws r"element type" evaluate!(zeros(Int,8,5),data,points)
        end

        basis_data = BasisFunctionalData(T[sin(i+j) for i in 1:8,j in 1:3],
            MonomialBasis((zero(T),one(T)),2))
        multi = MultivariateFunctionalData([fd,basis_data])
        common = evaluate(multi,points,PiecewiseLinear())
        out = similar(common)
        @test (@inferred evaluate!(out,multi,points,PiecewiseLinear())) === out
        @test out == common
        @test_throws r"raw multivariate" evaluate!(out,multi,points)
        @test_throws r"inside each component" evaluate!(out,multi,fill(T(2),5),PiecewiseLinear())
        separate_points = [points,T[.2,.4,1]]
        separate = evaluate(multi,separate_points,PiecewiseLinear())
        buffers = similar.(separate)
        @test evaluate!(buffers,multi,separate_points,PiecewiseLinear()) === buffers
        @test buffers == separate
        @test_throws DimensionMismatch evaluate!(buffers,multi,[points],PiecewiseLinear())
        @test_throws DimensionMismatch evaluate!([buffers[1]],multi,separate_points,PiecewiseLinear())
        @test_throws r"must be matrices" evaluate!(Any[buffers[1],1],multi,separate_points,PiecewiseLinear())
        @test_throws r"must not alias each other" evaluate!([buffers[1],buffers[1]],multi,[points,points],PiecewiseLinear())
        @test_throws r"element type" evaluate!([zeros(Int,8,5),zeros(Int,8,3)],multi,separate_points,PiecewiseLinear())
        all_basis = MultivariateFunctionalData([basis_data,basis_data])
        all_basis_out = similar(evaluate(all_basis,points))
        @test evaluate!(all_basis_out,all_basis,points) == evaluate(all_basis,points)
        basis_buffers = similar.(evaluate(all_basis,separate_points))
        @test evaluate!(basis_buffers,all_basis,separate_points) == evaluate(all_basis,separate_points)

        fitted = StatsAPI.fit(FPCA,fd;ncomponents=2,max_k=3)
        basis_fit = StatsAPI.fit(FPCA,basis_data;ncomponents=2)
        d = derivative(fitted;bandwidth=2.)
        for model in (fitted,basis_fit,d), target in (model.grid,Float64.(model.grid))
            expected = reconstruct(model;grid=target).values
            restored = similar(expected)
            snapshot = deepcopy(model)
            @test (@inferred reconstruct!(restored,model;grid=target)) === restored
            @test restored == expected
            @test scores(model) == scores(snapshot)
            @test model.settings == snapshot.settings
            @test reconstruct!(restored,model,view(scores(model),:,:);grid=target) == expected
            parent = zeros(eltype(expected),2size(expected,1),2size(expected,2))
            strided = view(parent,1:2:size(parent,1),1:2:size(parent,2))
            @test reconstruct!(strided,model;grid=target) === strided
            @test strided == expected
            @test_throws DimensionMismatch reconstruct!(zeros(eltype(expected),1,1),model;grid=target)
            @test_throws r"element type" reconstruct!(zeros(Int,size(expected)),model;grid=target)
            @test_throws DimensionMismatch reconstruct!(restored,model,zeros(T,8,3);grid=target)
            @test_throws ArgumentError reconstruct!(restored,model,fill(T(NaN),8,2);grid=target)
            @test_throws ArgumentError reconstruct!(restored,model;grid=T[0])
        end
        # Protect even stored arrays that the reconstruction does not read.
        @test_throws r"must not alias" reconstruct!(fitted.covariance,fitted,
            zeros(T,length(grid),2))
        @test_throws r"must not alias" reconstruct!(reshape(fitted.grid,1,:),fitted,
            zeros(T,1,2))

        fit_multi = MultivariateFunctionalData([fd,to_grid(basis_data,grid)])
        marginal = StatsAPI.fit(FPCA,fit_multi;ncomponents=2,max_k=3)
        joint = StatsAPI.fit(MFPCA,fit_multi;ncomponents=2)
        for model in (marginal,joint)
            expected = reconstruct(model)
            buffers = [similar(component(expected,j).values) for j in 1:2]
            @test (@inferred reconstruct!(buffers,model)) === buffers
            @test all(buffers[j] == component(expected,j).values for j in 1:2)
            @test reconstruct!(buffers,model) === buffers
            @test_throws DimensionMismatch reconstruct!([buffers[1]],model)
            @test_throws r"must not alias each other" reconstruct!([buffers[1],buffers[1]],model)
        end
        expected = reconstruct(marginal)
        buffers = [similar(component(expected,j).values) for j in 1:2]
        @test_throws DimensionMismatch reconstruct!(buffers,marginal,[scores(marginal)[1]])
        @test_throws r"must be real matrices" reconstruct!(buffers,marginal,Any[1,2])
        @test_throws DimensionMismatch reconstruct!(buffers,marginal,[zeros(T,2,2),zeros(T,3,2)])
        @test_throws DimensionMismatch reconstruct!(buffers,marginal,[zeros(T,8,3),zeros(T,8,2)])
        @test_throws ArgumentError reconstruct!(buffers,joint,zeros(T,0,2))
        @test_throws DimensionMismatch reconstruct!(buffers,joint,zeros(T,8,3))
        mixed_scores = [scores(marginal)[1],Float64.(scores(marginal)[2])]
        mixed_expected = reconstruct(marginal,mixed_scores)
        mixed_buffers = [similar(component(mixed_expected,j).values) for j in 1:2]
        @test reconstruct!(mixed_buffers,marginal,mixed_scores) === mixed_buffers
        @test all(mixed_buffers[j] == component(mixed_expected,j).values for j in 1:2)
        integral_scores = round.(Int,scores(joint))
        integral_expected = reconstruct(joint,integral_scores)
        integral_buffers = [similar(component(integral_expected,j).values) for j in 1:2]
        @test reconstruct!(integral_buffers,joint,integral_scores) === integral_buffers
        @test all(integral_buffers[j] == component(integral_expected,j).values for j in 1:2)

        derivative_nodes = derivative(marginal;bandwidth=2.)
        derivative_expected = reconstruct(derivative_nodes)
        derivative_buffers = [similar(component(derivative_expected,j).values) for j in 1:2]
        @test reconstruct!(derivative_buffers,derivative_nodes) === derivative_buffers
        @test all(derivative_buffers[j] == component(derivative_expected,j).values for j in 1:2)
    end

    # The cycle and nested array exercise storage checks without traversing the
    # elements of large numeric arrays or recursing forever through metadata.
    cache = Dict{Symbol,Any}(:out=>zeros(2,3))
    cache[:self] = cache
    data = GridFunctionalData(ones(2,3),[0.,.5,1.];
        metadata=(; cache=Any[cache], callback=sin, module_type=Float64))
    @test_throws r"must not alias" evaluate!(data.metadata.cache[1][:out],data,
        data.grid,PiecewiseLinear())
    constant = StatsAPI.fit(FPCA,GridFunctionalData([-1. -1.;0. 0.;1. 1.],[0.,1.]);
        ncomponents=1,max_k=1)
    constant.mean.values .= floatmax(Float64)
    huge_scores = fill(floatmax(Float64),3,1)
    @test_throws ArgumentError reconstruct(constant,huge_scores)
    @test_throws r"reconstructed values must be finite" reconstruct!(zeros(3,2),constant,huge_scores)
    sparse = golden_sparse3(golden_read("m3_sparse_input.csv"))
    face = StatsAPI.fit(FPCA,sparse;method=:face,face=(;lambda=2.5,lambda_mean=.75))
    factor = face.covariance.factor
    factor_grid = collect(range(first(face.grid),last(face.grid);length=size(factor,2)))
    snapshot = copy(factor)
    @test_throws r"must not alias" reconstruct!(factor,face,
        zeros(size(factor,1),ncomponents(face));grid=factor_grid)
    @test factor == snapshot
end
