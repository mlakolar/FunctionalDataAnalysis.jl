@testset "Tables and subject IDs" begin
    table = (subject=[20, 10, 20, 10, 20], t=[1.0, 0.5, 0.0, 1.0, 0.5], y=[3.0, 5.0, 1.0, 6.0, 2.0])
    ir = IrregularFunctionalData(table; id=:subject, time=:t, value=:y)
    @test ids(ir) == [20, 10]
    @test ir[1] == ([0.0, 0.5, 1.0], [1.0, 2.0, 3.0])
    @test ir[2] == ([0.5, 1.0], [5.0, 6.0])
    @test IrregularFunctionalData(Tables.rowtable(table); id=:subject, time=:t, value=:y).values == ir.values
    g = GridFunctionalData(table; id=:subject, time=:t, value=:y)
    @test ids(g) == [20, 10]
    @test g.grid == [0.0, 0.5, 1.0]
    @test g.observed == Bool[1 1 1; 0 1 1]
    @test g.values == [1 2 3; 0 5 6]
    @test GridFunctionalData(Tables.rowtable(table); id=:subject, time=:t, value=:y).values == g.values
    for fd in (ir, g)
        @test Tables.istable(typeof(fd))
        @test Tables.columnaccess(typeof(fd))
        columns = Tables.columntable(fd)
        @test keys(columns) == (:id, :time, :value)
        expectedids = fd isa GridFunctionalData ? [20, 20, 20, 10, 10, 10] : [20, 20, 20, 10, 10]
        @test columns.id == expectedids
        @test Tables.schema(fd).names == (:id, :time, :value)
        @test length(Tables.rowtable(fd)) == length(columns.time)
        copyfd = fd isa GridFunctionalData ? GridFunctionalData(columns; id=:id, time=:time, value=:value) :
            IrregularFunctionalData(columns; id=:id, time=:time, value=:value)
        @test copyfd.values == fd.values
        @test ids(copyfd) == ids(fd)
        columns.time[1] = -10
        @test domain(fd) == (0.0, 1.0)
    end
    @test Tables.columntable(ir).id == [20, 20, 20, 10, 10]
    roundtrip = GridFunctionalData(Tables.columntable(g); id=:id, time=:time, value=:value)
    @test roundtrip.observed == g.observed
    @test roundtrip.grid == g.grid
    @test roundtrip.values == g.values

    f32 = (id=[2, 1, 2, 1], time=Float32[0, 0, 1, 1], value=Float32[1, 2, 3, 4])
    gf = GridFunctionalData(f32; id=:id, time=:time, value=:value)
    iff = IrregularFunctionalData(f32; id=:id, time=:time, value=:value)
    @test gf isa GridFunctionalData{Float32}
    @test iff isa IrregularFunctionalData{Float32}
    @test (@inferred evaluate(gf, Float32[0.5], PiecewiseLinear())) isa Matrix{Float32}
    @test (@inferred evaluate(iff, Float32[0], NoInterpolation())) isa Matrix{Float32}
    @test (@inferred gf[1:1]) isa GridFunctionalData{Float32}
    @test (@inferred iff[Bool[1, 0]]) isa IrregularFunctionalData{Float32}
    @test ids(gf) == [2, 1]
    @test GridFunctionalData((id=[1, 1], time=[0, 1], value=[2, 3]); id=:id, time=:time, value=:value) isa GridFunctionalData{Float64}

    @testset "Nullable and signed zero times" begin
        for T in (Float32, Float64)
            nullable = (id=[2, 1, 2, 1], time=Union{Missing,T}[1, 0, 0, 1], value=T[3, 2, 1, 4])
            griddata = GridFunctionalData(nullable; id=:id, time=:time, value=:value)
            irregular = IrregularFunctionalData(nullable; id=:id, time=:time, value=:value)
            @test griddata isa GridFunctionalData{T}
            @test irregular isa IrregularFunctionalData{T}
            @test griddata.grid == T[0, 1]
            @test griddata.values == T[1 3; 2 4]
            @test irregular.times == T[0, 1, 0, 1]
            @test irregular.values == T[1, 3, 2, 4]
            @test ids(griddata) == ids(irregular) == [2, 1]

            signedzero = (id=[2, 2, 1, 1], time=T[-0.0, 1, 0, 1], value=T[1, 2, 3, 4])
            normalized = GridFunctionalData(signedzero; id=:id, time=:time, value=:value)
            @test normalized isa GridFunctionalData{T}
            @test isequal(normalized.grid, T[0, 1])
            @test normalized.values == T[1 2; 3 4]
            @test normalized.observed === nothing
            @test ids(normalized) == [2, 1]
            @test signbit(signedzero.time[1])
            duplicatezero = (id=[1, 1, 1], time=T[-0.0, 0, 1], value=T[1, 2, 3])
            @test_throws ArgumentError("table must have unique (id, time) pairs") GridFunctionalData(duplicatezero; id=:id, time=:time, value=:value)
        end
        for times in (Union{Missing,Float64}[0.0, missing], [missing, missing], ["0", "1"])
            invalid = (id=[1, 1], time=times, value=[1.0, 2.0])
            @test_throws ArgumentError GridFunctionalData(invalid; id=:id, time=:time, value=:value)
            @test_throws ArgumentError IrregularFunctionalData(invalid; id=:id, time=:time, value=:value)
        end
    end

    labels = [20, 10]
    raw = @inferred GridFunctionalData([1 2 3; 4 5 6], [0.0, 0.5, 1.0]; ids=labels)
    labels[1] = -1
    @test ids(raw) == [20, 10]
    copied = ids(raw)
    copied[1] = -1
    @test ids(raw) == [20, 10]
    irregular = @inferred to_irregular(raw)
    basis = @inferred to_basis(raw, MonomialBasis((0, 1), 1))
    @test ids(irregular) == ids(basis) == [20, 10]
    @test ids(to_grid(irregular, raw.grid, NoInterpolation())) == [20, 10]
    @test ids(to_grid(irregular, raw.grid, PiecewiseLinear())) == [20, 10]
    @test ids(to_grid(basis, raw.grid)) == [20, 10]
    @test ids(to_irregular(basis; grid=raw.grid)) == [20, 10]
    @test ids(to_irregular(irregular)) == [20, 10]
    @test ids(basis[1]) == [20]
    @test ids(to_grid(raw, raw.grid, NoInterpolation())) == [20, 10]
    for fd in (raw, irregular, basis)
        @test ids(@inferred fd[1:1]) == [20]
        @test ids(@inferred fd[[2, 1]]) == [10, 20]
        @test ids(@inferred fd[Bool[0, 1]]) == [10]
        @test ids(vcat(fd[1:1], fd[2:2])) == [20, 10]
        @test_throws ArgumentError fd[[1, 1]]
        @test_throws ArgumentError vcat(fd, fd)
    end
    @test Tables.columntable(basis; grid=raw.grid).id == [20, 20, 20, 10, 10, 10]
    @test_throws ArgumentError Tables.columntable(basis)
    @test_throws ArgumentError vcat(raw, GridFunctionalData([1 2 3], raw.grid))
    @test_throws ArgumentError vcat(irregular, IrregularFunctionalData([[0.0, 1.0]], [[1, 2]]))
    @test_throws ArgumentError vcat(basis, BasisFunctionalData([1 2], basis.basis))
    @test_throws DimensionMismatch GridFunctionalData([1 2], [0, 1]; ids=[1, 2])
    @test_throws DimensionMismatch IrregularFunctionalData([[0, 1]], [[1, 2]]; ids=[1, 2])
    @test_throws DimensionMismatch BasisFunctionalData([1 2], basis.basis; ids=[1, 2])
    @test_throws ArgumentError GridFunctionalData([1 2; 3 4], [0, 1]; ids=[1, 1])
    @test_throws ArgumentError IrregularFunctionalData([[0, 1], [0, 1]], [[1, 2], [3, 4]]; ids=[1, 1])
    @test_throws ArgumentError BasisFunctionalData([1 2; 3 4], basis.basis; ids=[1, 1])
    @test_throws ArgumentError GridFunctionalData([1 2], [0, 1]; ids=[missing])
    @test ids(MultivariateFunctionalData([raw, basis])) == [20, 10]
    @test_throws ArgumentError MultivariateFunctionalData([raw, basis[[2, 1]]])
    mixed = MultivariateFunctionalData([raw, BasisFunctionalData(basis.coefs, basis.basis)])
    other = MultivariateFunctionalData([GridFunctionalData(raw.values, raw.grid),
        BasisFunctionalData(basis.coefs, basis.basis; ids=[30, 40])])
    @test ids(vcat(mixed, other)) == [20, 10, 30, 40]
    @test ids(mixed[1:1]) == [20]
    for fd in (GridFunctionalData([1 2], [0, 1]), IrregularFunctionalData([[0, 1]], [[1, 2]]),
            BasisFunctionalData([1 2], basis.basis))
        @test ids(fd) === nothing
    end
    unlabeled = GridFunctionalData([1 2], [0, 1])
    @test Tables.columntable(unlabeled).id == [1, 1]
    @test ids(unlabeled) === nothing
    @test ids(MultivariateFunctionalData([unlabeled])) === nothing
    @test_throws ArgumentError vcat(mixed, MultivariateFunctionalData([GridFunctionalData(raw.values, raw.grid), BasisFunctionalData(basis.coefs, basis.basis)]))

    duplicate = (id=[1, 1], time=[0.0, 0.0], value=[1.0, 3.0])
    @test_throws ArgumentError GridFunctionalData(duplicate; id=:id, time=:time, value=:value)
    @test_throws ArgumentError IrregularFunctionalData(duplicate; id=:id, time=:time, value=:value)
    @test IrregularFunctionalData(duplicate; id=:id, time=:time, value=:value, ties=:mean).values == [2.0]
    missingvalues = (id=[1, 1], time=[0.0, 1.0], value=[missing, 2.0])
    @test GridFunctionalData(missingvalues; id=:id, time=:time, value=:value).observed == Bool[0 1]
    @test_throws ArgumentError IrregularFunctionalData(missingvalues; id=:id, time=:time, value=:value)
    for table in ((id=Int[], time=Float64[], value=Float64[]),
            (id=[1, 1], time=[0.0, Inf], value=[1.0, 2.0]),
            (id=[1, 1], time=[0.0, 1.0], value=[1.0, NaN]),
            (id=[1, 1], time=[0.0, 1.0], value=[1.0, Inf]))
        @test_throws ArgumentError GridFunctionalData(table; id=:id, time=:time, value=:value)
        @test_throws ArgumentError IrregularFunctionalData(table; id=:id, time=:time, value=:value)
    end
    @test_throws ArgumentError GridFunctionalData((id=[1], time=[0.0], value=[1.0]); id=:id, time=:time, value=:value)
    @test_throws ArgumentError GridFunctionalData(table; id=:unknown, time=:t, value=:y)
    @test_throws ArgumentError GridFunctionalData(1; id=:id, time=:time, value=:value)
    @test_throws DimensionMismatch GridFunctionalData((id=[1, 1], time=[0.0], value=[1.0]); id=:id, time=:time, value=:value)
end
