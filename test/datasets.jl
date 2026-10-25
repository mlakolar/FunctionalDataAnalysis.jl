# The committed reference.csv is generated in R from the native data objects,
# before loading their CSV exports in Julia. CI needs neither R nor a network.
function dataset_reference(root)
    lines = filter(l -> !startswith(l, "#"), readlines(joinpath(root, "reference.csv")))
    [split(replace(l, '"' => ""), ',') for l in lines[2:end]]
end

function check_dataset_references(root; sample)
    loaded = Dict(row.name => load_dataset(row.name; sample) for row in datasets())
    for row in dataset_reference(root)
        name, node = Symbol(row[1]), Symbol(row[2])
        result = loaded[name]
        names = getproperty.(result.source.components, :name)
        j = only(findall(==(node), names))
        fd = result.data isa MultivariateFunctionalData ? component(result.data, j) : result.data
        n, m, observed, absent = parse.(Int, row[3:6])
        total, squares, grid_sum, first_value = parse.(Float64, row[7:10])
        @test nobs(fd) == n
        @test ids(fd) == result.covariates.id
        if fd isa GridFunctionalData
            @test size(fd.values) == (n, m)
            @test sum(nobs_per_curve(fd)) == observed
            @test (fd.observed === nothing ? 0 : count(!, fd.observed)) == absent
            @test sum(fd.grid) ≈ grid_sum rtol=1e-12
            values = fd.observed === nothing ? vec(fd.values) : fd.values[fd.observed]
            @test first(fd[1][2]) == first_value
        else
            @test length(fd.values) == observed
            @test observed + absent == n * m
            @test domain(fd) == (-18.0, 42.0)
            values = fd.values
            @test first(values) == first_value
        end
        @test sum(values) ≈ total rtol=1e-12
        @test sum(abs2, values) ≈ squares rtol=1e-12
    end
    loaded
end

@testset "Classic datasets" begin
    sample_root = joinpath(@__DIR__, "..", "data", "samples")
    @test getproperty.(@inferred(datasets()), :name) ==
        [:growth, :canadian_weather, :gait, :tecator, :cd4, :dti]
    @test all(r -> !isempty(r.description), datasets())
    @test_throws ArgumentError load_dataset(:unknown)
    @test_throws ArgumentError load_dataset("growth"; sample=true)
    loaded = check_dataset_references(sample_root; sample=true)
    for result in values(loaded)
        @test propertynames(result) == (:data, :covariates, :description, :source, :license)
        @test Tables.istable(result.covariates)
        @test Tables.columnaccess(result.covariates)
        @test length(result.covariates.id) == nobs(result.data)
        @test allunique(result.covariates.id)
        @test eltype(result.covariates.id) == String
        @test result.source.sample
        @test length(result.source.rows) == nobs(result.data)
        @test !isempty(result.source.citation)
        @test !isempty(result.license)
        @test occursin("sample = true", sprint(show, result.source))
    end

    @testset "Source grids, values, covariates and ownership" begin
        growth = loaded[:growth]
        @test growth.data isa GridFunctionalData{Float64}
        @test growth.covariates.id == ["boy01", "boy02", "girl01", "girl02"]
        @test growth.covariates.sex == ["male", "male", "female", "female"]
        @test growth.data.grid[1:6] == [1, 1.25, 1.5, 1.75, 2, 3]
        @test growth.data.weights[1:6] == [0.125, 0.25, 0.25, 0.25, 0.625, 1.0]
        @test growth.data.values[1, 1] == 81.3
        @test growth.data.values[3, 1] == 76.2
        @test growth.source.version == "6.3.0"

        weather = loaded[:canadian_weather]
        @test weather.data isa MultivariateFunctionalData{Float64}
        @test getproperty.(weather.source.components, :name) == [:temperature, :precipitation]
        @test weather.covariates.id == ["St. Johns", "Halifax", "Sydney"]
        @test weather.covariates.latitude == [47.34, 44.39, 46.09]
        @test weather.covariates.region == fill("Atlantic", 3)
        @test component(weather.data, 1).grid == collect(1.0:365.0)
        @test component(weather.data, 2).values[1, 1] == 5.2
        @test to_array(weather.data)[1, :, 1] == [-3.6, 5.2]

        gait = loaded[:gait]
        @test gait.data isa MultivariateFunctionalData{Float64}
        @test gait.covariates.id == ["boy1", "boy2", "boy3"]
        @test getproperty.(gait.source.components, :name) == [:hip, :knee]
        @test component(gait.data, 1).grid ≈ collect(0.025:0.05:0.975) rtol=1e-14

        tecator = loaded[:tecator]
        @test tecator.data isa GridFunctionalData{Float64}
        @test tecator.covariates.Fat == [22.5, 40.1, 8.4]
        @test tecator.covariates.Water == [60.5, 46.0, 71.0]
        @test tecator.covariates.Protein == [16.7, 13.5, 20.5]
        @test extrema(tecator.data.grid) == (850.0, 1050.0)
        @test tecator.source.full_nobs == 215
        @test occursin("TECATOR.txt", tecator.license)

        cd4 = loaded[:cd4]
        @test cd4.data isa IrregularFunctionalData{Float64}
        @test cd4.data[1] == ([-9.0, -3.0, 3.0], [548.0, 893.0, 657.0])
        @test nobs_per_curve(cd4.data) == [3, 4, 8]
        @test cd4.source.full_nobs == 366
        @test cd4.source.version == "0.1-40"

        # Every load is independent of earlier loads and covariate mutations.
        growth.data.values[1, 1] = -1
        growth.data.grid[1] = -1
        growth.covariates.sex[1] = "changed"
        growth.covariates.id[1] = "changed"
        fresh = load_dataset(:growth; sample=true)
        @test fresh.data.values[1, 1] == 81.3
        @test first(fresh.data.grid) == 1.0
        @test fresh.covariates.sex[1] == "male"
        @test ids(fresh.data)[1] == fresh.covariates.id[1] == "boy01"
    end

    @testset "DTI retains both tracts, visits and masks" begin
        d = loaded[:dti]
        @test d.data isa MultivariateFunctionalData{Float64}
        @test ncomponents(d.data) == 2
        cca, rcst = component(d.data, 1), component(d.data, 2)
        @test cca isa GridFunctionalData{Float64}
        @test rcst isa GridFunctionalData{Float64}
        @test (cca.metadata.name, rcst.metadata.name) == (:cca, :rcst)
        @test cca.grid == collect(1.0:93.0)
        @test rcst.grid == collect(1.0:55.0)
        @test size(cca.values) == (5, 93)
        @test size(rcst.values) == (5, 55)
        @test count(!, cca.observed) == 2
        @test count(!, rcst.observed) == 26
        @test all(iszero, cca.values[.!cca.observed])
        @test all(iszero, rcst.values[.!rcst.observed])
        @test !rcst.observed[2, 1] # source NaN
        @test !cca.observed[5, 67] && !cca.observed[5, 68]
        @test ids(cca) == ids(rcst) == d.covariates.id ==
            ["1001_1", "1002_1", "2001_1", "2001_2", "2017_1"]
        @test propertynames(d.covariates) ==
            (:id, :ID, :visit, Symbol("visit.time"), :Nscans, :case, :sex, :pasat)
        @test d.covariates.ID == [1001.0, 1002.0, 2001.0, 2001.0, 2017.0]
        @test d.covariates.visit == [1, 1, 1, 2, 1]
        @test d.covariates.var"visit.time" == [0, 0, 0, 90, 0]
        @test d.covariates.case == [0.0, 0.0, 1.0, 1.0, 1.0]
        @test isequal(d.covariates.pasat, [missing, missing, 31, 31, 34])
        @test d.covariates.sex == ["female", "female", "female", "female", "male"]
        @test d.source.rows == [1, 2, 43, 44, 125]
        @test d.source.full_nobs == 382
        @test occursin("Kennedy-Krieger", d.source.notes)
        @test_throws ArgumentError to_array(d.data) # different tract grids and missing data
        @test ids(d.data[3:4]) == ["2001_1", "2001_2"]
        d.covariates.id[1] = "changed"
        @test ids(d.data)[1] == "1001_1"
        cca.values[1, 1] = -1
        @test component(load_dataset(:dti; sample=true).data, 1).values[1, 1] == 0.49093448369395709
    end

    @testset "Malformed exports fail explicitly" begin
        function malformed(name, file, before, after, exception)
            mktempdir() do root
                cp(joinpath(sample_root, String(name)), joinpath(root, String(name)))
                path = joinpath(root, String(name), file)
                content = read(path, String)
                @test occursin(before, content)
                write(path, replace(content, before => after; count=1))
                @test_throws exception FunctionalDataAnalysis._load_dataset(name, root)
            end
        end
        malformed(:growth, "height_values.csv", "81.299999999999997", "Inf", ArgumentError)
        malformed(:growth, "height_grid.csv", "\"1.25\"", "\"1\"", ArgumentError)
        malformed(:growth, "height_values.csv", "\"boy01\"", "\"wrong\"", ArgumentError)
        malformed(:growth, "height_values.csv", "\"id\",\"v1\"", "\"id\"", DimensionMismatch)
        malformed(:growth, "covariates.csv", "\"boy02\"", "\"boy01\"", ArgumentError)
        malformed(:growth, "metadata.csv", "\"nobs\",\"4\"", "\"nobs\",\"5\"", DimensionMismatch)
        malformed(:growth, "metadata.csv", "\"name\",\"growth\"", "\"name\",\"gait\"", ArgumentError)
        malformed(:growth, "metadata.csv", "\"format_version\",\"1\"", "\"format_version\",\"2\"", ArgumentError)
        malformed(:dti, "components.csv", "\"cca\"", "\"rcst\"", ArgumentError)
        malformed(:dti, "covariates.csv", "\"1001_1\",\"1001\"", "\"1001_1\",\"NaN\"", ArgumentError)
        malformed(:dti, "schema.csv", "\"float\"", "\"unknown\"", ArgumentError)
        malformed(:cd4, "count_observations.csv", "\"548\"", "NA", ArgumentError)
        malformed(:cd4, "count_observations.csv", "\"1\",\"-9\"", "\"absent\",\"-9\"", ArgumentError)
        malformed(:gait, "covariates.csv", "\"boy1\"", "\"boy1", ArgumentError)
        mktempdir() do root
            @test_throws ArgumentError FunctionalDataAnalysis._load_dataset(:growth, root)
        end
        # Quoted commas, escaped quotes and UTF-8 are supported without a CSV dependency.
        mktempdir() do dir
            path = joinpath(dir, "quoted.csv")
            write(path, "# header\na,b,c\n\"Montréal, QC\",\"say \"\"yes\"\"\",\n")
            header, rows = FunctionalDataAnalysis._dataset_csv(path)
            @test header == ["a", "b", "c"]
            @test rows == [["Montréal, QC", "say \"yes\"", ""]]
            write(path, "a,b\n1\n")
            @test_throws DimensionMismatch FunctionalDataAnalysis._dataset_csv(path)
        end
    end

    @testset "Samples without an artifact depot" begin
        saved_depots = copy(DEPOT_PATH)
        mktempdir() do depot
            try
                empty!(DEPOT_PATH)
                push!(DEPOT_PATH, depot)
                @test length(datasets()) == 6
                for row in datasets()
                    @test nobs(load_dataset(row.name; sample=true).data) > 0
                end
                @test !isdir(joinpath(depot, "artifacts"))
            finally
                empty!(DEPOT_PATH)
                append!(DEPOT_PATH, saved_depots)
            end
        end
    end

    if get(ENV, "FDA_TEST_ARTIFACTS", "0") == "1"
        @testset "Full lazy artifact (opt in)" begin
            root = FunctionalDataAnalysis._dataset_artifact()
            full = check_dataset_references(root; sample=false)
            @test [nobs(full[r.name].data) for r in datasets()] == [93, 35, 39, 215, 366, 382]
            @test all(d -> !d.source.sample, values(full))
            dti = full[:dti]
            @test length(unique(dti.covariates.ID)) == 142
            @test count(!, component(dti.data, 1).observed) == 36
            @test count(!, component(dti.data, 2).observed) == 738
            @test length(full[:cd4].data.values) == 1888
        end
    end
end
