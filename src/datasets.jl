const _DATASET_DESCRIPTIONS = (
    growth = "Berkeley heights at 31 uneven ages, with sex (93 curves).",
    canadian_weather = "Daily Canadian temperature and precipitation, with station covariates (35 stations).",
    gait = "Hip and knee angles at 20 gait times (39 boys).",
    tecator = "Meat absorbance spectra, with Fat, Water and Protein percentages (215 samples).",
    cd4 = "Sparse CD4 counts at months since seroconversion (366 subjects).",
    dti = "DTI cca (93 points) and rcst (55 points), with subject and visit covariates (382 scans).",
)

const _DATASET_NODES = (
    growth = ("height",), canadian_weather = ("temperature", "precipitation"),
    gait = ("hip", "knee"), tecator = ("absorbance",), cd4 = ("count",),
    dti = ("cca", "rcst"),
)

"""
    datasets()

List the available classic datasets as a vector of `(name, description)` named
tuples. Listing datasets does not access the network or install artifacts.
See [`load_dataset`](@ref) for provenance and offline samples.
"""
datasets() = [(name=name, description=getproperty(_DATASET_DESCRIPTIONS, name))
    for name in keys(_DATASET_DESCRIPTIONS)]

"""
    load_dataset(name::Symbol; sample=false)

Return `(; data, covariates, description, source, license)` for a name listed by
[`datasets`](@ref). `covariates` is a Tables.jl column table: its unique string
`id` column equals `ids(data)` in curve order. Each call reads fresh arrays.
The full data are downloaded through a lazy Julia artifact on first use.
`sample=true` reads a small, committed subset without network access. The
default `false` and the sample selections are package choices; selections and
provenance are recorded in `source` and displayed by the named tuple's `show`.

Sources are R fda 6.3.0 (`:growth`, `:canadian_weather`, `:gait`), fda.usc 2.2.0
(`:tecator`), and refund 0.1-40 (`:cd4`, `:dti`). No smoothing, interpolation,
normalization, or statistical tuning is applied. Growth preserves uneven ages
and sex. Weather nodes are temperature and raw precipitation, in that order;
gait nodes are hip and knee. CD4 is irregular data at observed months only.

`:dti` returns `MultivariateFunctionalData` with nodes **cca then rcst**:
corpus callosum (93 points) and right corticospinal tract (55 points), each a
`GridFunctionalData` on its own point-index grid. `component(data, 1)` selects
cca. Refund's NA/NaN values become unobserved mask entries. Native `ID_visit`
scan labels identify curves; the original repeated subject `ID`, `visit`,
`visit.time`, `Nscans`, `case`, `sex` labels, and `pasat` remain covariates.
There are 382 scans from 142 subjects. In publications acknowledge: "The
MRI/DTI data were collected at Johns Hopkins University and the Kennedy-Krieger
Institute". See the Datasets page and attached source notices for citations
and redistribution terms, including Tecator's permission note.
"""
function load_dataset(name::Symbol; sample::Bool=false)
    _check_dataset_name(name)
    root = sample ? joinpath(@__DIR__, "..", "data", "samples") : _dataset_artifact()
    _load_dataset(name, root)
end

load_dataset(name; kwargs...) = throw(ArgumentError("dataset name must be a Symbol listed by datasets()"))

function _check_dataset_name(name)
    haskey(_DATASET_DESCRIPTIONS, name) || throw(ArgumentError(
        "unknown dataset $name; available names are $(keys(_DATASET_DESCRIPTIONS))"))
end

function _dataset_artifact()
    try
        artifact"fda_datasets"
    catch err
        throw(ErrorException("could not download the FDA datasets artifact from the " *
            "datasets-v1 GitHub release. " *
            "Use load_dataset(name; sample=true) for offline samples. Original error: " *
            sprint(showerror, err)))
    end
end

# Restricted CSV dialect emitted by build_datasets.R: one record per line,
# comma delimiters, double-quoted fields with doubled quote escapes, and #
# provenance comments. Multiline fields are deliberately rejected.
function _dataset_csv_fields(line, path)
    chars = collect(line)
    fields = String[]
    i, n = 1, length(chars)
    while true
        field = IOBuffer()
        if i <= n && chars[i] == '"'
            i += 1
            closed = false
            while i <= n
                c = chars[i]
                i += 1
                if c == '"'
                    if i <= n && chars[i] == '"'
                        write(field, '"')
                        i += 1
                    else
                        closed = true
                        break
                    end
                else
                    write(field, c)
                end
            end
            closed || throw(ArgumentError("unterminated CSV quote in dataset file $path"))
            i <= n && chars[i] != ',' && throw(ArgumentError("invalid CSV quote in dataset file $path"))
        else
            while i <= n && chars[i] != ','
                chars[i] == '"' && throw(ArgumentError("unexpected CSV quote in dataset file $path"))
                write(field, chars[i])
                i += 1
            end
        end
        push!(fields, String(take!(field)))
        i > n && return fields
        i += 1 # delimiter (including a final empty field)
    end
end

function _dataset_csv(path)
    isfile(path) || throw(ArgumentError("dataset file is missing: $path"))
    # rstrip('\r') accepts CRLF checkouts of the sample files.
    lines = filter(line -> !startswith(line, "#"), [String(rstrip(line, '\r')) for line in readlines(path)])
    isempty(lines) && throw(ArgumentError("dataset CSV is empty: $path"))
    header = _dataset_csv_fields(first(lines), path)
    all(!isempty, header) && allunique(header) ||
        throw(ArgumentError("dataset CSV column names must be nonempty and unique: $path"))
    rows = [_dataset_csv_fields(line, path) for line in lines[2:end]]
    all(r -> length(r) == length(header), rows) ||
        throw(DimensionMismatch("dataset CSV row width differs from its header: $path"))
    header, rows
end

function _dataset_rows(path, expected)
    header, rows = _dataset_csv(path)
    header == expected || throw(ArgumentError("dataset CSV columns must be $expected: $path"))
    rows
end

function _dataset_number(::Type{T}, token, context) where {T<:Real}
    value = tryparse(T, token)
    value !== nothing && isfinite(value) ||
        throw(ArgumentError("dataset $context must be a finite $T, got $(repr(token))"))
    value
end

function _dataset_column(tokens, type, name)
    type == "string" && return Union{Missing,String}[t == "NA" ? missing : t for t in tokens]
    T = type == "int" ? Int : type == "float" ? Float64 :
        throw(ArgumentError("unknown dataset covariate type $type for $name"))
    Union{Missing,T}[t == "NA" ? missing : _dataset_number(T, t, "covariate $name") for t in tokens]
end

function _dataset_covariates(dir)
    schema = _dataset_rows(joinpath(dir, "schema.csv"), ["column", "type"])
    names = [r[1] for r in schema]
    !isempty(schema) && first(schema) == ["id", "string"] ||
        throw(ArgumentError("dataset covariate schema must begin with a string id"))
    rows = _dataset_rows(joinpath(dir, "covariates.csv"), names)
    isempty(rows) && throw(ArgumentError("dataset covariates must contain at least one curve"))
    columns = Any[_dataset_column([r[j] for r in rows], schema[j][2], names[j]) for j in eachindex(names)]
    labels = first(columns)
    any(ismissing, labels) && throw(ArgumentError("dataset covariate ids must not be missing"))
    allunique(labels) || throw(ArgumentError("dataset covariate ids must be unique"))
    columns[1] = String[labels...]
    NamedTuple{Tuple(Symbol.(names))}(Tuple(columns))
end

function _dataset_grid(dir, node, labels)
    name, _, unit, grid_unit = node
    gridrows = _dataset_rows(joinpath(dir, name * "_grid.csv"), ["time"])
    grid = [_dataset_number(Float64, r[1], "$name grid") for r in gridrows]
    rows = _dataset_rows(joinpath(dir, name * "_values.csv"),
        vcat("id", ["v$j" for j in eachindex(grid)]))
    [r[1] for r in rows] == labels ||
        throw(ArgumentError("dataset $name curve ids must match covariates in order"))
    values = Matrix{Union{Missing,Float64}}(undef, length(rows), length(grid))
    for i in eachindex(rows), j in eachindex(grid)
        token = rows[i][j + 1]
        values[i, j] = token == "NA" ? missing : _dataset_number(Float64, token, "$name values")
    end
    GridFunctionalData(values, grid; ids=labels,
        metadata=(name=Symbol(name), unit=unit, grid_unit=grid_unit))
end

function _dataset_irregular(dir, node, labels)
    name = first(node)
    rows = _dataset_rows(joinpath(dir, name * "_observations.csv"), ["id", "time", "value"])
    table = (id=[r[1] for r in rows],
        time=[_dataset_number(Float64, r[2], "$name times") for r in rows],
        value=[_dataset_number(Float64, r[3], "$name observations") for r in rows])
    unique(table.id) == labels || throw(ArgumentError(
        "dataset $name observed curve ids must match covariates in order, with no empty curves"))
    bounds = _dataset_rows(joinpath(dir, name * "_domain.csv"), ["time"])
    length(bounds) == 2 || throw(DimensionMismatch("dataset $name domain must have two endpoints"))
    domain = Tuple(_dataset_number(Float64, r[1], "$name domain") for r in bounds)
    IrregularFunctionalData(table; id=:id, time=:time, value=:value, domain)
end

function _load_dataset(name::Symbol, root)
    _check_dataset_name(name)
    dir = joinpath(root, String(name))
    rows = _dataset_rows(joinpath(dir, "metadata.csv"), ["key", "value"])
    allunique(first.(rows)) || throw(ArgumentError("dataset metadata keys must be unique"))
    meta = Dict(r[1] => r[2] for r in rows)
    required = ("format_version", "name", "description", "package", "package_version", "object",
        "url", "source_sha256", "package_license", "license", "citation", "notes", "nobs",
        "full_nobs", "sample", "source_rows", "R_version")
    all(k -> haskey(meta, k), required) || throw(ArgumentError("dataset $name metadata is incomplete"))
    meta["format_version"] == "1" || throw(ArgumentError("unsupported dataset format version"))
    meta["name"] == String(name) || throw(ArgumentError("dataset metadata name does not match $name"))
    meta["sample"] in ("true", "false") || throw(ArgumentError("dataset sample flag must be true or false"))
    covariates = _dataset_covariates(dir)
    _dataset_number(Int, meta["nobs"], "nobs") == length(covariates.id) ||
        throw(DimensionMismatch("dataset $name nobs does not match covariate rows"))
    nodes = _dataset_rows(joinpath(dir, "components.csv"), ["name", "kind", "unit", "grid_unit"])
    Tuple(first.(nodes)) == getproperty(_DATASET_NODES, name) ||
        throw(ArgumentError("dataset $name component names or order differ from the documented nodes"))
    kind = name == :cd4 ? "irregular" : "grid"
    all(r -> r[2] == kind, nodes) || throw(ArgumentError("dataset $name requires $kind components"))
    data = if name == :cd4
        _dataset_irregular(dir, only(nodes), covariates.id)
    elseif length(nodes) == 1
        _dataset_grid(dir, only(nodes), covariates.id)
    else
        MultivariateFunctionalData([_dataset_grid(dir, node, covariates.id) for node in nodes])
    end
    source = (package=meta["package"], version=meta["package_version"], object=meta["object"],
        url=meta["url"], sha256=meta["source_sha256"], package_license=meta["package_license"],
        citation=meta["citation"], notes=meta["notes"], sample=meta["sample"] == "true",
        full_nobs=_dataset_number(Int, meta["full_nobs"], "full_nobs"),
        rows=[_dataset_number(Int, t, "source rows") for t in split(meta["source_rows"], ';')],
        components=[(name=Symbol(r[1]), kind=Symbol(r[2]), unit=r[3], grid_unit=r[4]) for r in nodes],
        R_version=meta["R_version"])
    (; data, covariates, description=meta["description"], source, license=meta["license"])
end
