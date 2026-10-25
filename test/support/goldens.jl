module GoldenTestHelpers

using DelimitedFiles: readdlm
using LinearAlgebra: norm
using FunctionalDataAnalysis: IrregularFunctionalData

export golden_read, golden_relative_error, golden_values4, golden_matrix4,
    golden_values6, golden_matrix6, golden_matrix5, golden_derivative_values6,
    golden_indexed4, golden_observations3, golden_sparse3, golden_sparse4,
    golden_sparse6

golden_read(name, types...; kwargs...) = readdlm(
    joinpath(@__DIR__, "..", "data", name), ',', types...; comments=true, kwargs...)

# Preserve the original norm, denominator and zero-reference behaviour.
golden_relative_error(actual, expected) = norm(actual-expected)/norm(expected)
function golden_relative_error(observed_errors::AbstractVector, actual, expected)
    error = golden_relative_error(actual, expected)
    push!(observed_errors, error)
    error
end

function _golden_matrix(rows, row_column, column_column, value_column)
    reshape(Float64.(rows[:,value_column]), Int(maximum(rows[:,row_column])),
        Int(maximum(rows[:,column_column])))
end

# Four columns: quantity, row (or subject), column (or flattened index), value.
# Values stay in file order. Reshape is always Julia's column-major reshape.
function golden_values4(table, quantity; index=nothing)
    rows = table[:,1] .== quantity
    index === nothing || (rows .&= table[:,2] .== index)
    Float64.(table[rows,4])
end
function golden_matrix4(table, quantity; allow_empty=false)
    rows = table[table[:,1] .== quantity,:]
    allow_empty && isempty(rows) && return zeros(0,0)
    _golden_matrix(rows,2,3,4)
end

# Six columns: mode/basis, quantity, node, row, column, value (MFPCA/FGM).
function _golden_rows6(table, mode, quantity, node)
    table[(table[:,1] .== mode) .& (table[:,2] .== quantity) .&
        (table[:,3] .== node),:]
end
golden_values6(table, mode, quantity, node=0) =
    Float64.(_golden_rows6(table,mode,quantity,node)[:,6])
golden_matrix6(table, mode, quantity, node=0) =
    _golden_matrix(_golden_rows6(table,mode,quantity,node),4,5,6)

# Five columns: quantity, node, row, column, value (MFPCA input matrices).
golden_matrix5(table, quantity, node) = _golden_matrix(
    table[(table[:,1] .== quantity) .& (table[:,2] .== node),:],3,4,5)

# Derivative references have a different six-column schema:
# observation design, derivative method/order label, quantity, row, column, value.
golden_derivative_values6(table, design, method, quantity) = Float64.(table[
    (table[:,1] .== design) .& (table[:,2] .== method) .&
    (table[:,3] .== quantity),6])

# Indexed four-column files explicitly place entries at (row,column). Unlike
# reshape-based references, these permit entries in arbitrary file order.
function golden_indexed4(path)
    raw = readdlm(path, ',', Any; comments=true, comment_char='#')
    names = unique(String.(raw[:,1]))
    Dict(name => begin
        entries = raw[raw[:,1] .== name,:]
        out = zeros(Float64, maximum(Int.(entries[:,2])), maximum(Int.(entries[:,3])))
        for row in eachrow(entries)
            out[Int(row[2]),Int(row[3])] = Float64(row[4])
        end
        out
    end for name in names)
end

# Three columns: subject, time, observation. Preserve subject and row order,
# floating conversion, and caller-supplied labels without inferring new IDs.
function golden_observations3(table; subjects=Int.(unique(table[:,1])), scalar=Float64)
    times = [scalar.(table[table[:,1] .== i,2]) for i in subjects]
    values = [scalar.(table[table[:,1] .== i,3]) for i in subjects]
    (; times, values)
end
function golden_sparse3(table; subjects=Int.(unique(table[:,1])), scalar=Float64, ids=nothing)
    observations = golden_observations3(table; subjects, scalar)
    IrregularFunctionalData(observations.times,observations.values; ids)
end

# Four columns: node, subject, time, observation (cross-node input).
golden_sparse4(table, node; subjects, ids=nothing) =
    golden_sparse3(table[table[:,1] .== node,2:4]; subjects, ids)

# Sparse FGM input uses the six-column mode/quantity/node schema, with subject
# in column four and observation position in column five.
function golden_sparse6(table, mode, node; subjects, ids=nothing)
    rows = table[(table[:,1] .== mode) .& (table[:,3] .== node),:]
    times = [Float64.(rows[(rows[:,2] .== "time") .& (rows[:,4] .== i),6])
        for i in subjects]
    values = [Float64.(rows[(rows[:,2] .== "value") .& (rows[:,4] .== i),6])
        for i in subjects]
    IrregularFunctionalData(times,values; ids)
end

end
