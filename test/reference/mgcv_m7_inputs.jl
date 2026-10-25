# Run from the repository root: julia --project=. test/reference/mgcv_m7_inputs.jl
# Writes the package's own bases and penalties for test/reference/mgcv_m7.R.
# Deterministic: no random numbers. test/penalized.jl rebuilds the same inputs.
using FunctionalDataAnalysis

function m7_pls_inputs()
    n = 80
    x1 = [((i - 0.5) / n)^1.25 for i in 1:n]
    x2 = [mod(i * 0.6180339887498949, 1.0) for i in 1:n]
    z = [cos(3.0 * i) for i in 1:n]
    w = [0.5 + 1.5 * mod(i * 0.37, 1.0) for i in 1:n]
    spline = BSplineBasis(4, (0.0, 1.0), 10)
    fourier = FourierBasis((0.0, 1.0), 7)
    margin = BSplineBasis(4, (0.0, 1.0), 5)
    A, B = basis_matrix(margin, x1), basis_matrix(margin, x2)
    P = penalty(margin, 2)
    identity5 = [Float64(i == j) for i in 1:5, j in 1:5]
    (; x1, x2, z, w,
        spline_design=basis_matrix(spline, x1), spline_penalty=penalty(spline, 2),
        fourier_design=basis_matrix(fourier, x2)[:, 2:7],
        fourier_penalty=penalty(fourier, :harmonic_acceleration)[2:7, 2:7],
        tensor_design=reduce(vcat, [kron(A[i:i, :], B[i:i, :]) for i in 1:n]),
        tensor_penalty1=kron(P, identity5), tensor_penalty2=kron(identity5, P))
end

if abspath(PROGRAM_FILE) == @__FILE__
    inputs = m7_pls_inputs()
    open("test/data/m7_pls_input.csv", "w") do io
        println(io, "# FunctionalDataAnalysis m7_pls_inputs(); BSplineBasis, FourierBasis, penalty")
        println(io, "# quantity,row,column,value; nonzero entries only; dim_<name> rows give sizes")
        for name in keys(inputs)
            value = inputs[name]
            matrix = value isa AbstractVector ? reshape(value, :, 1) : value
            println(io, "dim_", name, ",", size(matrix, 1), ",", size(matrix, 2), ",0")
            for j in axes(matrix, 2), i in axes(matrix, 1)
                iszero(matrix[i, j]) && continue
                println(io, name, ",", i, ",", j, ",", repr(matrix[i, j]))
            end
        end
    end
end
