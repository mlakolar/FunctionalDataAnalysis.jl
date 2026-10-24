# Generate the FDA inputs for mfpca_m4.R from the repository root:
# julia --project=. test/reference/mfpca_m4.jl
# Generated with Julia 1.13.2, FunctionalDataAnalysis 0.1.0.
using FunctionalDataAnalysis, LinearAlgebra
import StatsAPI

n = 24
grids = [collect(range(0, 1; length=17)).^1.2,
    -1 .+ 3 .* collect(range(0, 1; length=19)).^1.1,
    2 .+ 0.7 .* collect(range(0, 1; length=21)).^1.3]
latent = [sqrt(2 * (n-1)/n) * cos(2pi*k*(i-0.5)/n) for i in 1:n, k in 1:6]
mixing = [2.0 0.3 0 0 0 0; 0 1.2 0 0 0 0;
    0.7 0 1.5 0.2 0 0; 0 0.4 0 0.8 0 0;
    0.2 0 0.5 0 1.1 0.1; 0 0.3 0 0.2 0 0.5]
joint = latent * mixing'
nodes = [begin
    w = quadrature_weights(g)
    linear = g .- dot(w, g)/sum(w)
    phi = hcat(ones(length(g))/sqrt(sum(w)), linear/sqrt(dot(w, linear.^2)))
    GridFunctionalData(j .+ g' .+ joint[:, 2j-1:2j] * phi', g)
end for (j,g) in enumerate(grids)]
data = MultivariateFunctionalData(nodes)
fitted = StatsAPI.fit(FPCA, data; basis=:marginal, ncomponents=2, maxK=2)
open(joinpath(@__DIR__, "..", "data", "m4_mfpca_input.csv"), "w") do io
    println(io, "# Julia ", VERSION, "; FunctionalDataAnalysis 0.1.0; deterministic 24 subjects, 3 nodes; quantity,node,row,column,value")
    function write_matrix(quantity, node, a)
        for c in axes(a, 2), r in axes(a, 1)
            println(io, quantity, ',', node, ',', r, ',', c, ',', a[r, c])
        end
    end
    for j in 1:3
        write_matrix("grid", j, reshape(grids[j], :, 1))
        write_matrix("value", j, nodes[j].values)
        write_matrix("function", j, eigenfunctions(fitted)[j].values)
        write_matrix("score", j, scores(fitted, j))
    end
end

# A mixed dense/PACE first step with deliberately nonzero-mean conditional
# scores. Export univariate reconstructions only as the grid data required by
# R's given expansion; the supplied functions and scores determine step two.
dense = nodes[1][1:8]
g = [2.0, 2.1, 2.4, 2.7, 3.0]
times = [g[[1, 2+i%2, 5]] for i in 1:8]
sparse = IrregularFunctionalData(times,
    [fill(5+cos(i), length(t)) for (i,t) in enumerate(times)]; domain=(2,3))
pace_data = MultivariateFunctionalData([dense, sparse])
options = (; ncomponents=[2, 1], maxK=[2, 1], grid=[nothing, g],
    mean=[nothing, (; grid=g, values=zeros(5))],
    covariance=[nothing, (; grid=g, surface=ones(5, 5))],
    noise_variance=[nothing, 0.2])
pace = StatsAPI.fit(FPCA, pace_data; basis=:marginal, options...)
open(joinpath(@__DIR__, "..", "data", "m4_mfpca_pace_input.csv"), "w") do io
    println(io, "# Julia ", VERSION, "; FunctionalDataAnalysis 0.1.0; mixed dense/PACE, 8 subjects, 2 nodes; quantity,node,row,column,value")
    function write_matrix(quantity, node, a)
        for c in axes(a, 2), r in axes(a, 1)
            println(io, quantity, ',', node, ',', r, ',', c, ',', a[r, c])
        end
    end
    for j in 1:2
        univariate = component(pace, j)
        write_matrix("grid", j, reshape(univariate.grid, :, 1))
        write_matrix("value", j, reconstruct(univariate).values)
        write_matrix("mean", j, univariate.mean.values)
        write_matrix("function", j, eigenfunctions(pace)[j].values)
        write_matrix("score", j, scores(pace, j))
    end
end
