# Generate FDA milestone-4 reference values from FunctionalGraphicalModels.jl.
# Reference: branch bsplinekit, commit 6f0eac3d5cbe0fad93e88dcad28f389d89b61495.
# Copy that checkout to a temporary directory before instantiating it. Then run:
#   julia +release scripts/instantiate.jl
#   julia +release --project=. /path/to/FDA/test/reference/fgm_m4.jl /path/to/FDA/test/data
# The original FGM checkout must not be instantiated or modified.

using FunctionalGraphicalModels
using Pkg

const FGM_BRANCH = "bsplinekit"
const FGM_COMMIT = "6f0eac3d5cbe0fad93e88dcad28f389d89b61495"
const OUTPUT_DIRECTORY = isempty(ARGS) ? joinpath(@__DIR__, "..", "data") : only(ARGS)
const FGM_ROOT = dirname(dirname(pathof(FunctionalGraphicalModels)))
readchomp(`git -C $FGM_ROOT rev-parse HEAD`) == FGM_COMMIT ||
    error("FGM reference must use branch $FGM_BRANCH at commit $FGM_COMMIT")
# A copied worktree's .git file may still point at the preserved checkout.
# Check the loaded estimator files themselves against that pinned commit too.
for source in ("src/data.jl", "src/fpca.jl", "src/doubly.jl")
    read(joinpath(FGM_ROOT, source), String) ==
        read(`git -C $FGM_ROOT show $(FGM_COMMIT * ":" * source)`, String) ||
        error("FGM reference source $source differs from pinned commit $FGM_COMMIT")
end

function header(io, description)
    println(io, "# ", description)
    println(io, "# FunctionalGraphicalModels.jl ", pkgversion(FunctionalGraphicalModels),
        "; branch=", FGM_BRANCH,
        "; commit=", FGM_COMMIT, "; Julia=", VERSION)
    dependencies = Pkg.dependencies()
    for name in ("BSplineKit", "CovarianceSelection", "ProximalBase", "ProximalSolvers")
        matches = filter(p -> p.second.name == name, collect(dependencies))
        isempty(matches) && continue
        dependency = only(matches).second
        println(io, "# ", name, "=", dependency.version,
            "; git_revision=", dependency.git_revision,
            "; tree_hash=", dependency.tree_hash)
    end
    println(io, "# basis,quantity,node,row,column,value")
end

write_row(io, basis, quantity, node, row, column, value) =
    println(io, join((basis, quantity, node, row, column, repr(value)), ','))

function write_fit(io, basis, fitted)
    n, p, m = size(fitted.scores)
    for j in 1:p
        for g in eachindex(fitted.grid)
            write_row(io, basis, "mean", j, g, 0, fitted.mean[j, g])
        end
        for k in 1:m
            write_row(io, basis, "eigenvalue", j, k, 0, fitted.eigenvalues[k, j])
            for g in eachindex(fitted.grid)
                write_row(io, basis, "eigenfunction", j, g, k,
                    fitted.eigenfunctions[g, k, j])
            end
            for i in 1:n
                write_row(io, basis, "score", j, i, k, fitted.scores[i, j, k])
            end
        end
    end
end

function dense_reference(path)
    grid = [0.0, 0.06, 0.17, 0.32, 0.5, 0.69, 0.83, 0.94, 1.0]
    n, p = 12, 2
    # Entirely deterministic, with distinct marginal and pooled eigenvalues.
    values = [0.4j + (0.3 + 0.1j)*t +
        (1.1 + 0.2j)*cospi(2i/n)*cospi(t) +
        (0.5 + 0.15j)*sinpi(4i/n)*sinpi(2t) +
        0.12cospi(6i/n)*cospi(3t) for i in 1:n, j in 1:p, t in grid]
    data = DenseFunctionalData(values, grid)
    open(path, "w") do io
        header(io, "Deterministic dense marginal and pooled FPCA; n=12, p=2, ncomponents=2; FGM score variances divide by n.")
        for (g, t) in enumerate(grid)
            write_row(io, "input", "grid", 0, g, 0, t)
        end
        for g in eachindex(grid), j in 1:p, i in 1:n
            write_row(io, "input", "value", j, i, g, values[i, j, g])
        end
        for basis in (:marginal, :pooled)
            write_fit(io, basis, fit_fpca(data; basis, ncomponents=2))
        end
    end
end

function sparse_reference(path)
    grid = collect(range(0.0, 1.0; length=21))
    n, p = 18, 2
    times = Matrix{Vector{Float64}}(undef, n, p)
    values = similar(times)
    for j in 1:p, i in 1:n
        count = 4 + mod(i + j, 3)
        # Unequal curve lengths and interior times exercise subject weighting
        # and interpolation. Endpoints ensure that the work grid is covered.
        times[i, j] = [k == 0 ? 0.0 : k == count - 1 ? 1.0 :
            (k + 0.16sin(i + 2j + k))/(count - 1) for k in 0:(count - 1)]
        values[i, j] = [0.3j + 0.25t +
            (1.0 + 0.2j)*cospi(2i/n)*cospi(t) +
            (0.6 + 0.1j)*sinpi(4i/n)*sinpi(2t) +
            0.08cospi(6i/n)*cospi(3t) +
            0.03sin(i + 3j + 7t) for t in times[i, j]]
    end
    bandwidth, noise = 0.3, [0.04, 0.07]
    fitted = fit_sparse_fpca(SparseFunctionalData(times, values);
        grid, ncomponents=2, bandwidth, noise_variance=noise, kernel=:gaussian)
    open(path, "w") do io
        header(io, "Deterministic sparse marginal FPCA; n=18, p=2, ncomponents=2; Gaussian kernel, subject weighting, bandwidth=0.3, supplied noise variances 0.04 and 0.07.")
        println(io, "# FGM positive-eigenvalue cutoff is 100eps(Float64) times maximum absolute eigenvalue; all positive components enter conditional scores.")
        for (g, t) in enumerate(grid)
            write_row(io, "input", "grid", 0, g, 0, t)
        end
        for j in 1:p
            write_row(io, "input", "noise_variance", j, 0, 0, noise[j])
            for i in 1:n, k in eachindex(times[i, j])
                write_row(io, "input", "time", j, i, k, times[i, j][k])
                write_row(io, "input", "value", j, i, k, values[i, j][k])
            end
        end
        write_fit(io, :marginal, fitted)
        for j in 1:p, t in eachindex(grid), s in eachindex(grid)
            write_row(io, :marginal, "projected_covariance", j, s, t,
                fitted.covariances[s, t, j])
        end
    end
end

dense_reference(joinpath(OUTPUT_DIRECTORY, "m4_fgm_dense.csv"))
sparse_reference(joinpath(OUTPUT_DIRECTORY, "m4_fgm_sparse.csv"))
println("Wrote dense and sparse milestone-4 references to ", OUTPUT_DIRECTORY)
