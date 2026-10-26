using FunctionalDataAnalysis, Tables, Test, Artifacts, TOML
import Pkg

# Resolve package dependencies before this file runs. Only the dataset artifact
# uses this empty depot, so a prior Julia/package cache cannot turn this into a
# cache-hit test. Restore the caller's process state even on a failed download.
mktempdir() do depot
    saved = copy(DEPOT_PATH)
    try
        empty!(DEPOT_PATH)
        push!(DEPOT_PATH,depot)
        withenv("FDA_TEST_ARTIFACTS"=>"1", "JULIA_PKG_SERVER"=>"") do
            @testset "Fresh dataset release download" begin
                @test !isdir(joinpath(depot,"artifacts"))
                include(joinpath(@__DIR__,"..","datasets.jl"))
                manifest = TOML.parsefile(joinpath(@__DIR__,"..","..","Artifacts.toml"))
                tree = Base.SHA1(manifest["fda_datasets"]["git-tree-sha1"])
                @test isdir(artifact_path(tree))
                @test Pkg.Artifacts.verify_artifact(tree)
                println("Verified fresh dataset artifact ",tree)
            end
        end
    finally
        empty!(DEPOT_PATH)
        append!(DEPOT_PATH,saved)
    end
end
