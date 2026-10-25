#!/usr/bin/env julia
# Packs exports from fda 6.3.0, fda.usc 2.2.0 and refund 0.1-40.
# Run from the repository root: julia data/make_artifacts.jl [download-url]
# Uses Julia standard libraries only; does not download or publish anything.
using Pkg.Artifacts
using SHA
using TOML

const DATA_DIR = @__DIR__
const PAYLOAD = joinpath(DATA_DIR, "build", "payload")
const TARBALL = joinpath(DATA_DIR, "build", "fda-datasets-v1.tar.gz")
const BINDINGS = joinpath(DATA_DIR, "..", "Artifacts.toml")
const URL = isempty(ARGS) ?
    "https://github.com/mlakolar/FunctionalDataAnalysis.jl/releases/download/DATASETS_RELEASE_TAG/fda-datasets-v1.tar.gz" : only(ARGS)

isdir(PAYLOAD) || error("run Rscript data/build_datasets.R before packing artifacts")
for name in ("growth", "canadian_weather", "gait", "tecator", "cd4", "dti")
    isfile(joinpath(PAYLOAD, name, "metadata.csv")) || error("missing export for $name")
end
hash = create_artifact() do dir
    for file in readdir(PAYLOAD)
        cp(joinpath(PAYLOAD, file), joinpath(dir, file))
    end
end
archive_artifact(hash, TARBALL)
# Pkg's gzip writer inserts the current time. Clear only the gzip MTIME
# field so repeated packing of the same tree gives identical bytes. This
# field is outside the compressed-payload checksum; reject header CRCs.
open(TARBALL, "r+") do io
    header = read(io, 10)
    header[1:3] == [0x1f, 0x8b, 0x08] || error("expected a gzip artifact archive")
    iszero(header[4] & 0x02) || error("gzip header CRC is unsupported")
    seek(io, 4)
    write(io, zeros(UInt8, 4))
end
sha = bytes2hex(open(sha256, TARBALL))
bind_artifact!(BINDINGS, "fda_datasets", hash; lazy=true, force=true,
    download_info=[(URL, sha)])
# Hash the exact normalized file that will be uploaded.
binding = TOML.parsefile(BINDINGS)["fda_datasets"]
@assert binding["lazy"] && binding["git-tree-sha1"] == string(hash)
println("Tarball: ", TARBALL)
println("git-tree-sha1: ", hash)
println("sha256: ", sha)
println("Download URL: ", URL)
