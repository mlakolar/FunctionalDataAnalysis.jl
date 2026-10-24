using Documenter
using FunctionalDataAnalysis
using Tables

makedocs(;
    sitename = "FunctionalDataAnalysis.jl",
    modules = [FunctionalDataAnalysis],
    format = Documenter.HTML(; edit_link = "main"),
    pages = [
        "Home" => "index.md",
        "Interface" => "interface.md",
        "Bases" => "bases.md",
        "Data types" => "data.md",
        "Table input and simulation" => "io.md",
        "Roadmap" => "roadmap.md",
    ],
)

deploydocs(;
    repo = "github.com/mlakolar/FunctionalDataAnalysis.jl.git",
    devbranch = "main",
)
