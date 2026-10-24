using Documenter
using FunctionalDataAnalysis

makedocs(;
    sitename = "FunctionalDataAnalysis.jl",
    modules = [FunctionalDataAnalysis],
    format = Documenter.HTML(; edit_link = "main"),
    pages = [
        "Home" => "index.md",
        "Interface" => "interface.md",
        "Bases" => "bases.md",
        "Roadmap" => "roadmap.md",
    ],
)

deploydocs(;
    repo = "github.com/mlakolar/FunctionalDataAnalysis.jl.git",
    devbranch = "main",
)
