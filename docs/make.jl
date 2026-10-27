using Documenter
using FunctionalDataAnalysis
using LinearAlgebra
using Tables
import StatsAPI

makedocs(;
    sitename = "FunctionalDataAnalysis.jl",
    modules = [FunctionalDataAnalysis],
    format = Documenter.HTML(; edit_link = "master"),
    pages = [
        "Home" => "index.md",
        "Interface" => "interface.md",
        "Bases" => "bases.md",
        "Data types" => "data.md",
        "Table input and simulation" => "io.md",
        "Smoothing" => "smoothing.md",
        "Penalized fitting" => "penalized.md",
        "FPCA" => "fpca.md",
        "Regression" => "regression.md",
        "Penalized regression" => "penalized_regression.md",
        "Functional CCA" => "cca.md",
        "Multivariate FPCA (MFPCA)" => "mfpca.md",
        "Cross-structure" => "cross_structure.md",
        "Plotting" => "plotting.md",
        "Uncertainty" => "uncertainty.md",
        "Datasets" => "datasets.md",
        "Comparison with fdapace" => "fdapace.md",
        "Multivariate FPCA validation" => "m4_report.md",
        "Cross-covariance validation" => "m4_cross_report.md",
        "Comparison with FGM" => "m4_fgm.md",
        "Roadmap" => "roadmap.md",
    ],
)

deploydocs(;
    repo = "github.com/mlakolar/FunctionalDataAnalysis.jl.git",
    devbranch = "master",
)
