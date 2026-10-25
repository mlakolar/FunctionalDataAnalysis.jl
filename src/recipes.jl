# Plot wrappers carry only precomputed data. All numerical work lives in plot_data.jl.
struct _FDAPlot{K,D}
    data::D
    node::Int
end
_FDAPlot{K}(data, node=0) where {K} = _FDAPlot{K,typeof(data)}(data, node)

struct _NodePlots{K,D}
    nodes::D
end
_NodePlots{K}(nodes) where {K} = _NodePlots{K,typeof(nodes)}(nodes)
_plot_payload(::Val{K}, data) where {K} =
    hasproperty(data, :nodes) ? _NodePlots{K}(data.nodes) : _FDAPlot{K}(data)
_plot_title(p::_FDAPlot, text) = p.node == 0 ? text : "Node $(p.node): $text"

@recipe function f(p::_NodePlots{K}) where {K}
    layout --> (length(p.nodes), 1)
    for (j, data) in enumerate(p.nodes)
        @series begin
            subplot := j
            _FDAPlot{K}(data, j)
        end
    end
    nothing
end

@recipe function f(fd::Union{GridFunctionalData,IrregularFunctionalData,BasisFunctionalData,MultivariateFunctionalData})
    grid = pop!(plotattributes, :evaluation_grid, nothing)
    curves = pop!(plotattributes, :curves, :all)
    _plot_payload(Val(:curves), curve_data(fd; grid, curves))
end

@recipe function f(p::_FDAPlot{:curves})
    d = p.data
    xlabel --> "Time"
    ylabel --> "Value"
    title --> _plot_title(p, "Functional data ($(d.design))")
    for j in eachindex(d.curves)
        @series begin
            seriestype --> :path
            markershape --> (d.design === :sparse ? :circle : :none)
            label --> "Curve $(d.curves[j])"
            d.times[j], d.values[j]
        end
    end
    nothing
end

@recipe function f(m::SmoothedMean)
    _FDAPlot{:mean}(mean_data(m))
end

@recipe function f(p::_FDAPlot{:mean})
    xlabel --> "Time"
    ylabel --> "Mean"
    title --> _plot_title(p, "Smoothed mean")
    label --> "Mean"
    seriestype --> :path
    p.data.grid, p.data.values
end

"""
    modes_plot(fit, k; multipliers=(-2, -1, 1, 2), kwargs...)

Plot the mean and mode-of-variation curves from [`modes_of_variation`](@ref).
The default multipliers follow fdapace 0.6.0's envelope boundaries. Titles
show the component, eigenvalue, and individual FVE. Multivariate fits have
one panel per node. Load Plots.jl before calling; `kwargs` set plot attributes.
"""
modes_plot(fit, k; multipliers=(-2, -1, 1, 2), kwargs...) =
    RecipesBase.plot(_plot_payload(Val(:modes), modes_of_variation(fit, k; multipliers)); kwargs...)

@recipe function f(p::_FDAPlot{:modes})
    d = p.data
    xlabel --> "Time"
    ylabel --> "Value"
    title --> _plot_title(p, "Mode $(d.component): λ=$(round(d.eigenvalue; sigdigits=5)), FVE=$(round(100*d.fve; digits=2))%")
    @series begin
        seriestype --> :path
        linewidth --> 2
        label --> "Mean"
        d.grid, d.mean
    end
    for j in eachindex(d.multipliers)
        @series begin
            seriestype --> :path
            label --> "c=$(d.multipliers[j])"
            d.grid, d.values[:, j]
        end
    end
    nothing
end

"""
    scree_plot(fit; kwargs...)

Plot individual FVE bars and cumulative FVE, following fdapace 0.6.0
`CreateScreePlot`, using [`scree_data`](@ref). Show selected K and the FVE
threshold when applicable. All retained components are displayed. Load Plots.jl
first. Plot attributes may be overridden with `kwargs`.
"""
scree_plot(fit; kwargs...) = RecipesBase.plot(_plot_payload(Val(:scree), scree_data(fit)); kwargs...)

@recipe function f(p::_FDAPlot{:scree})
    d = p.data
    xlabel --> "Number of components"
    ylabel --> "Fraction of variance explained"
    title --> _plot_title(p, "Scree plot ($(d.rule), K=$(d.chosen))")
    ylims --> (0, 1.05)
    @series begin
        seriestype := :bar
        label --> "Individual FVE"
        d.components, d.individual
    end
    @series begin
        seriestype := :path
        markershape --> :circle
        label --> "Cumulative FVE"
        d.components, d.cumulative
    end
    @series begin
        seriestype := :vline
        linestyle --> :dash
        label --> "Selected K=$(d.chosen)"
        [d.chosen]
    end
    if d.threshold !== nothing
        @series begin
            seriestype := :hline
            linestyle --> :dot
            label --> "FVE threshold=$(d.threshold)"
            [d.threshold]
        end
    end
    nothing
end

"""
    cov_plot(x; kwargs...)
    cov_plot(f::FPCA; surface=:stored, kwargs...)

Plot [`covariance_data`](@ref) as a heatmap, a package display default.
Use `seriestype=:surface` for a surface plot. FPCA accepts `surface=:fitted`
for the full retained positive-semidefinite projection. Load Plots.jl first;
other keywords set plot attributes.
"""
cov_plot(x; kwargs...) = RecipesBase.plot(_FDAPlot{:covariance}(covariance_data(x)); kwargs...)
cov_plot(f::FPCA; surface=:stored, kwargs...) =
    RecipesBase.plot(_FDAPlot{:covariance}(covariance_data(f; surface)); kwargs...)

@recipe function f(p::_FDAPlot{:covariance})
    d = p.data
    seriestype --> :heatmap
    xlabel --> "Time (first component)"
    ylabel --> "Time (second component)"
    title --> "Covariance ($(d.surface))"
    colorbar_title --> "Covariance"
    d.grid1, d.grid2, transpose(d.values)
end

"""
    design_plot(fd; diagonal=false, kwargs...)

Plot observed time pairs using [`design_data`](@ref). Diagonal pairs are
excluded by default and color counts use 1, 2, 3, 4+, as in fdapace 0.6.0.
Exact counts remain available in `design_data`. Multivariate data have one
panel per node. Load Plots.jl first; `kwargs` override plot attributes.
"""
design_plot(fd; diagonal=false, kwargs...) =
    RecipesBase.plot(_plot_payload(Val(:design), design_data(fd; diagonal)); kwargs...)

@recipe function f(p::_FDAPlot{:design})
    d = p.data
    seriestype --> :scatter
    xlabel --> "Observed time"
    ylabel --> "Observed time"
    title --> _plot_title(p, "Observation design (diagonal=$(d.diagonal))")
    label --> ""
    aspect_ratio --> :equal
    marker_z --> d.counts.display
    clims --> (1, 4)
    colorbar_title --> "Pair count"
    colorbar_ticks --> ([1, 2, 3, 4], ["1", "2", "3", "4+"])
    d.counts.x, d.counts.y
end

"""
    fit_plot(fit, i; grid=nothing, kwargs...)

Plot the original observations and selected-K reconstruction from
[`fitted_curve_data`](@ref). `grid=nothing` uses the fit's stored grid(s),
following fdapace 0.6.0 `CreatePathPlot`. MFPCA has one panel per node.
Load Plots.jl first; `kwargs` override plot attributes.
"""
function fit_plot(fit, i; grid=nothing, kwargs...)
    data = grid === nothing ? fitted_curve_data(fit, i) : fitted_curve_data(fit, i; grid)
    RecipesBase.plot(_plot_payload(Val(:fit), data); kwargs...)
end

@recipe function f(p::_FDAPlot{:fit})
    d = p.data
    xlabel --> "Time"
    ylabel --> "Value"
    title --> _plot_title(p, "Curve $(d.curve): $(d.method), K=$(d.ncomponents)")
    @series begin
        seriestype := :scatter
        label --> "Observed"
        d.observed.times, d.observed.values
    end
    @series begin
        seriestype --> :path
        label --> "Fitted"
        d.grid, d.values
    end
    nothing
end

@recipe function f(d::Union{FPCADerivative,MultivariateFPCADerivative})
    grid = pop!(plotattributes, :evaluation_grid, nothing)
    curves = pop!(plotattributes, :curves, :all)
    ylabel --> "Derivative"
    _plot_payload(Val(:curves), curve_data(d; grid, curves))
end

"""
    band_plot(b; kwargs...)
    band_plot(b::MeanBands; coverage=b.coverage, kwargs...)

Plot precomputed [`fitted_bands`](@ref) or [`mean_bands`](@ref) results as
ribbons around their fitted curves or means, using [`band_data`](@ref).
For mean results `coverage` selects a computed type; the default shows all
computed types. Multivariate results use one panel per node. Load Plots.jl
first; `kwargs` override plot attributes. Default ribbon opacity is 0.2, a
package display choice. No bootstrap is run by plotting.
"""
band_plot(b; kwargs...) =
    RecipesBase.plot(_plot_payload(Val(:band), band_data(b)); kwargs...)
band_plot(b::MeanBands; coverage=b.coverage, kwargs...) =
    RecipesBase.plot(_FDAPlot{:band}(band_data(b; coverage)); kwargs...)

@recipe function f(p::_FDAPlot{:band})
    d = p.data
    xlabel --> "Time"
    ylabel --> "Value"
    title --> _plot_title(p, "$(100*(1-d.alpha))% $(d.target) uncertainty")
    for i in axes(d.values, 1)
        @series begin
            seriestype --> :path
            label --> (d.target === :mean ? "Mean ($(d.types[i]))" : "Curve $(d.curves[i]) ($(d.types[i]))")
            ribbon --> (d.values[i, :] - d.lower[i, :], d.upper[i, :] - d.values[i, :])
            fillalpha --> 0.2
            d.grid, d.values[i, :]
        end
    end
    nothing
end
