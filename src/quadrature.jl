"""
    quadrature_weights(grid)

Trapezoidal integration weights on a finite, strictly increasing grid of at
least two points. The weights sum to `last(grid) - first(grid)`.

The result preserves the input's floating-point type. Integer grids produce
`Float64` weights. Invalid grids throw an `ArgumentError`.
"""
function quadrature_weights(grid::AbstractVector{<:Real})
    Base.require_one_based_indexing(grid)
    length(grid) >= 2 || throw(ArgumentError("the grid needs at least two points"))
    all(isfinite, grid) || throw(ArgumentError("the grid must be finite"))
    x = float.(collect(grid))
    all(>(0), diff(x)) || throw(ArgumentError("the grid must be strictly increasing"))
    w = similar(x)
    w[1] = (x[2] - x[1]) / 2
    w[end] = (x[end] - x[end - 1]) / 2
    for t in 2:length(x)-1
        w[t] = (x[t + 1] - x[t - 1]) / 2
    end
    w
end
