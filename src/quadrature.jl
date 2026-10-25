function _validate_strict_grid(grid::AbstractVector{<:Real})
    Base.require_one_based_indexing(grid)
    length(grid) >= 2 || throw(ArgumentError("the grid needs at least two points"))
    all(isfinite, grid) || throw(ArgumentError("the grid must be finite"))
    # Match differences after floating conversion, including integer grids
    # whose adjacent values collapse to the same floating-point number.
    all(i -> float(grid[i+1]) - float(grid[i]) > 0, 1:length(grid)-1) ||
        throw(ArgumentError("the grid must be strictly increasing"))
    nothing
end

"""
    quadrature_weights(grid)

Trapezoidal integration weights on a finite, strictly increasing grid of at
least two points. The weights sum to `last(grid) - first(grid)`.

The result preserves the input's floating-point type. Integer grids produce
`Float64` weights. Invalid grids throw an `ArgumentError`.
"""
function quadrature_weights(grid::AbstractVector{<:Real})
    _validate_strict_grid(grid)
    x = float.(collect(grid))
    w = similar(x)
    w[1] = (x[2] - x[1]) / 2
    w[end] = (x[end] - x[end - 1]) / 2
    for t in 2:length(x)-1
        w[t] = (x[t + 1] - x[t - 1]) / 2
    end
    w
end
