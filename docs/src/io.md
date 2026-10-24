# Table input and simulation

Grid and irregular constructors consume Tables.jl sources with ordinary
Symbol column selectors. Curve order follows first appearance of IDs.
Within curves, times are sorted. Grid input uses the sorted union of all
times; absent rows become missing without interpolation. Duplicate
`(id, time)` pairs throw for grid data; irregular data can explicitly use
`ties=:mean`. Missing irregular observations throw rather than being dropped.
Nullable time columns are accepted when no entries are missing. Grid input
treats `-0.0` and `0.0` as the same time; duplicate rows at that time still throw.

```jldoctest
julia> using FunctionalDataAnalysis, Tables

julia> table = (subject=[2, 1, 2, 1], t=[0.0, 0.0, 1.0, 1.0], y=[1.0, 2.0, 3.0, 4.0]);

julia> fd = GridFunctionalData(table; id=:subject, time=:t, value=:y);

julia> ids(fd)
2-element Vector{Any}:
 2
 1

julia> Tables.columntable(fd)
(id = Any[2, 2, 1, 1], time = [0.0, 1.0, 0.0, 1.0], value = Union{Missing, Float64}[1.0, 3.0, 2.0, 4.0])
```

All univariate representations support optional IDs. IDs are copied, must
be unique, and follow every subset and conversion. Their absence is
`nothing`. Duplicate subject subsets of labeled data throw. Concatenation
requires labels on both sides or neither and rejects duplicate IDs across
inputs. Labeled multivariate components must agree in the same order;
unlabeled components use the existing subject order.

`Tables.columntable(fd)` exports long `(id, time, value)` columns for raw
data. Unlabeled data use positional IDs in the exported table, without
changing `ids(fd)`. Basis data require an explicit `grid` keyword to produce
a long table. No sample grid is chosen implicitly.

Table constructors extract columns and then enter a typed function barrier.
Runtime Symbol selectors are not required to pass `@inferred`; scalar types
are checked directly. Numeric methods and subject subsetting retain inference
checks. Integer IDs do not widen Float32 measurement/time columns.

## Simulation

`simulate_kl` takes an explicit RNG. Scores are independent Gaussian draws
with the supplied variances. Measurement noise is independent Gaussian noise.
The returned `truth` is an independent data object on the same observed
times, containing the noise-free values. Regular sampling returns grid data;
sparse sampling returns irregular data with the requested count range and
distinct uniform observation times. Sampling times do not change when only
the noise standard deviation changes under the same RNG seed.

```jldoctest
julia> using FunctionalDataAnalysis, Random

julia> result = simulate_kl(Xoshiro(42), 3; mean=t -> 1 + t,
           eigenfunctions=fourier_eigenfunctions((0, 1), 3),
           eigenvalues=[1.0, 0.5, 0.25], noise_sd=0,
           sampling=RegularSampling([0.0, 0.5, 1.0]));

julia> size(result.scores)
(3, 3)

julia> result.data.values == result.truth.values
true
```

Fourier functions start with a constant, then sine/cosine pairs. Legendre
functions start with degree zero and use the shifted, normalized polynomials
on the supplied domain. Both generators are deterministic utilities. Seeded
tests check reproducibility, eigenfunction orthonormality, and sample means
and score variances for 20000 curves within six Monte Carlo standard errors.

```@docs
ids
RegularSampling
SparseSampling
simulate_kl
fourier_eigenfunctions
legendre_eigenfunctions
Tables.columntable(::BasisFunctionalData)
```
