# Milestone 5, task 4: dataset handoff

Implemented `datasets()` and `load_dataset(name; sample=false)` for all six
candidates. DTI follows the 25 October author decision: a multivariate object
with cca first (93 points) and rcst second (55 points), each on its own grid.
All 382 scans, 142 subject IDs, visits, and scalar covariates are retained.
`component(d, 1)` selects cca. Both source missing masks are preserved.

No candidate was excluded. The audit inspected the versioned CRAN archives,
their DESCRIPTION files, dataset documentation, and native data objects.
Package MD5s match the downloaded CRAN PACKAGES index; SHA-256s are recorded in
`source-lock.csv`. fda 6.3.0 and refund 0.1-40 declare GPL (>= 2); fda.usc 2.2.0
declares GPL-2. Tecator's original StatLib permission explicitly allows
redistribution with the attached permission note. Source notices, GPL text,
and Tecator's note accompany both samples and full data. The DTI acknowledgment
is included in metadata, the public docstring, user docs and release notes.

## Artifact and publication

The full tarball is left locally at **`data/build/fda-datasets-v1.tar.gz`**
(ignored, so binary data and source archives are not committed). The script
also installs the content in the local artifact cache. The repository includes
`Artifacts.toml` with `lazy = true`, an immutable git-tree hash, tarball SHA-256,
and the URL of the tarball attached to the `datasets-v1` release.

- git-tree-sha1: `b4a7d08b6826cf57a804710db60204957237f738`
- tarball SHA-256: `ed2db0310705dac6852d37aca6d14bf59e2c68e24269c87254ceffdb16660cd2`
- Upload notes: `data/RELEASE_NOTES.md`.

Mladen creates the repository release and chooses its tag. For a release
named `datasets-v1`, run these commands from the repository root **after the
release exists**, before publishing the package change:

```sh
julia +release data/make_artifacts.jl 'https://github.com/mlakolar/FunctionalDataAnalysis.jl/releases/download/datasets-v1/fda-datasets-v1.tar.gz'
gh release upload datasets-v1 data/build/fda-datasets-v1.tar.gz --repo mlakolar/FunctionalDataAnalysis.jl
gh release edit datasets-v1 --repo mlakolar/FunctionalDataAnalysis.jl --notes-file data/RELEASE_NOTES.md
FDA_TEST_ARTIFACTS=1 julia +release --project=. -e 'using FunctionalDataAnalysis, Tables, Test; include("test/datasets.jl")'
```

Use the chosen tag in place of `datasets-v1` if different. The packing command
rewrites the binding to the exact uploaded bytes; include that Artifacts.toml
change in the controller's reviewed commit. Do not replace an asset already
published under an immutable binding. To test an actual download, run the
last command in an environment without this artifact hash cached (the local
packing test alone exercises cache loading).

To recreate all inputs from scratch with base R and Julia standard libraries:

```sh
Rscript data/build_datasets.R
julia data/make_artifacts.jl
```

R source packages are not installed. Archives are downloaded at the pinned
versions, checked, unpacked, and loaded directly, avoiding their R dependency
trees. Use R 4.5.1 to reproduce the recorded R-version metadata and tree hash.
Source archives, full exports and the tarball live under ignored `data/build/`.

## Defaults and choices left open by the specification

| Choice | Resolution and source |
|:--|:--|
| Public result | The requested five-field named tuple; source is a named tuple containing version, citation, checksums, rows, units, node order and representation notes |
| `sample` keyword | `false` by default (full data); `true` reads committed subsets for examples and network-free tests; package API choice |
| Storage | One platform-independent lazy artifact containing all six datasets; one release tarball; package storage choice |
| Numeric representation | Float64 functional values and coordinates; source integers vs doubles retained for scalar covariates; package defaults |
| Covariate key | Unique string column `id`, exactly matching `ids(data)`; source labels where supplied |
| Growth | Boys then girls; native boy/girl labels; sex derived from hgtm/hgtf; native uneven ages and cm from fda documentation |
| Weather | Temperature then raw precipitation, from the named source array slices; no log/zero replacement; day-of-year indices 1:365 because dailyAv supplies calendar-day labels rather than numeric coordinates |
| Weather covariates | place, province, latitude, west_longitude, region, geogindex; native order; monthly summaries outside the daily representation |
| Gait | Hip then knee; native standardized times and boy labels, fda array dimnames |
| Tecator | All 215 fda.usc rows, native wavelength grid and Fat/Water/Protein; no split and no substitution of StatLib's 240-row object |
| CD4 | Observed cells only, source month labels, full domain (-18,42); positional string IDs since refund has no row labels; no observed values or subjects dropped |
| DTI | Both nodes in the author-specified order; native scan IDs ID_visit; original repeated numeric ID and all scalar covariates; point-index grids 1:93 and 1:55 because refund supplies no physical coordinates |
| Missing values | Grid NA/NaN become the existing observed mask; CD4 NA cells are absent observations; covariate NA becomes missing; source sex factor labels become strings and original level order is recorded |
| Samples | Growth rows 1,2,40,41; weather/gait/Tecator/CD4 rows 1:3; DTI rows 1,2,43,44,125, covering repeated visits, both case classes and missing values in both nodes |
| CSV format | Version 1, one line per record, quoted fields/escaped quotes, explicit NA, provenance comment headers, numeric fields with 17 significant digits; dedicated reader to avoid new runtime dependencies |
| Archive recreation | Text files use mode 0644 and directories 0755; gzip MTIME is zeroed, preserving the payload checksum; package reproducibility choices |
| Tuning | None: loaders do not smooth, interpolate, normalize, select components, or randomize |
| Plot example | Four sample growth curves; dense FPCA explicitly sets ncomponents=2 and maxK=2; mean and first mode because task 2 is absent from this checkout |
| Hosting | Placeholder until the author creates the GitHub release; no tag, release, commit, branch or remote change made here |

Public API additions and the two-tract DTI representation are reflected in
the local roadmap, README, docstrings and Datasets page. The external plan
should likewise reflect the two new functions, optional offline sample mode
and author-decided multivariate DTI representation.
