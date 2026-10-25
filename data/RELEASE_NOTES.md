Classic functional datasets, version 1

This asset contains CSV exports of growth, CanadianWeather and gait from R
fda 6.3.0; tecator from fda.usc 2.2.0; and cd4 and DTI from refund 0.1-40.
It is the lazy `fda_datasets` artifact for FunctionalDataAnalysis.jl.

DTI includes all 382 scans and both tracts: cca (93 points) followed by rcst
(55 points), with original missing cells, subject IDs and visit covariates.
The CD4 export contains all 1,888 observed counts from 366 subjects.
Tecator uses the 215-row fda.usc object.

Each dataset includes metadata, citation, original R package DESCRIPTION and
dataset documentation, and source licence text. The data retain their source
GPL terms separately from the Julia package's MIT licence. Tecator's original
redistribution permission note is attached as TECATOR.txt; publications should
mention the Tecator Infratec Food and Feed Analyzer and Tecator.

For DTI publications include this acknowledgment: "The MRI/DTI data were
collected at Johns Hopkins University and the Kennedy-Krieger Institute".

Generation scripts and pinned source checksums are in the repository's data/
directory. Numerical reference aggregates come from the native R objects.
