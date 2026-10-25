#!/bin/bash -l
#SBATCH --job-name=fda-sigma2
#SBATCH --array=1-48%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-sigma2-%A_%a.out

# Planning template only. The owner selects site allocation/partition and
# reviews total resources before submission. No account or credentials here.
set -euo pipefail
: "${JULIA_MODULE:?Set JULIA_MODULE to the Discovery Julia module name}"
: "${FDA_PROJECT:?Set FDA_PROJECT to an instantiated package checkout}"
: "${SIGMA2_CONFIG:?Set SIGMA2_CONFIG to the absolute configuration CSV path}"
: "${SIGMA2_OUTPUT:?Set SIGMA2_OUTPUT to the absolute results directory}"

module load "$JULIA_MODULE"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$SIGMA2_OUTPUT"
printf -v cell '%03d' "$SLURM_ARRAY_TASK_ID"
srun --ntasks=1 --cpus-per-task=1 julia --project="$FDA_PROJECT" \
    "$FDA_PROJECT/benchmark/sigma2/run.jl" "$SIGMA2_CONFIG" \
    "$SLURM_ARRAY_TASK_ID" "$SIGMA2_OUTPUT/cell-$cell.csv"
