#!/bin/bash -l
#SBATCH --job-name=fda-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-scaling-%j.out

# Planning template: the owner selects the current Discovery partition and
# allocation, and reviews memory, time and worker concurrency before submission.
# The full benchmark is owner-run; this file does not authorize submission.
# Resource flags can be overridden on the sbatch command line.
set -euo pipefail
: "${JULIA_MODULE:?Set JULIA_MODULE to the Discovery Julia module name}"
: "${FDA_PROJECT:?Set FDA_PROJECT to an instantiated package checkout}"
: "${SCALING_CONFIG:?Set SCALING_CONFIG to the absolute configuration CSV path}"
: "${SCALING_UNITS:?Set SCALING_UNITS to the absolute work-list TSV path}"
: "${SCALING_OUTPUT:?Set SCALING_OUTPUT to the absolute results directory}"
: "${SLURM_NTASKS:?Run this script inside a Slurm allocation}"
workers=${SCALING_WORKERS:-$SLURM_NTASKS}
[[ $workers =~ ^[1-9][0-9]*$ && $workers -le $SLURM_NTASKS ]] || {
    echo "SCALING_WORKERS must be positive and no larger than SLURM_NTASKS" >&2; exit 2;
}
[[ ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || {
    echo "Request one CPU per task; select concurrency with --ntasks" >&2; exit 2;
}
for path in "$FDA_PROJECT" "$SCALING_CONFIG" "$SCALING_UNITS" "$SCALING_OUTPUT"; do
    [[ $path == /* ]] || { echo "Project, input and output paths must be absolute" >&2; exit 2; }
done

module load "$JULIA_MODULE"
command -v parallel >/dev/null
export FDA_PROJECT SCALING_CONFIG SCALING_OUTPUT
# Each step needs an explicit memory request; without one (or with --mem=0)
# a step claims all job memory and the steps run one at a time.
export SCALING_STEP_MEM=${SCALING_STEP_MEM:-1900M}
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$FDA_PROJECT/benchmark/scaling/work_units.py" "$SCALING_CONFIG" "$SCALING_UNITS" --check
mkdir -p "$SCALING_OUTPUT"
# Preserve these tiny run inputs. A resumed job must keep their bytes and the
# joblog unchanged so GNU Parallel's sequence numbers identify the same work.
for pair in "$SCALING_CONFIG:configurations.csv" "$SCALING_UNITS:work-units.tsv"; do
    source=${pair%:*} snapshot="$SCALING_OUTPUT/${pair##*:}"
    if [[ -e $snapshot ]]; then
        cmp -s "$source" "$snapshot" || {
            echo "Resume input changed: $snapshot; use a fresh output directory" >&2; exit 2;
        }
    else
        cp -n "$source" "$snapshot"
    fi
done
parallel_args=(--jobs "$workers" --colsep '\t' --header :
    --joblog "$SCALING_OUTPUT/parallel-joblog.tsv" --halt soon,fail=1)
[[ ! -e "$SCALING_OUTPUT/parallel-joblog.tsv" ]] || parallel_args+=(--resume-failed)
# Fit/metric failures are successful benchmark attempts. An aborted worker
# stops new launches, retains its files, and causes a nonzero driver exit.
parallel "${parallel_args[@]}" \
    'srun --exclusive --exact -N1 -n1 -c1 --mem=${SCALING_STEP_MEM:-1900M} bash "$FDA_PROJECT/benchmark/scaling/worker.sh" {cell} {route} {first_replicate} {last_replicate} {output}' \
    :::: "$SCALING_UNITS"
