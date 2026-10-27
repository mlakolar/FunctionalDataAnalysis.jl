#!/bin/bash
# One exclusive single-CPU step runs one route's replicate block.
set -euo pipefail
: "${FDA_PROJECT:?Set FDA_PROJECT to an instantiated package checkout}"
: "${STUDY_CONFIG:?Set STUDY_CONFIG to the absolute configuration CSV path}"
: "${STUDY_OUTPUT:?Set STUDY_OUTPUT to the absolute results directory}"
[[ $# == 5 ]] || { echo "usage: worker.sh CELL ROUTE FIRST LAST OUTPUT_BASENAME" >&2; exit 2; }
cell=$1 route=$2 first=$3 last=$4 output=$5
[[ $cell =~ ^[1-9][0-9]*$ && $first =~ ^[1-9][0-9]*$ && $last =~ ^[1-9][0-9]*$ ]] || exit 2
case $route in
    boxplot|outliergram|msplot|kmeans_curves|kmeans_scores|kcfc|mixture) ;;
    *) exit 2 ;;
esac
printf -v expected 'cell-%03d-%s-%04d-%04d.csv' "$cell" "$route" "$first" "$last"
[[ $output == "$expected" && $first -le $last ]] || exit 2
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec julia --project="$FDA_PROJECT" "$FDA_PROJECT/benchmark/outlier_study/run.jl" \
    "$STUDY_CONFIG" "$cell" "$STUDY_OUTPUT/$output" "$first" "$last" "$route"
