#!/bin/bash
# One exclusive single-CPU step calls one route's replicate block.
set -euo pipefail
: "${FDA_PROJECT:?Set FDA_PROJECT to an instantiated package checkout}"
: "${SCALING_CONFIG:?Set SCALING_CONFIG to the absolute configuration CSV path}"
: "${SCALING_OUTPUT:?Set SCALING_OUTPUT to the absolute results directory}"
[[ $# == 5 ]] || { echo "usage: worker.sh CELL ROUTE FIRST LAST OUTPUT_BASENAME" >&2; exit 2; }
cell=$1 route=$2 first=$3 last=$4 output=$5
[[ $cell =~ ^[1-9][0-9]*$ && $first =~ ^[1-9][0-9]*$ && $last =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $route == dense || $route == face || $route == pace || $route == face_sparse ]] || exit 2
printf -v expected 'cell-%03d-%s-%04d-%04d.csv' "$cell" "$route" "$first" "$last"
[[ $output == "$expected" && $first -le $last ]] || exit 2
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec julia --project="$FDA_PROJECT" "$FDA_PROJECT/benchmark/scaling/run.jl" \
    "$SCALING_CONFIG" "$cell" "$SCALING_OUTPUT/$output" "$first" "$last" "$route"
