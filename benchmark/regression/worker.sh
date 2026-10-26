#!/bin/bash
# One exclusive single-CPU step runs one route's replicate block.
set -euo pipefail
: "${FDA_PROJECT:?Set FDA_PROJECT to an instantiated package checkout}"
: "${REGRESSION_CONFIG:?Set REGRESSION_CONFIG to the absolute configuration CSV path}"
: "${REGRESSION_OUTPUT:?Set REGRESSION_OUTPUT to the absolute results directory}"
[[ $# == 5 ]] || { echo "usage: worker.sh CELL ROUTE FIRST LAST OUTPUT_BASENAME" >&2; exit 2; }
cell=$1 route=$2 first=$3 last=$4 output=$5
[[ $cell =~ ^[1-9][0-9]*$ && $first =~ ^[1-9][0-9]*$ && $last =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $route == fpc || $route == fpc_bic || $route == pfr ]] || exit 2
printf -v expected 'cell-%03d-%s-%04d-%04d.csv' "$cell" "$route" "$first" "$last"
[[ $output == "$expected" && $first -le $last ]] || exit 2
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec julia --project="$FDA_PROJECT" "$FDA_PROJECT/benchmark/regression/run.jl" \
    "$REGRESSION_CONFIG" "$cell" "$REGRESSION_OUTPUT/$output" "$first" "$last" "$route"
