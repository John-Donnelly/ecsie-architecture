#!/usr/bin/env bash
# ECSIE benchmark suite runner
# Runs all benchmark workloads and writes results to benchmarks/results/
#
# Usage:
#   ./benchmarks/run_all.sh [--model <path>] [--build-dir <dir>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS="$SCRIPT_DIR/results"

# Default model path (WSL mount of A:\AI\Models\...)
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/ci}"

# Parse CLI overrides.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL_PATH="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2";  shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

# Locate benchmark binaries (CMake places them in the build dir).
BIN_TPS="$BUILD_DIR/ecsie_bench_measure_tps"
BIN_LAT="$BUILD_DIR/ecsie_bench_latency_variance"
BIN_GPU="$BUILD_DIR/ecsie_bench_gpu_utilisation"

for bin in "$BIN_TPS" "$BIN_LAT" "$BIN_GPU"; do
    if [[ ! -f "$bin" ]]; then
        echo "[error] $bin not found."
        echo "        Run: cmake --preset ci -B build/ci -DECSIE_BUILD_BENCHMARKS=ON"
        echo "             cmake --build build/ci -j\$(nproc)"
        exit 1
    fi
done

mkdir -p "$RESULTS"

MODEL_ARG=""
if [[ -f "$MODEL_PATH" ]]; then
    MODEL_ARG="--model $MODEL_PATH"
    echo "[ecsie] model: $MODEL_PATH"
else
    echo "[warn] model not found at $MODEL_PATH — running without model (routing-only)"
fi

WORKLOADS=(stable mixed high_entropy)

for wl in "${WORKLOADS[@]}"; do
    echo "[ecsie] running workload: $wl"

    # shellcheck disable=SC2086
    "$BIN_TPS" \
        $MODEL_ARG \
        --workload "$SCRIPT_DIR/workloads/${wl}.json" \
        --out-tps  "$RESULTS/tps_${wl}.csv"

    # shellcheck disable=SC2086
    "$BIN_LAT" \
        $MODEL_ARG \
        --workload    "$SCRIPT_DIR/workloads/${wl}.json" \
        --out-latency "$RESULTS/latency_${wl}.csv"

    # shellcheck disable=SC2086
    "$BIN_GPU" \
        $MODEL_ARG \
        --workload "$SCRIPT_DIR/workloads/${wl}.json" \
        --out-gpu  "$RESULTS/gpu_util_${wl}.csv"
done

# Merge per-workload TPS into a single summary file.
echo "step,tps,workload" > "$RESULTS/tps.csv"
for wl in "${WORKLOADS[@]}"; do
    [[ -f "$RESULTS/tps_${wl}.csv" ]] && \
        tail -n +2 "$RESULTS/tps_${wl}.csv" | awk -v wl="$wl" -F, '{print $0","wl}' \
        >> "$RESULTS/tps.csv"
done

echo "[ecsie] benchmarks complete. Results in $RESULTS/"

