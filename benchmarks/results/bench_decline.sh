#!/usr/bin/env bash
# Single warm-decode pass with decline-reason diagnostics.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
R=$SRC_DIR/repos/ECSIE/benchmarks/results

# Warm cache first.
"$BIN" --model "$M" --workload "$W" --out-tps /dev/null > /dev/null 2>&1
# Real run.
"$BIN" --model "$M" --workload "$W" --out-tps /dev/null 2>&1 | \
    grep -E 'warm_step_rate=|fused_multi_(q4k|ternary):|dispatch tiers|decline breakdown'
