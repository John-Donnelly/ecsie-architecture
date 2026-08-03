#!/usr/bin/env bash
# v1.0.136 A/B: baseline (cleanup only) vs ECSIE_VRAM_PREFETCH=on (the
# existing predictive-prefetch path from v1.0.124, now on top of v1.0.135's
# async H2D + sticky LRU).
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
R=$SRC_DIR/repos/ECSIE/benchmarks/results

# Warm the OS page cache first.
"$BIN" --model "$M" --workload "$W" --out-tps /dev/null > /dev/null 2>&1

run_one () {
    local tag="$1"; shift
    local out
    out=$(env "$@" "$BIN" --model "$M" --workload "$W" --out-tps /dev/null 2>&1)
    printf '\n=== %s ===\n' "$tag"
    echo "$out" | grep -E 'warm_step_rate=|fused_multi_(q4k|ternary):|dispatch tiers|decline breakdown|ExpertPredictor'
}

# Baseline (no prefetch) — 3 trials.
for i in 1 2 3; do
    run_one "baseline_${i}"
done

# Prefetch on — 3 trials.
for i in 1 2 3; do
    run_one "prefetch_${i}" ECSIE_VRAM_PREFETCH=on
done
