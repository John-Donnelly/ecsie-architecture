#!/usr/bin/env bash
# v1.0.135 verification: sticky-VRAM LRU eviction + async pinned-staging H2D.
# Look for: (a) coherent output, (b) higher fused_multi_q4k engagement,
# (c) higher warm_step_rate vs the 1.88-2.26 baseline.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
R=$SRC_DIR/repos/ECSIE/benchmarks/results

run_one () {
    local tag="$1"; shift
    local out
    out=$(env "$@" ECSIE_DUMP_TOKENS="$R/tok_v135.txt" "$BIN" \
          --model "$M" --workload "$W" --out-tps /dev/null 2>&1)
    printf '\n=== %s ===\n' "$tag"
    echo "$out" | grep -E 'warm_step_rate=|fused_multi_(q4k|ternary):|dispatch tiers' | tail -8
    echo -n "  out: "; head -c 80 "$R/tok_v135.txt.text" 2>/dev/null; echo
}

# 5 trials median.
for i in 1 2 3 4 5; do
    run_one "v135_trial_${i}"
done
