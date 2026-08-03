#!/usr/bin/env bash
# Phase LT (ternary graph capture) A/B vs baseline.  First trial of each is
# cold-cache; trials 2-5 are warm decode.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
R=$SRC_DIR/repos/ECSIE/benchmarks/results

# Warm OS page cache.
"$BIN" --model "$M" --workload "$W" --out-tps /dev/null > /dev/null 2>&1

run_one () {
    local tag="$1"; shift
    local out
    out=$(env "$@" ECSIE_DUMP_TOKENS="$R/tok_phase_lt.txt" "$BIN" \
          --model "$M" --workload "$W" --out-tps /dev/null 2>&1)
    printf '\n=== %s ===\n' "$tag"
    echo "$out" | grep -E 'warm_step_rate=|fused_multi_ternary:'
    echo -n "  out: "; head -c 80 "$R/tok_phase_lt.txt.text" 2>/dev/null; echo
}

echo "--- baseline (Phase LT off) ---"
for i in 1 2 3 4 5; do run_one "baseline_${i}"; done

echo
echo "--- Phase LT on ---"
for i in 1 2 3 4 5; do run_one "phase_lt_${i}" ECSIE_PHASE_LT=on; done
