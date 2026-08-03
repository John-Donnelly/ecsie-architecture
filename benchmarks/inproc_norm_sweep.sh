#!/usr/bin/env bash
# v1.0.214 — in-process head-norm placement sweep.  ONE model load; STRICT
# acceptance (measures the lossless true greedy-match rate).  Roadmap YELLOW
# remedy: "investigate norm placement."  Reads [normsweep] lines from stderr.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
TAG="${TAG:-q4gguf}"
COMBOS="${COMBOS:-default,no_fc,no_post,no_final,fusion,hidden_first,no_fc+no_post,fusion+no_fc}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/normsweep_${TAG}.log

echo "MODEL=$MODEL"   >  "$OUT"
echo "COMBOS=$COMBOS" >> "$OUT"
env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_EAGLE3_NORM_SWEEP="$COMBOS" \
    "$BIN" --model "$MODEL" --workload "$WL" >> "$OUT" 2>&1
echo "RUN_EXIT=$?" >> "$OUT"
grep -aE "\[normsweep\]" "$OUT"
