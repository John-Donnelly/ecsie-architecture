#!/usr/bin/env bash
# v1.0.214 — fine-tune ceiling diagnostic across the 3 model formats.
# Each model runs in a FRESH process (the static SpecDiagAccum dtor prints the
# [specdiag] summary at process exit).  Strict K=1.  E[target_top1] estimates the
# accept ceiling a fine-tuned head could reach; headroom = E[p1] - cur_accept.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
DIR=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/ceiling_diag.log
: > "$OUT"

run() {
    local tag="$1"; local model="$2"
    echo "######## $tag ########" | tee -a "$OUT"
    env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DIAG=1 \
        "$BIN" --model "$model" --workload "$WL" 2>&1 \
        | grep -aE "\[specdiag\]|spec:|MoE dispatch tier" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

run "q4_gguf"        "$DIR/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
run "ecsie_baseline" "$DIR/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie"
run "n8h8_ecsie"     "$DIR/Qwen3-30B-A3B-Instruct-2507-n8h8.ecsie"

echo "=== SUMMARY ===" | tee -a "$OUT"
grep -aE "^########|\[specdiag\]" "$OUT" | tee -a "$OUT"
