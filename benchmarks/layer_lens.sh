#!/usr/bin/env bash
# v1.0.214 — per-layer logit lens across the 3 model formats.  For each base
# layer, how often does its hidden (→ final_norm → LM head) already argmax to the
# token the FULL model generates?  Shows where the next token locks in, and thus
# whether a later fusion layer would carry more info than the head's layer 44.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_60.json
DIR=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/layer_lens.log
: > "$OUT"

run() {
    local tag="$1"; local model="$2"
    echo "######## $tag ########" | tee -a "$OUT"
    env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_LAYER_LENS=1 \
        "$BIN" --model "$model" --workload "$WL" 2>&1 \
        | grep -aE "\[layerlens\]" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

run "q4_gguf"        "$DIR/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
run "ecsie_baseline" "$DIR/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie"
run "n8h8_ecsie"     "$DIR/Qwen3-30B-A3B-Instruct-2507-n8h8.ecsie"
echo "DONE"
