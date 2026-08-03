#!/usr/bin/env bash
# v1.0.214 — A/B the logit-lens self-draft vs the trained EAGLE-3 head.
# Each arm is a fresh process (ECSIE_SPEC_DRAFT is read once at start).
# Reads [latency] (warm_step_rate) + spec: (accept rate) from stderr.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
TAG="${TAG:-q4gguf}"
LAYER="${LAYER:-46}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/lensdraft_${TAG}.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

echo "MODEL=$MODEL LAYER=$LAYER" | tee -a "$OUT"
arm "A eagle3 head"          ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1
arm "B lens L$LAYER"          ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=$LAYER
arm "C lens L$LAYER + r0.1"   ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=$LAYER ECSIE_SPEC_ACCEPT=ratio:0.1

echo "=== SUMMARY ($TAG) ===" | tee -a "$OUT"
grep -aE "^########|spec:" "$OUT" | tee -a "$OUT"
