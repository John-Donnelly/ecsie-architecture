#!/usr/bin/env bash
# v1.0.214c — confidence-gated verify-skip sweep on Q4 lens K1 (lossless).
# Low lens top1-top2 margin -> skip the verify forward, emit true greedy token.
# Watch: drafted < steps (skipped steps) and mean_step drop -> eff tps.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/verify_skip_ab.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -2 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

LENS="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46"
arm "no gate (ref)"   $LENS
arm "gate 2.0"        $LENS ECSIE_LENS_MARGIN_GATE=2.0
arm "gate 4.0"        $LENS ECSIE_LENS_MARGIN_GATE=4.0
arm "gate 8.0"        $LENS ECSIE_LENS_MARGIN_GATE=8.0
echo "DONE"
