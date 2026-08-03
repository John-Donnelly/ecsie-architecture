#!/usr/bin/env bash
# Clean back-to-back: lens K1 sequential vs lens K1 batched vs lens_hybrid D2
# batched — same thermal window. Tests whether batched verify alone (no eagle
# tail) captures the ~9 tok/s gain.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/k1_batched_test.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -2 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

arm "lens K1 seq"         ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46
arm "lens K1 batched"     ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
arm "lens_hybrid D2 batched" ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
echo "DONE"
