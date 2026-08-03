#!/usr/bin/env bash
# v1.0.214 — K=2 depth probe. Measures whether batched depth-2 verify buys
# tokens/forward on this Q4 MoE, or whether expert-divergence eats it.
# eff_decode_tps = (tokens/steps) * warm_step_rate  (prefill-independent).
# Arms: lens K=1 (ref), eagle3 K=1, eagle3 K=2 seq, eagle3 K=2 batched.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/k2_probe.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -3 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

arm "lens   K1"          ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46
arm "eagle3 K1"          ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1
arm "eagle3 K2 seq"      ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2
arm "eagle3 K2 batched"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_BATCHED=on
echo "DONE"
