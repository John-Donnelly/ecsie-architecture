#!/usr/bin/env bash
# v1.0.214 — lens_hybrid (lens +1 spliced over eagle depth-2, batched verify) vs
# references. eff_decode_tps = (tokens/steps) * warm_step_rate.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/lens_hybrid_ab.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -3 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

arm "lens K1 (ref)"            ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46
arm "lens_hybrid D2 seq"       ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46
arm "lens_hybrid D2 batched"   ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
arm "lens_hybrid D3 batched"   ECSIE_SPEC=on ECSIE_SPEC_DEPTH=3 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
echo "DONE"
