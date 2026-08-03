#!/usr/bin/env bash
# v1.0.214b — mirror the winning config (lens_hybrid D2 batched) to .ecsie
# baseline + both n8h8 30B variants, with a lens-K1 reference per format.
# eff_decode_tps = (tokens/steps) * warm_step_rate.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
DIR=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/hybrid_mirror.log
: > "$OUT"

arm() {
    local label="$1"; local model="$2"; shift 2
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$model" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:|MoE dispatch tier" | head -3 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

ECSIE_BASE="$DIR/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie"
N8H8="$DIR/Qwen3-30B-A3B-Instruct-2507-n8h8.ecsie"
BF16N8H8="$DIR/Qwen3-30B-A3B-Instruct-2507-bf16direct-n8h8.ecsie"

K1="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46"
HYB="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on"

arm "ecsie_base  lens K1"        "$ECSIE_BASE" $K1
arm "ecsie_base  hybrid D2 bat"  "$ECSIE_BASE" $HYB
arm "n8h8        lens K1"        "$N8H8"       $K1
arm "n8h8        hybrid D2 bat"  "$N8H8"       $HYB
arm "bf16_n8h8   lens K1"        "$BF16N8H8"   $K1
arm "bf16_n8h8   hybrid D2 bat"  "$BF16N8H8"   $HYB
echo "DONE"
