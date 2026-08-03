#!/usr/bin/env bash
# Phase 0 (in-scope, tunable): sweep ECSIE_LENS_DRAFT_LAYER to find the layer whose
# logit-lens draft maximizes pos-1 acceptance. We fixed 46; a different mid/late
# layer may accept higher (=> more tok/step => higher user-facing tps). This is
# fine-tuning the existing lens self-draft, NOT adaptive depth (Phase 3).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p0_lens_layer_sweep.log
: > "$OUT"
for L in 40 42 43 44 45 46 47; do
    echo "## LENS_LAYER=$L ##" | tee -a "$OUT"
    env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=$L \
        ECSIE_CPU_EXPERTS=on \
        "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -2 | sed "s/^/    /" | tee -a "$OUT"
done
echo "DONE" | tee -a "$OUT"
