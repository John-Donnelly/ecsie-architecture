#!/usr/bin/env bash
# Phase 0 (in-scope tuning, NOT adaptive=Phase3): compare FIXED spec depths.
# lens d1 is production (1.82 tok/step). Test whether the (recent, May-28) trained
# eagle3 sidecar makes a fixed depth-2/3 lens_hybrid yield more tok/step ->
# higher user-facing tps. Reports spec rate, steps, and computes tok/step.
# user-facing tok/s = warm_step_rate(steps/s) * tokens_emitted/steps.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p0_depth_compare.log
: > "$OUT"
arm() {
    echo "## $1 ##" | tee -a "$OUT"
    env "${@:2}" ECSIE_CPU_EXPERTS=on "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:" | head -2 | sed "s/^/    /" | tee -a "$OUT"
}
arm "lens d1 (production)"   ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens        ECSIE_LENS_DRAFT_LAYER=46
arm "lens_hybrid d2"         ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
arm "lens_hybrid d3"         ECSIE_SPEC=on ECSIE_SPEC_DEPTH=3 ECSIE_SPEC_DRAFT=lens_hybrid ECSIE_LENS_DRAFT_LAYER=46 ECSIE_SPEC_BATCHED=on
arm "eagle3 d3 (head only)"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=3 ECSIE_SPEC_DRAFT=eagle3
echo "DONE" | tee -a "$OUT"
