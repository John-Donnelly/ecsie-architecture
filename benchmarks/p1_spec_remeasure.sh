#!/usr/bin/env bash
# P1.6 follow-up: re-measure spec-on all-CPU vs hybrid (repeated, for variance)
# and confirm the verify forward uses the single-token path (batched=0) under the
# depth-1 lens config. Decides whether the +10% spec-on gain was a real
# composition gap or balloon variance.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_spec_remeasure.log
LENS="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46"
: > "$OUT"
arm() {
    echo "## $1 ##" | tee -a "$OUT"
    env "${@:2}" ECSIE_MOE_PATHCOUNT=1 "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:|moe-path" | head -3 | sed "s/^/    /" | tee -a "$OUT"
}
arm "spec-on all-CPU r1" $LENS ECSIE_CPU_EXPERTS=on
arm "spec-on all-CPU r2" $LENS ECSIE_CPU_EXPERTS=on
arm "spec-on hybrid r1"  $LENS ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
arm "spec-on hybrid r2"  $LENS ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
echo "DONE" | tee -a "$OUT"
