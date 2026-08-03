#!/usr/bin/env bash
# Phase 0 x Phase 1 combined: does the hot/cold overlap (P1.4) compose with the
# lens self-draft spec decode (Phase 0)?  Reports raw decode (spec off) and
# user-facing (spec on) for all-CPU vs hybrid N=3.
#   spec-off warm_step_rate = raw decode tok/s
#   spec-on  tps            = user-facing tok/s (accepted tokens / s)
# NOTE: CPU experts run in the single-token decode path; the spec multi-token
# verify may take the GPU batched path — this measures whether they compose.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_combined_spec.log
LENS="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46"
: > "$OUT"
arm() {
    echo "## $1 ##" | tee -a "$OUT"
    env "${@:2}" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:|accept" | head -3 | sed "s/^/    /" | tee -a "$OUT"
}
arm "spec-off all-CPU"        ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "spec-off hybrid N=3"     ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
arm "spec-on  all-CPU (P0)"   ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 ECSIE_CPU_EXPERTS=on
arm "spec-on  hybrid (P0+P1)" ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
echo "DONE" | tee -a "$OUT"
