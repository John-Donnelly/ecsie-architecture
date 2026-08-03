#!/usr/bin/env bash
# Phase 1 P1.4 — hot/cold N sweep + repeated all-CPU baselines (balloon variance).
# Find the N that peaks tok/s before the M=1 GPU dispatch becomes the bottleneck.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_hotcold_sweep.log
: > "$OUT"
arm() {
    echo "## $1 ##" | tee -a "$OUT"
    env "${@:2}" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]" | head -1 | tee -a "$OUT"
}
arm "all-CPU r1"       ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "all-CPU r2"       ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "hybrid HOTCOLD=3" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
arm "hybrid HOTCOLD=4" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=4
arm "hybrid HOTCOLD=5" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=5
arm "hybrid HOTCOLD=6" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=6
arm "hybrid HOTCOLD=3 (repeat)" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
echo "DONE" | tee -a "$OUT"
