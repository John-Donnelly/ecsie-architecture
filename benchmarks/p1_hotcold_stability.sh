#!/usr/bin/env bash
# Phase 1 P1.4 — hot/cold (N=3) stability across NOVEL workloads vs all-CPU.
# Confirms the overlap gain holds beyond the long_instruct tuning workload and
# that the hybrid is stable (no garbage / no regression) on diverse inputs.
# Spec OFF to isolate raw decode (warm_step_rate = tok/s).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WLDIR=$SRC_DIR/repos/ECSIE/benchmarks/workloads
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_hotcold_stability.log
: > "$OUT"
run() {
    env "${@:2}" "$BIN" --model "$MODEL" --workload "$1" 2>&1 \
        | grep -aE "^\[latency\]" | head -1 | sed "s/^/    /"
}
for w in novel_math novel_code novel_prose novel_technical_qa novel_chat; do
    echo "==== $w ====" | tee -a "$OUT"
    echo "  all-CPU:" | tee -a "$OUT"
    run "$WLDIR/$w.json" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on | tee -a "$OUT"
    echo "  hybrid N=3:" | tee -a "$OUT"
    run "$WLDIR/$w.json" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3 | tee -a "$OUT"
done
echo "DONE" | tee -a "$OUT"
