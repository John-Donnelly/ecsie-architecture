#!/usr/bin/env bash
# Phase 1 — CPU MoE worker-count sweep.  The box is a Ryzen 7 5800X: 8 physical
# cores / 16 SMT threads.  Default runner picks 8 (physical).  If the kernel is
# memory-LATENCY bound, SMT (12/16) hides stalls and helps; if it is execution-
# port / bandwidth bound, >8 contends and hurts.  Spec off, ext4-resident model.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_thread_sweep.log
: > "$OUT"

echo "[cache] $(grep Cached /proc/meminfo)" | tee -a "$OUT"
for T in 4 6 8 12 16; do
    echo "######## ECSIE_CPU_MOE_THREADS=$T ########" | tee -a "$OUT"
    env ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_CPU_MOE_THREADS=$T \
        "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]" | head -1 | tee -a "$OUT"
done
echo "DONE" | tee -a "$OUT"
