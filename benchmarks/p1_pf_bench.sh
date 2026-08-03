#!/usr/bin/env bash
# Quick repeated all-CPU spec-off bench (prefetch kernel) vs the ~9.9 fused baseline.
set -u
B=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
M="${MODEL:-$HOME/ecsie_models/q4.gguf}"
W=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
cat "$M" > /dev/null 2>&1
for i in 1 2 3 4; do
    printf 'r%s: ' "$i"
    ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on "$B" --model "$M" --workload "$W" 2>&1 | grep -a latency
done
echo DONE
