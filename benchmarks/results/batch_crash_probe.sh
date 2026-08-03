#!/usr/bin/env bash
# Probe Phase E batched MoE at various batch sizes to find the crash.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/workloads/batch_test.json

for B in 4 8; do
    echo "=== batch_size=$B ==="
    ECSIE_BATCHED_FFN=1 timeout 60 "$BIN" --model "$M" --workload "$W" \
        --batch-size "$B" --out-tps /dev/null 2>&1 | tail -8
    echo "rc=$?"
done
