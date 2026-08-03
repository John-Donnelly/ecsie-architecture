#!/usr/bin/env bash
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/workloads/batch_test.json
SANI=/usr/local/cuda-13.2/bin/compute-sanitizer
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/sani_batch4

ECSIE_BATCHED_FFN=1 "$SANI" --tool=memcheck --launch-skip 200 --launch-count 200 \
    --print-limit 10 \
    "$BIN" --model "$M" --workload "$W" --batch-size 4 --out-tps /dev/null \
    > "${OUT}.stdout" 2> "${OUT}.stderr"
echo "rc=$?"
echo "--- sanitizer (first 30 lines) ---"
head -30 "${OUT}.stdout"
echo "--- error summary ---"
grep -E "ERROR SUMMARY|illegal|access|invalid" "${OUT}.stdout" | head -10
