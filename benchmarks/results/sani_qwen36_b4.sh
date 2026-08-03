#!/usr/bin/env bash
# Capture the illegal-memory-access location at batch=4 on Qwen3.6-35B-A3B.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
W=$SRC_DIR/repos/ECSIE/benchmarks/workloads/oom_repro.json
SANI=/usr/local/cuda-13.2/bin/compute-sanitizer
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/sani_qwen36_b4

# Skip the first 500 launches (model load + warm-up) then capture next 300.
ECSIE_BATCHED_FFN=1 "$SANI" --tool=memcheck \
    --launch-skip 500 --launch-count 300 \
    --print-limit 5 \
    "$BIN" --model "$M" --workload "$W" --batch-size 4 --out-tps /dev/null \
    > "${OUT}.stdout" 2> "${OUT}.stderr"
echo "rc=$?"
echo "--- first 5 errors ---"
head -80 "${OUT}.stdout"
