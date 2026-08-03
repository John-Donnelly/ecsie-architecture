#!/usr/bin/env bash
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
SANI=/usr/local/cuda-13.2/bin/compute-sanitizer
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/sani_initcheck

ECSIE_PHASE_LT=on "$SANI" --tool=initcheck \
  --launch-skip 100 --launch-count 100 \
  --print-limit 20 \
  "$BIN" --model "$M" --workload "$W" --out-tps /dev/null \
  > "${OUT}.stdout" 2> "${OUT}.stderr"
echo "rc=$?"
echo "--- initcheck output ---"
head -200 "${OUT}.stdout"
