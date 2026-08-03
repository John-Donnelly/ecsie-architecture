#!/usr/bin/env bash
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/nsys_attn_v139

nsys profile \
  --trace=cuda --sample=none \
  --delay=20 --duration=4 \
  --output "$OUT" --force-overwrite=true \
  "$BIN" --model "$M" --workload "$W" --out-tps /dev/null \
  > "${OUT}.stdout" 2>&1
echo "rc=$?"
echo "=== top kernels ==="
nsys stats --report cuda_gpu_kern_sum --format csv "${OUT}.nsys-rep" 2>/dev/null \
  | head -30
