#!/usr/bin/env bash
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_t6_ab.txt; : > "$OUT"
for arm in 8 6 8 6; do
  echo "===== threads=$arm =====" | tee -a "$OUT"
  ECSIE_CPU_MOE_THREADS=$arm "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate" | tee -a "$OUT"
done
echo DONE | tee -a "$OUT"
