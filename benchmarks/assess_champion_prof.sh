#!/usr/bin/env bash
# Task #33: champion 30B decode profile + CPU-MoE lever sweep (INT_DOT, threads).
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_champion_prof.txt; : > "$OUT"
run(){ local label="$1"; shift
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_PROFILE_SUMMARY=1 ECSIE_MOE_TIMING=1 \
    "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "latency\]|profiler|moe-timing|prefill\]" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
run "champion default (auto)"
run "INT_DOT=1"            ECSIE_CPU_INT_DOT=1
run "threads=6"            ECSIE_CPU_MOE_THREADS=6
run "threads=12"           ECSIE_CPU_MOE_THREADS=12
echo DONE | tee -a "$OUT"
