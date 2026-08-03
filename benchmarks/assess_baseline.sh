#!/usr/bin/env bash
# Session assessment: 30B (default vs champion env) + 9B (default), warm runs.
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
M9="A:/AI/Models/Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_baseline.txt; : > "$OUT"
run(){ local label="$1" model="$2" tag="$3"; shift 3
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_DUMP_TOKENS="$D/ab_$tag" "$BIN" --model "$model" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate|latency\]|prefill|tok/s" | tee -a "$OUT"
  echo "coh: $(cat "$D/ab_$tag".text 2>/dev/null | head -c 120 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
run "30B default #1"  "$M30" d30a
run "30B default #2"  "$M30" d30b
run "30B champion (CPU_EXPERTS+HOTCOLD=3)" "$M30" c30 ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
run "9B default #1"   "$M9"  d9a
run "9B default #2"   "$M9"  d9b
echo DONE | tee -a "$OUT"
