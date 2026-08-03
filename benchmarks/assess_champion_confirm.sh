#!/usr/bin/env bash
# Confirm INT_DOT / threads=6 wins vs warm champion control + PPL gate.
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
PPL=/b/Source/repos/ECSIE/benchmarks/workloads/ppl_short.txt
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_champion_confirm.txt; : > "$OUT"
run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_DUMP_TOKENS="$D/cc_$tag" "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate|latency\]" | tee -a "$OUT"
  echo "coh: $(cat "$D/cc_$tag.text" 2>/dev/null | head -c 110 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
ppl(){ local label="$1"; shift
  echo "===== PPL $label =====" | tee -a "$OUT"
  env "$@" ECSIE_PERPLEXITY_CORPUS="$PPL" "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "PPL=" | tee -a "$OUT"; }
run "control champion (8thr, fp)" ctl
run "INT_DOT=1"                   id   ECSIE_CPU_INT_DOT=1
run "threads=6"                   t6   ECSIE_CPU_MOE_THREADS=6
run "INT_DOT+threads=6"           idt6 ECSIE_CPU_INT_DOT=1 ECSIE_CPU_MOE_THREADS=6
run "control champion AGAIN"      ctl2
ppl "baseline"
ppl "INT_DOT=1" ECSIE_CPU_INT_DOT=1
echo DONE | tee -a "$OUT"
