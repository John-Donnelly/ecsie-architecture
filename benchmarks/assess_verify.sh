#!/usr/bin/env bash
# Verify auto-defaults + true batched spec verify + PPL gate. NO concurrent work.
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
SIDE="$M30.eagle3"; NEW=/b/Source/repos/ECSIE/tools/eagle3_train/qwen30b_eagle1fwd.eagle3
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
PPL=/b/Source/repos/ECSIE/benchmarks/workloads/ppl_short.txt
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_verify.txt; : > "$OUT"
run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_DUMP_TOKENS="$D/av_$tag" "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate|latency\]|spec:|moe-overlap|defaulting" | tee -a "$OUT"
  echo "coh: $(cat "$D/av_$tag.text" 2>/dev/null | head -c 120 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
# 1) default env x2 (auto-defaults should engage; report 2nd)
run "30B DEFAULT (auto) #1" auto1
run "30B DEFAULT (auto) #2" auto2
# 2) forced-off = old out-of-box behaviour
run "30B CPU_EXPERTS=off (old default)" off ECSIE_CPU_EXPERTS=off
# 3) explicit champion (sanity: should equal auto)
run "30B explicit champion" champ ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
# 4) PPL gate on the default(auto) path
echo "===== PPL default(auto) =====" | tee -a "$OUT"
ECSIE_PERPLEXITY_CORPUS="$PPL" "$BIN" --model "$M30" --workload "$WL" 2>&1 \
  | grep -iE "PPL=" | tee -a "$OUT"
# 5) TRUE batched resident verify (first real measurement) + overlap diag
cp "$SIDE" "$SIDE.prod.bak"; cp "$NEW" "$SIDE"
trap '[ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"' EXIT
for d in 1 2 3; do
  run "e1f d=$d TRUE-batched verify + overlap" e1f$d \
    ECSIE_SPEC=on ECSIE_SPEC_DEPTH=$d ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1 \
    ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3 ECSIE_CPU_EXPERTS_BATCHED=on \
    ECSIE_SPEC_DIAG=1 ECSIE_MOE_OVERLAP=1
done
echo DONE | tee -a "$OUT"
