#!/usr/bin/env bash
# Verify the spec-ON prefill fix: e1f d=2/d=3 total tps should now approach
# (decode-only) levels; base run for reference. NO concurrent work.
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M30="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
SIDE="$M30.eagle3"; NEW=/b/Source/repos/ECSIE/tools/eagle3_train/qwen30b_eagle1fwd.eagle3
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_e1f_prefill.txt; : > "$OUT"
run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_DUMP_TOKENS="$D/pf_$tag" "$BIN" --model "$M30" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate|latency\]|spec:|moe-overlap" | tee -a "$OUT"
  echo "coh: $(cat "$D/pf_$tag.text" 2>/dev/null | head -c 120 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
run "base (auto-default) ref" b0
cp "$SIDE" "$SIDE.prod.bak"; cp "$NEW" "$SIDE"
trap '[ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"' EXIT
for d in 2 3; do
  run "e1f d=$d batched verify + FAST prefill" d$d \
    ECSIE_SPEC=on ECSIE_SPEC_DEPTH=$d ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1 \
    ECSIE_CPU_EXPERTS_BATCHED=on ECSIE_SPEC_DIAG=1
done
echo DONE | tee -a "$OUT"
