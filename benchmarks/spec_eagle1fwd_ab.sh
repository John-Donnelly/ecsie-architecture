#!/usr/bin/env bash
# A/B: eagle1fwd single-forward spec (new head + loop) vs base 16.7.
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
GGUF="A:/AI/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
SIDE="$GGUF.eagle3"
NEW=/b/Source/repos/ECSIE/tools/eagle3_train/qwen30b_eagle1fwd.eagle3
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results
OUT=$D/spec_eagle1fwd_ab.txt; : > "$OUT"
cp "$SIDE" "$SIDE.prod.bak"; cp "$NEW" "$SIDE"
restore() { [ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"; }
trap restore EXIT
run () {
  local label="$1"; local tag="$2"; shift 2
  echo "==================== $label ====================" | tee -a "$OUT"
  env "$@" ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3 ECSIE_SPEC_DIAG=1 \
    ECSIE_DUMP_TOKENS="$D/e1f_$tag" \
    "$BIN" --model "$GGUF" --workload "$WL" 2>&1 \
    | grep -iE "warm_step_rate|spec:|latency\]" | tee -a "$OUT"
  echo "--- coherence (first 240 chars) ---" | tee -a "$OUT"
  head -c 240 "$D/e1f_$tag".seq0.text 2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"; echo "" | tee -a "$OUT"
}
run "B0 base (spec OFF)" b0
run "E1 eagle1fwd depth=1" e1 ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1
run "E2 eagle1fwd depth=2" e2 ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1
run "E3 eagle1fwd depth=3" e3 ECSIE_SPEC=on ECSIE_SPEC_DEPTH=3 ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1
echo DONE | tee -a "$OUT"
