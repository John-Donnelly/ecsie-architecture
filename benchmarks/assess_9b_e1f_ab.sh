#!/usr/bin/env bash
# Dense 9B eagle1fwd A/B: convert the freshly-trained head -> sidecar, then
# base vs e1f spec on the 9B. Judge by TOTAL tps only (mean_step excludes verify).
set -uo pipefail
PY="A:/Users/JohnD/AppData/Local/Programs/Python/Python312/python.exe"
T=/b/Source/repos/ECSIE/tools/eagle3_train
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M9="A:/AI/Models/Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
SIDE="$M9.eagle3"
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_9b_e1f_ab.txt; : > "$OUT"

echo "== convert 9B eagle1fwd head ==" | tee -a "$OUT"
"$PY" "$T/convert_hf_to_ecsie.py" --hf "$T/runs_9b_eagle1fwd" \
  --out "$T/qwen9b_eagle1fwd.eagle3" 2>&1 | tail -8 | tee -a "$OUT"
ls -la "$T/qwen9b_eagle1fwd.eagle3" | tee -a "$OUT"

run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_EAGLE3_LAYERS=1,15,30 ECSIE_DUMP_TOKENS="$D/e9_$tag" \
    ECSIE_PROFILE_SUMMARY=1 "$BIN" --model "$M9" --workload "$WL" 2>&1 \
    | grep -iE "latency\]|spec:|prefill\]|profiler.*(step|verify|moe|lm_head|attention)" | tee -a "$OUT"
  echo "coh: $(cat "$D/e9_$tag.text" 2>/dev/null | head -c 120 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }

run "9B base warm#1" b1
run "9B base warm#2 (REF)" b2
# swap in the new head
[ -f "$SIDE" ] && cp "$SIDE" "$SIDE.prod.bak"
cp "$T/qwen9b_eagle1fwd.eagle3" "$SIDE"
trap '[ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"' EXIT
for d in 1 2; do
  run "9B e1f d=$d" e$d \
    ECSIE_SPEC=on ECSIE_SPEC_DEPTH=$d ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1 ECSIE_SPEC_DIAG=1
done
echo DONE | tee -a "$OUT"
