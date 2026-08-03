#!/usr/bin/env bash
# eagle1fwd on the dense 9B via per-position SSM state slots (v2 head, native convention).
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M9="A:/AI/Models/Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
SIDE="$M9.eagle3"; V2=/b/Source/repos/ECSIE/tools/eagle3_train/qwen9b_eagle1fwd_v2.eagle3
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_9b_e1f_ssm.txt; : > "$OUT"
run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_EAGLE3_LAYERS=1,15,30 ECSIE_DUMP_TOKENS="$D/es_$tag" \
    "$BIN" --model "$M9" --workload "$WL" 2>&1 \
    | grep -iE "latency\]|spec:|EAGLE1FWD ignored|eagle1fwd disabled|prefill\]" | tee -a "$OUT"
  echo "coh: $(cat "$D/es_$tag.text" 2>/dev/null | head -c 120 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
run "base ref" base
cp "$SIDE" "$SIDE.prod.bak"; cp "$V2" "$SIDE"
trap '[ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"' EXIT
for d in 1 2; do
  run "e1f-SSM d=$d (v2 head)" e$d \
    ECSIE_SPEC=on ECSIE_SPEC_DEPTH=$d ECSIE_SPEC_BATCHED=on ECSIE_SPEC_EAGLE1FWD=1 ECSIE_SPEC_DIAG=1
done
echo DONE | tee -a "$OUT"
