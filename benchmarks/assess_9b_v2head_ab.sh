#!/usr/bin/env bash
set -uo pipefail
BIN=/b/Source/repos/ECSIE/bin/ecsie_bench_measure_tps.exe
M9="A:/AI/Models/Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
SIDE="$M9.eagle3"; V2=/b/Source/repos/ECSIE/tools/eagle3_train/qwen9b_eagle1fwd_v2.eagle3
WL=/b/Source/repos/ECSIE/benchmarks/workloads/long_chatml.json
D=/b/Source/repos/ECSIE/benchmarks/results; OUT=$D/assess_9b_v2head_ab.txt; : > "$OUT"
run(){ local label="$1" tag="$2"; shift 2
  echo "===== $label =====" | tee -a "$OUT"
  env "$@" ECSIE_EAGLE3_LAYERS=1,15,30 ECSIE_DUMP_TOKENS="$D/v2h_$tag" \
    "$BIN" --model "$M9" --workload "$WL" 2>&1 \
    | grep -iE "latency\]|spec:" | tee -a "$OUT"
  echo "coh: $(cat "$D/v2h_$tag.text" 2>/dev/null | head -c 110 | tr '\n' ' ')" | tee -a "$OUT"
  echo "" | tee -a "$OUT"; }
run "base ref" base
run "spec OLD head d1" old ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DIAG=1
cp "$SIDE" "$SIDE.prod.bak"; cp "$V2" "$SIDE"
trap '[ -f "$SIDE.prod.bak" ] && mv -f "$SIDE.prod.bak" "$SIDE"' EXIT
run "spec V2 head d1" v2d1 ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DIAG=1
run "spec V2 head d2" v2d2 ECSIE_SPEC=on ECSIE_SPEC_DEPTH=2 ECSIE_SPEC_DIAG=1
echo DONE | tee -a "$OUT"
