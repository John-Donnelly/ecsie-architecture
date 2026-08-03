#!/usr/bin/env bash
# v1.0.214 — in-process relaxed-acceptance sweep.  ONE model load; the bench
# flips ECSIE_SPEC_ACCEPT between generate() runs.  Measures pos-1 accept (K=1)
# under strict vs ratio:R.  Reads [sweep] lines from stderr.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct.json}"
TAG="${TAG:-q4gguf}"
MODES="${MODES:-strict,ratio:0.5,ratio:0.3,ratio:0.1,ratio:0.05}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/inproc_${TAG}.log

echo "MODEL=$MODEL"   >  "$OUT"
echo "MODES=$MODES"   >> "$OUT"
env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_ACCEPT_SWEEP="$MODES" \
    "$BIN" --model "$MODEL" --workload "$WL" >> "$OUT" 2>&1
echo "RUN_EXIT=$?" >> "$OUT"
echo "=== [sweep] lines ($TAG) ==="
grep -aE "\[sweep\]" "$OUT"
