#!/usr/bin/env bash
# v1.0.214 — relaxed-acceptance (ECSIE_SPEC_ACCEPT) sweep.
# Measures pos-1 EAGLE-3 acceptance at K=1 under strict vs ratio:R, then a
# batched-depth arm to probe effective TPS toward the Phase 0 gate.
# Each arm runs in a FRESH process.  Reads the engine's "spec:" + "[latency]"
# lines from stderr.
set -u
BIN=$SRC_DIR/repos/ECSIE/build_linux/bin/ecsie_bench_measure_tps
[ -x "$BIN" ] || BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct.json}"
TAG="${TAG:-q4gguf}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/relaxed_${TAG}.txt
: > "$OUT"

run() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -E '^\[latency\]|spec:' | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

echo "MODEL=$MODEL" | tee -a "$OUT"
echo "WL=$WL" | tee -a "$OUT"

# K=1 isolation of pos-1 acceptance (rate == pos-1 accept when mean_depth=1).
run "A strict   K=1"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1
run "B ratio0.5 K=1"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_ACCEPT=ratio:0.5
run "C ratio0.3 K=1"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_ACCEPT=ratio:0.3
run "D ratio0.1 K=1"  ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_ACCEPT=ratio:0.1

# Depth multiplier toward 20 tps: relaxed + batched K-verify.
run "E ratio0.3 K=4 batched" ECSIE_SPEC=on ECSIE_SPEC_DEPTH=4 ECSIE_SPEC_BATCHED=on ECSIE_SPEC_ACCEPT=ratio:0.3

echo "=== SUMMARY ($TAG) ===" | tee -a "$OUT"
grep -E '^########|spec:|warm_step_rate' "$OUT" | tee -a "$OUT"
