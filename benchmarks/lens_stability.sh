#!/usr/bin/env bash
# v1.0.214 — Phase-0 GO-gate stability: lens self-draft (strict, K=1) pos-1
# acceptance across the novel-workload suite + the gate workload. Each model
# format × each workload is a fresh process. Gate: pos-1 accept >= 0.65 on all.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
TAG="${TAG:-q4gguf}"
WLDIR=$SRC_DIR/repos/ECSIE/benchmarks/workloads
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/lens_stability_${TAG}.log
: > "$OUT"

WLS="long_instruct novel_code novel_math novel_prose novel_chat novel_technical_qa novel_structured"

for w in $WLS; do
    WL="$WLDIR/$w.json"
    [ -f "$WL" ] || { echo "MISSING $WL" | tee -a "$OUT"; continue; }
    echo "######## $w ########" | tee -a "$OUT"
    env ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 \
        "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "spec:" | tee -a "$OUT"
done

echo "=== STABILITY SUMMARY ($TAG, lens strict K=1) ===" | tee -a "$OUT"
paste -d' ' <(grep -aE "^########" "$OUT" | sed 's/#//g') <(grep -aE "rate=" "$OUT" | grep -oE "rate=[0-9.]+") 2>/dev/null | tee -a "$OUT"
