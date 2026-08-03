#!/usr/bin/env bash
# Pass 2 A/B benchmark: warm-decode TPS on baseline (type-100) vs entropy
# (type-101) .ecsie variants of the same source GGUF.
set -euo pipefail

OUT_DIR="${OUT_DIR:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_decode.json}"
RESULTS_DIR="${RESULTS_DIR:-$SRC_DIR/repos/ECSIE/benchmarks/results}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"

BASELINE="$OUT_DIR/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie"
ENTROPY="$OUT_DIR/Qwen3-30B-A3B-Instruct-2507-entropy.ecsie"

mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== baseline (type-100) ==="
date '+%H:%M:%S'
"$BENCH" --model "$BASELINE" --workload "$WORKLOAD" \
         --out-tps "$RESULTS_DIR/entropy_ab_baseline_${STAMP}.csv" \
         2>&1 | tee "$RESULTS_DIR/entropy_ab_baseline_${STAMP}.stderr" | tail -25
date '+%H:%M:%S'

echo
echo "=== entropy (type-101) ==="
date '+%H:%M:%S'
"$BENCH" --model "$ENTROPY" --workload "$WORKLOAD" \
         --out-tps "$RESULTS_DIR/entropy_ab_entropy_${STAMP}.csv" \
         2>&1 | tee "$RESULTS_DIR/entropy_ab_entropy_${STAMP}.stderr" | tail -25
date '+%H:%M:%S'

echo
echo "=== TPS comparison ==="
echo "stamp: $STAMP"
for tag in baseline entropy; do
    csv="$RESULTS_DIR/entropy_ab_${tag}_${STAMP}.csv"
    if [ -s "$csv" ]; then
        # CSV: step,timestamp_ms,tokens,tps  — average tps across rows
        avg=$(awk -F, 'NR>1 && $4!="" {sum+=$4; n++} END { if (n>0) printf "%.2f", sum/n; else print "n/a" }' "$csv")
        last=$(awk -F, 'END{print $4}' "$csv")
        rows=$(awk -F, 'END{print NR-1}' "$csv")
        printf "  %-9s  rows=%-3s  avg_tps=%-8s  last_tps=%-8s  (file: %s)\n" \
            "$tag" "$rows" "$avg" "$last" "$(basename "$csv")"
    else
        printf "  %-9s  (no rows)\n" "$tag"
    fi
done
