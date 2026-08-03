#!/usr/bin/env bash
# v1.0.175 35B-hybrid entropy A/B bench.  NREPS-rep multi-run on both
# the original type-100 hybrid-n8h8 model and the v1.0.175 type-101
# entropy conversion.
set -uo pipefail

NREPS="${NREPS:-3}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/spec_repetitive.json}"
RESULTS_DIR="${RESULTS_DIR:-$SRC_DIR/repos/ECSIE/benchmarks/results}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"
TAG="${TAG:-entropy35b_ab}"

BASELINE="$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie"
ENTROPY="$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8-entropy.ecsie"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

run_rep() {
    local label="$1"; shift
    local model="$1"; shift
    local rep="$1"; shift
    local rep_tag="${label}_r${rep}"
    echo "=== $rep_tag ==="
    date '+%H:%M:%S'
    "$BENCH" --model "$model" --workload "$WORKLOAD" \
             --out-tps "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.csv" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.stderr" | tail -10
    date '+%H:%M:%S'
    echo
}

for ((r = 1; r <= NREPS; ++r)); do
    run_rep "baseline" "$BASELINE" "$r"
done
for ((r = 1; r <= NREPS; ++r)); do
    run_rep "entropy" "$ENTROPY" "$r"
done

echo "=== TPS comparison ==="
echo "stamp: $STAMP  nreps: $NREPS"
for label in baseline entropy; do
    rates_csv=""
    for ((r = 1; r <= NREPS; ++r)); do
        stderr="$RESULTS_DIR/${TAG}_${label}_r${r}_${STAMP}.stderr"
        if [ -s "$stderr" ]; then
            wsr=$(grep -oE 'warm_step_rate=[0-9.]+' "$stderr" | head -1 | sed 's/.*=//')
            [ -n "$wsr" ] && rates_csv="${rates_csv}${rates_csv:+,}$wsr"
        fi
    done
    if [ -n "$rates_csv" ]; then
        stats=$(echo "$rates_csv" | awk -F, '
            { for (i=1;i<=NF;i++) { s+=$i; n++; vs[i]=$i } }
            END {
                m = s/n; ss=0
                for (i in vs) ss += (vs[i]-m)*(vs[i]-m)
                sd = (n>1) ? sqrt(ss/(n-1)) : 0
                printf "mean=%.3f sd=%.3f n=%d", m, sd, n
            }')
        printf "  %-9s  warm_step_rate: %s  rates=[%s]\n" "$label" "$stats" "$rates_csv"
    fi
done
