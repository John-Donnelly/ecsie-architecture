#!/usr/bin/env bash
# Prefetch A/B benchmark: warm-decode TPS with ECSIE_VRAM_PREFETCH=off vs on.
# Uses the same NREPS multi-run pattern as run_spec_ab_bench.sh.
#
# v1.0.124 introduced overlapped VRAM-staging expert prefetch but it benched
# neutral on Qwen3-30B-A3B (cache effectively 100 % cached after warm-up so
# nothing left to stage).  Re-measure on Qwen3.6-35B-A3B (transfer-bound,
# 43 % H2D of GPU activity per v1.0.162 profile) where prefetch should
# actually have something to do.
set -euo pipefail

MODEL="${MODEL:?MODEL must be set (path to .ecsie)}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/spec_repetitive.json}"
TAG="${TAG:-prefetch_ab}"
RESULTS_DIR="${RESULTS_DIR:-$SRC_DIR/repos/ECSIE/benchmarks/results}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"
NREPS="${NREPS:-1}"

mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_rep() {
    local label="$1"; shift
    local rep="$1"; shift
    local env_args="$1"; shift
    local rep_tag="${label}_r${rep}"
    echo "=== $rep_tag ($env_args) ==="
    date '+%H:%M:%S'
    env $env_args \
        "$BENCH" --model "$MODEL" --workload "$WORKLOAD" \
                 --out-tps "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.csv" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.stderr" | tail -25
    date '+%H:%M:%S'
    echo
}

run_arm() {
    local label="$1"; shift
    local env_args="$1"; shift
    for ((rep = 1; rep <= NREPS; ++rep)); do
        run_rep "$label" "$rep" "$env_args"
    done
}

run_arm "pref_off" ""
run_arm "pref_on"  "ECSIE_VRAM_PREFETCH=on"

echo "=== TPS comparison ==="
echo "stamp: $STAMP  tag: $TAG  nreps: $NREPS"
for label in pref_off pref_on; do
    rates_csv=""
    for ((rep = 1; rep <= NREPS; ++rep)); do
        stderr="$RESULTS_DIR/${TAG}_${label}_r${rep}_${STAMP}.stderr"
        if [ -s "$stderr" ]; then
            wsr=$(grep -oE 'warm_step_rate=[0-9.]+' "$stderr" | head -1 | sed 's/.*=//')
            if [ -n "$wsr" ]; then
                rates_csv="${rates_csv}${rates_csv:+,}$wsr"
            fi
        fi
    done
    if [ -n "$rates_csv" ]; then
        stats=$(echo "$rates_csv" | awk -F, '
            { for (i=1;i<=NF;i++) { s+=$i; n++; if ($i>max||max==0) max=$i; if ($i<min||min==0) min=$i; vs[i]=$i } }
            END {
                if (n==0) { print "n/a"; exit }
                m = s/n
                ss = 0
                for (i in vs) ss += (vs[i]-m)*(vs[i]-m)
                sd = (n>1) ? sqrt(ss/(n-1)) : 0
                printf "mean=%.3f sd=%.3f min=%.3f max=%.3f n=%d", m, sd, min, max, n
            }')
        printf "  %-9s  warm_step_rate: %s  rates=[%s]\n" "$label" "$stats" "$rates_csv"
    else
        printf "  %-9s  (no rows)\n" "$label"
    fi
done

echo
echo "=== Prefetch staging telemetry (from stderr) ==="
for label in pref_off pref_on; do
    for ((rep = 1; rep <= NREPS; ++rep)); do
        stderr="$RESULTS_DIR/${TAG}_${label}_r${rep}_${STAMP}.stderr"
        if [ -s "$stderr" ]; then
            staged=$(grep -oE 'vram_prefetch staged [0-9]+' "$stderr" | tail -1 | grep -oE '[0-9]+' || true)
            staged="${staged:-0}"
            printf "  %-9s r%d  vram_prefetch_staged=%s\n" "$label" "$rep" "$staged"
        fi
    done
done
