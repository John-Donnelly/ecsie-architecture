#!/usr/bin/env bash
# Maximally-optimized spec A/B bench: enables batched verify path and
# forces depth=6 to bypass Phase B's kill switch.  Measures spec_off
# vs spec_on with the full optimization stack engaged.
#
# v1.0.195 prerequisites:
#   - Drafter rework so drafts contain real tokens (not all-EOS sentinels)
#   - ECSIE_SPEC_BATCHED default flipped to off; this script explicitly
#     opts back in.
set -euo pipefail

MODEL="${MODEL:?MODEL must be set (path to .ecsie)}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/spec_repetitive.json}"
TAG="${TAG:-spec_optimized}"
RESULTS_DIR="${RESULTS_DIR:-$SRC_DIR/repos/ECSIE/benchmarks/results}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"
NREPS="${NREPS:-3}"

mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_rep() {
    local label="$1"; shift
    local rep="$1"; shift
    local spec_env="$1"; shift
    local rep_tag="${label}_r${rep}"
    echo "=== $rep_tag ($spec_env) ==="
    date '+%H:%M:%S'
    env $spec_env ECSIE_SPEC_BATCHED=on ECSIE_SPEC_DEPTH=6 \
        "$BENCH" --model "$MODEL" --workload "$WORKLOAD" \
                 --out-tps "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.csv" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.stderr" | tail -15
    date '+%H:%M:%S'
    echo
}

run_arm() {
    local label="$1"; shift
    local spec_env="$1"; shift
    for ((rep = 1; rep <= NREPS; ++rep)); do
        run_rep "$label" "$rep" "$spec_env"
    done
}

run_arm "spec_off" "ECSIE_SPEC=off"
run_arm "spec_on"  "ECSIE_SPEC=on"

echo "=== TPS comparison ==="
echo "stamp: $STAMP  tag: $TAG  nreps: $NREPS  model: $MODEL"
for label in spec_off spec_on; do
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

echo
echo "=== Spec acceptance per run ==="
for ((r = 1; r <= NREPS; ++r)); do
    stderr="$RESULTS_DIR/${TAG}_spec_on_r${r}_${STAMP}.stderr"
    [ -s "$stderr" ] || continue
    acc=$(grep -oE 'spec: steps=[0-9]+ drafted=[0-9]+ accepted=[0-9]+ rate=[0-9.]+ mean_depth=[0-9.]+' "$stderr" | head -1)
    printf "  spec_on_r%d  %s\n" "$r" "$acc"
done
