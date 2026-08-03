#!/usr/bin/env bash
# Spec A/B benchmark: warm-decode TPS with ECSIE_SPEC=off vs on on a single
# model + workload pair.  Used by Phase A (v1.0.166+) to verify that spec
# decode actually helps on the current rig before sinking implementation time
# into multi-needle PLD / tree drafts / adaptive K.
set -euo pipefail

MODEL="${MODEL:?MODEL must be set (path to .ecsie)}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/entropy_ab_short.json}"
TAG="${TAG:-spec_ab}"
RESULTS_DIR="${RESULTS_DIR:-$SRC_DIR/repos/ECSIE/benchmarks/results}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"
# v1.0.170: N repetitions per arm (each is its own cold process via the
# inherently cold-start bench binary), enables mean + std-dev computation
# to attenuate LearnedPolicy dynamic-depth variance.  Default 1 keeps the
# old behaviour; set NREPS=3 for statistical claims.
NREPS="${NREPS:-1}"

mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_rep() {
    local label="$1"; shift
    local rep="$1"; shift
    local spec_env="$1"; shift
    local rep_tag="${label}_r${rep}"
    echo "=== $rep_tag ($spec_env) ==="
    date '+%H:%M:%S'
    env $spec_env ECSIE_SPEC_DEBUG=1 \
        "$BENCH" --model "$MODEL" --workload "$WORKLOAD" \
                 --out-tps "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.csv" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_${rep_tag}_${STAMP}.stderr" | tail -25
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

# v1.0.167 Phase B: applied universally on top of whichever upstream selector
# produced the base depth (PolicyEngine static OR ECSIE_ENTROPY_SPEC bucket).
# The spec_on arm uses ECSIE_SPEC=on alone — matching the Phase A baseline,
# but with Phase B's adaptive-K adjustment engaged on the static path.
run_arm "spec_off" "ECSIE_SPEC=off"
run_arm "spec_on"  "ECSIE_SPEC=on"

echo "=== TPS comparison ==="
echo "stamp: $STAMP  tag: $TAG  nreps: $NREPS"
for label in spec_off spec_on; do
    # Aggregate across reps: extract warm_step_rate from each stderr.
    rates_csv=""
    for ((rep = 1; rep <= NREPS; ++rep)); do
        stderr="$RESULTS_DIR/${TAG}_${label}_r${rep}_${STAMP}.stderr"
        if [ -s "$stderr" ]; then
            # warm_step_rate=X.XX  appears in the [latency] step=0 line.
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
echo "=== Spec acceptance summary (from stderr) ==="
for label in spec_off spec_on; do
    for ((rep = 1; rep <= NREPS; ++rep)); do
        stderr="$RESULTS_DIR/${TAG}_${label}_r${rep}_${STAMP}.stderr"
        if [ -s "$stderr" ]; then
            accepts=$(grep -c '\[spec\] in=' "$stderr" || true)
            total_acc=$(grep '\[spec\] in=' "$stderr" | \
                        awk '{ for(i=1;i<=NF;i++) if($i ~ /^m=/) { sub("m=","",$i); s+=$i; n++ } }
                             END{ if(n>0) printf "mean_m=%.3f tokens_per_step=%.3f n=%d", s/n, s/n+1, n; else print "no spec lines" }')
            engine_summary=$(grep -E 'spec: steps=[0-9]' "$stderr" | tail -1)
            printf "  %-9s r%d  spec_lines=%s  %s\n" "$label" "$rep" "$accepts" "$total_acc"
            if [ -n "$engine_summary" ]; then
                printf "                engine: %s\n" "$engine_summary"
            fi
        fi
    done
done
