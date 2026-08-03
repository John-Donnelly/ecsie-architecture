#!/usr/bin/env bash
# benchmarks/scripts/hybrid_topn_sweep.sh
#
# Hybrid Top-N ablation sweep.  For each N in a configurable list:
#   1. Emit a `.hquant` sidecar via `ecsie_hybrid_topn_quantize`.
#   2. Run a TPS bench against the model with that sidecar in place.
#   3. Extract steady-state warm_step_rate from the bench output.
#   4. Append a CSV row: (top_n, high_tier, low_tier, rank_mode,
#                          avg_bpw_estimate, warm_step_mean, mean_step_ms).
#
# Output CSV: benchmarks/results/hybrid_topn_sweep_<timestamp>.csv
#
# Runtime note: as of v1.0.91 the engine LOADS the .hquant sidecar but
# the per-(layer, expert) dequant-tier dispatch in `Expert::forward` is
# the documented milestone-B follow-up.  Until that lands, the TPS column
# of this sweep will be noise-equal across all N — what's measured is the
# stability of the surrounding pipeline, not the quant tradeoff itself.
# Once milestone-B lands, re-run with the same N grid to map the actual
# accuracy-vs-speed curve.
#
# Usage:
#   ./benchmarks/scripts/hybrid_topn_sweep.sh \
#       --model /path/to/model.gguf \
#       [--ns "0 4 8 12 16 20 24 32 40"] \
#       [--high-tier 4] [--low-tier 2] [--rank-mode edges] \
#       [--workload benchmarks/workloads/quick.json] \
#       [--ram-headroom-gb 6]

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL=""
NS="0 4 8 12 16 20 24 32 40"
HIGH_TIER=4         # Q4_K
LOW_TIER=2          # Ternary
RANK_MODE="edges"
WORKLOAD="benchmarks/workloads/quick.json"
RAM_HEADROOM_GB=6
SAMPLE_STRIDE=64

# ── CLI ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2"; shift 2 ;;
        --ns)               NS="$2"; shift 2 ;;
        --high-tier)        HIGH_TIER="$2"; shift 2 ;;
        --low-tier)         LOW_TIER="$2"; shift 2 ;;
        --rank-mode)        RANK_MODE="$2"; shift 2 ;;
        --workload)         WORKLOAD="$2"; shift 2 ;;
        --ram-headroom-gb)  RAM_HEADROOM_GB="$2"; shift 2 ;;
        --sample-stride)    SAMPLE_STRIDE="$2"; shift 2 ;;
        -h|--help)
            echo "Hybrid Top-N ablation sweep.  See header for usage."
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "${MODEL}" ]]; then
    echo "missing required --model PATH" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    echo "model not found: ${MODEL}" >&2
    exit 2
fi

QUANTIZE_BIN="bin/ecsie_hybrid_topn_quantize"
BENCH_BIN="bin/ecsie_bench_measure_tps"
[[ -x "${QUANTIZE_BIN}" ]] || { echo "missing ${QUANTIZE_BIN} (build first)"; exit 2; }
[[ -x "${BENCH_BIN}"    ]] || { echo "missing ${BENCH_BIN} (build first)"; exit 2; }

# Stash any pre-existing sidecar so we can restore it later.
SIDECAR_PATH="${MODEL}.hquant"
BACKUP=""
if [[ -f "${SIDECAR_PATH}" ]]; then
    BACKUP="${SIDECAR_PATH}.swp.$$"
    mv "${SIDECAR_PATH}" "${BACKUP}"
fi
restore() {
    rm -f "${SIDECAR_PATH}" 2>/dev/null || true
    if [[ -n "${BACKUP}" && -f "${BACKUP}" ]]; then
        mv "${BACKUP}" "${SIDECAR_PATH}"
    fi
}
trap restore EXIT INT TERM

TS=$(date +%Y%m%d_%H%M%S)
CSV="benchmarks/results/hybrid_topn_sweep_${TS}.csv"
mkdir -p "$(dirname "${CSV}")"
echo "top_n,high_tier,low_tier,rank_mode,avg_bpw_est,warm_step_mean,mean_step_ms,note" > "${CSV}"

echo "[sweep] model=${MODEL}"
echo "[sweep] N grid: ${NS}"
echo "[sweep] high=${HIGH_TIER}  low=${LOW_TIER}  rank=${RANK_MODE}"
echo "[sweep] workload=${WORKLOAD}  ram_headroom=${RAM_HEADROOM_GB} GB"
echo "[sweep] CSV: ${CSV}"

# Compute bpw helper.
bpw() {
    case "$1" in
        2)  echo "1.625" ;;
        3)  echo "3.4375" ;;
        4)  echo "4.5" ;;
        5)  echo "5.5" ;;
        6)  echo "6.5625" ;;
        8)  echo "8.5" ;;
        16) echo "16.0" ;;
        32) echo "32.0" ;;
        *)  echo "4.5" ;;
    esac
}

# Detect num_layers by writing a sidecar at N=0 and reading the header.
# (Or just assume 40 for Qwen3.6 — the tool already prints it.)
# For the bpw estimate, we'll just compute from N and assume 40 layers.

for N in ${NS}; do
    echo
    echo "[sweep] === N=${N} ==="
    STDERR="benchmarks/results/hybrid_sweep_N${N}.stderr"

    # Emit sidecar.
    "${QUANTIZE_BIN}" \
        --model "${MODEL}" \
        --top-n "${N}" \
        --out   "${SIDECAR_PATH}" \
        --high-tier "${HIGH_TIER}" \
        --low-tier  "${LOW_TIER}" \
        --rank-mode "${RANK_MODE}" \
        --sample-stride "${SAMPLE_STRIDE}" 2>&1 | tail -2

    # Bench.
    ECSIE_RAM_HEADROOM_GB="${RAM_HEADROOM_GB}" \
        "${BENCH_BIN}" \
        --model "${MODEL}" \
        --workload "${WORKLOAD}" \
        --out-tps /tmp/sweep_N${N}.csv \
        2>"${STDERR}" || true

    # Extract steady-state warm_step_rate (steps 2..end, ignore cold steps 0,1).
    WARM_MEAN=$(grep "warm_step_rate" "${STDERR}" 2>/dev/null \
        | tail -n +3 \
        | sed -E 's/.*warm_step_rate=([0-9.]+).*/\1/' \
        | awk 'BEGIN{s=0;n=0} {s+=$1;n+=1} END{ if (n>0) printf "%.3f", s/n; else printf "NaN" }')
    MEAN_STEP=$(grep "mean_step" "${STDERR}" 2>/dev/null \
        | tail -n +3 \
        | sed -E 's/.*mean_step=([0-9.]+) ms.*/\1/' \
        | awk 'BEGIN{s=0;n=0} {s+=$1;n+=1} END{ if (n>0) printf "%.1f", s/n; else printf "NaN" }')

    L_HIGH=$(bpw "${HIGH_TIER}")
    L_LOW=$(bpw "${LOW_TIER}")
    # avg = (N * high + (40-N) * low) / 40    (assumes 40 layers).
    AVG_BPW=$(awk -v n="${N}" -v h="${L_HIGH}" -v l="${L_LOW}" \
        'BEGIN { printf "%.3f", (n*h + (40-n)*l) / 40 }')

    echo "${N},${HIGH_TIER},${LOW_TIER},${RANK_MODE},${AVG_BPW},${WARM_MEAN},${MEAN_STEP},sidecar_loaded_only_no_tier_apply_yet" >> "${CSV}"
    echo "[sweep] N=${N}  avg_bpw≈${AVG_BPW}  warm_step_mean=${WARM_MEAN}  mean_step=${MEAN_STEP} ms"
done

echo
echo "[sweep] done.  CSV: ${CSV}"
