#!/usr/bin/env bash
# ── Native-Linux validation bench (30-tps goal) ──────────────────────────────
# The 30-tps goal is blocked on THIS WSL2 box by memory bandwidth: a sequential
# DRAM read tops out at ~16 GB/s (membw.cpp), vs ~40-50 GB/s native on this
# Ryzen 5800X / DDR4-3200. The CPU-MoE path (Phase 1) is purely DRAM-bandwidth
# bound, so native Linux should ~2.5x it with ZERO code changes:
#     WSL2:   ~16 GB/s -> CPU-MoE ~74 ms/step -> ~17 tok/s user-facing
#     native: ~40 GB/s -> CPU-MoE ~29 ms/step -> ~30-33 tok/s user-facing
#
# This script PROVES or REFUTES that on native Linux. Run it on bare-metal Linux
# (NOT WSL). See NATIVE_LINUX_SETUP.md for build prerequisites.
#
# Usage:  MODEL=/path/to/Qwen3-30B-A3B-...-Q4_K_M.gguf  bash native_linux_bench.sh
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/bin/ecsie_bench_measure_tps"
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WL="$REPO/benchmarks/workloads/long_instruct_200.json"

echo "==================================================================="
echo " ECSIE native-Linux validation  ($(uname -sr))"
echo "==================================================================="
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "!! WARNING: this looks like WSL — run on BARE-METAL Linux for the real test."
fi
[ -x "$BIN" ] || { echo "!! $BIN missing — build first (see NATIVE_LINUX_SETUP.md)"; exit 1; }
[ -f "$MODEL" ] || { echo "!! model not found: $MODEL  (set MODEL=...)"; exit 1; }

# ── Step 1: memory-bandwidth probe (the decisive number) ─────────────────────
echo; echo "### STEP 1: DRAM bandwidth (decides everything) ###"
g++ -O3 -march=native -pthread "$REPO/benchmarks/membw.cpp" -o /tmp/membw 2>/dev/null \
    && /tmp/membw 8 || echo "membw build failed"
echo "   WSL2 reference: ~16 GB/s.  Native target: >=35 GB/s."
echo "   If this is ~16, the box itself is the limit (not WSL) — stop here."

# ── Step 2: warm the model into page cache ───────────────────────────────────
echo; echo "### STEP 2: warm model page cache ###"
cat "$MODEL" > /dev/null; grep -E "Cached" /proc/meminfo | head -1

# ── Step 3: engine benches (same configs measured on WSL) ────────────────────
echo; echo "### STEP 3: decode benches (warm_step_rate = raw tok/s; spec tps = user-facing) ###"
run() {
    echo "-- $1 --"
    env "${@:2}" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|attention=|moe_dispatch|spec:" | sed 's/^/   /'
}
run "raw all-CPU      (WSL was ~9.9)"  ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
run "raw hybrid N=3   (WSL was ~11.5)" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
run "spec lens (USER-FACING; WSL ~17)" ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 ECSIE_CPU_EXPERTS=on
echo
echo "INTERPRET: user-facing tok/s = (spec warm_step_rate) x (200/spec-steps)."
echo "  If membw ~40 GB/s AND moe_dispatch ~halved vs WSL's ~1.56 ms/layer,"
echo "  the spec config should land ~28-33 tok/s -> 30-tps goal MET on native HW."
echo "DONE"
