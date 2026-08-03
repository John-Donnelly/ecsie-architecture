#!/usr/bin/env bash
# Phase 1 — CPU MoE expert path vs GPU (PCIe-stream) path, Q4_K_M.
# Spec OFF to isolate raw decode rate (warm_step_rate = tok/s at 1 tok/step).
# Phase 1 GO gate: >= 60 tok/s (YELLOW 30-60, STOP < 30).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
WL="${WL:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json}"
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/phase1_cpu_experts.log
: > "$OUT"

arm() {
    local label="$1"; shift
    echo "######## $label ########" | tee -a "$OUT"
    env "$@" "$BIN" --model "$MODEL" --workload "$WL" 2>&1 \
        | grep -aE "^\[latency\]|spec:|CpuMoeRunner|tier distribution" | head -4 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

arm "GPU experts (baseline)"  ECSIE_SPEC=off
arm "CPU experts"             ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
echo "DONE"
