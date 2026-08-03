#!/usr/bin/env bash
# v1.0.162 Nsight Systems profile pass: warm-decode breakdown for 30B + 35B.
# Captures CUDA API + kernel + memcpy + sync events with NVTX ranges if any.
# Outputs .nsys-rep into benchmarks/results/profile_v1.0.162/.
set -uo pipefail

OUT="${OUT:-$SRC_DIR/repos/ECSIE/benchmarks/results/profile_v1.0.162}"
WORKLOAD="${WORKLOAD:-$SRC_DIR/repos/ECSIE/benchmarks/workloads/profile_ctx2k.json}"
BENCH="$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps"
NSYS=/usr/local/cuda-13.2/bin/nsys

mkdir -p "$OUT"

declare -A MODELS=(
    [30b_baseline]="$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie"
    [30b_entropy]="$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-entropy.ecsie"
    [35b_hybrid]="$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie"
)

# Run each in sequence — running in parallel would OOM the 8 GB GPU.
for tag in 30b_baseline 30b_entropy 35b_hybrid; do
    model="${MODELS[$tag]}"
    rep="$OUT/profile_${tag}"
    echo "=== $tag :: $(basename "$model") ==="
    date '+%H:%M:%S'

    # --trace=cuda,nvtx captures GPU API calls + any nvtx ranges.
    # --cuda-memory-usage=true marks memcpy/alloc events with size.
    # -f true overwrites prior rep file.
    # Use --duration to bound the capture; the bench self-terminates at 200 tokens
    # which is roughly 60-90 s on 30B and 200-400 s on 35B; let it run to natural exit.
    "$NSYS" profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        -o "$rep" \
        -f true \
        --stats=false \
        "$BENCH" --model "$model" --workload "$WORKLOAD" \
                 --out-tps "$OUT/tps_${tag}.csv" \
        > "$OUT/stderr_${tag}.txt" 2>&1
    date '+%H:%M:%S'
    echo
done

echo "All nsys runs complete."
ls -lh "$OUT"
