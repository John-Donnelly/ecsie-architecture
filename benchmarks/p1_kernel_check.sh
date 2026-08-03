#!/usr/bin/env bash
# P1.7 register-fused kernel: correctness (GPU vs new CPU kernel token match) +
# speed (all-CPU + hybrid N=3) vs the committed dequant-row baseline (~8.7/~11.5).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WLTOK=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_60.json
WLSPD=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_kernel_check.log
: > "$OUT"
echo "#### CORRECTNESS: GPU vs CPU(fused kernel) ####" | tee -a "$OUT"
ECSIE_SPEC=off ECSIE_DUMP_TOKENS=/tmp/k_gpu "$BIN" --model "$MODEL" --workload "$WLTOK" >/dev/null 2>&1
ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_DUMP_TOKENS=/tmp/k_cpu "$BIN" --model "$MODEL" --workload "$WLTOK" >/dev/null 2>&1
M=$(paste /tmp/k_gpu /tmp/k_cpu 2>/dev/null | awk '{if($1==$2)m++; else exit} END{print m+0}')
T=$(wc -l < /tmp/k_gpu 2>/dev/null)
echo "matching-prefix $M / $T tokens (GPU vs fused-CPU)" | tee -a "$OUT"
echo "#### SPEED (warm_step_rate = tok/s, spec off) ####" | tee -a "$OUT"
arm() { echo "## $1 ##" | tee -a "$OUT"; env "${@:2}" "$BIN" --model "$MODEL" --workload "$WLSPD" 2>&1 | grep -aE "^\[latency\]" | head -1 | tee -a "$OUT"; }
arm "all-CPU r1"      ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "all-CPU r2"      ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "hybrid N=3 r1"   ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
arm "hybrid N=3 r2"   ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
echo "DONE" | tee -a "$OUT"
