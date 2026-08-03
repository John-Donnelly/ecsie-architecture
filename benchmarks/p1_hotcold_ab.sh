#!/usr/bin/env bash
# Phase 1 P1.4 — hot/cold GPU+CPU overlap A/B.
#   all-CPU  : every routed expert on the CPU (the proven +97% path)
#   hybrid N : N hottest experts/layer on the GPU (resident), cold tail on the
#              CPU, run concurrently; combined at the MoE exit.
# Correctness: hybrid reorders the expert sum (GPU dequant + separate partial),
# so it won't byte-match all-CPU, but it must stay coherent + track closely.
# Speed: warm_step_rate (spec off => tok/s).  Bench self-warms (WSL balloon
# reclaims idle cache), so no pre-warm needed.
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WLSPEED=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
WLTOK=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_60.json

echo "######## CORRECTNESS (token match vs all-CPU) ########"
ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_DUMP_TOKENS=/tmp/hc_cpu \
    "$BIN" --model "$MODEL" --workload "$WLTOK" > /tmp/hc_cpu.log 2>&1
for N in 2 3; do
    ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=$N ECSIE_DUMP_TOKENS=/tmp/hc_h$N \
        "$BIN" --model "$MODEL" --workload "$WLTOK" > /tmp/hc_h$N.log 2>&1
    M=$(paste /tmp/hc_cpu /tmp/hc_h$N 2>/dev/null | awk '{if($1==$2)m++; else exit} END{print m+0}')
    T=$(wc -l < /tmp/hc_cpu)
    echo "HOTCOLD=$N: matching-prefix $M / $T tokens"
done
echo "first 24 ids all-CPU : $(head -24 /tmp/hc_cpu | tr '\n' ' ')"
echo "first 24 ids HOTCOLD2: $(head -24 /tmp/hc_h2 | tr '\n' ' ')"

echo "######## SPEED (warm_step_rate = tok/s, spec off) ########"
arm() {
    echo "## $1 ##"
    env "${@:2}" "$BIN" --model "$MODEL" --workload "$WLSPEED" 2>&1 \
        | grep -aE "^\[latency\]" | head -1
}
arm "all-CPU"        ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "hybrid HOTCOLD=1" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=1
arm "hybrid HOTCOLD=2" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=2
arm "hybrid HOTCOLD=3" ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3
echo "DONE"
