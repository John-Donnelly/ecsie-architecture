#!/usr/bin/env bash
# P1.8 integer (q8) CPU MoE dot: correctness + speed.
# Lossy (q8 activation) so token-match vs fp32/GPU may not be 60/60 — check how
# close + coherence, and the speed delta (fp32-fused baseline ~9.9).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL="${MODEL:-$HOME/ecsie_models/q4.gguf}"
WLTOK=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_60.json
WLSPD=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/p1_intdot_check.log
: > "$OUT"
echo "#### CORRECTNESS ####" | tee -a "$OUT"
ECSIE_SPEC=off ECSIE_DUMP_TOKENS=/tmp/i_gpu "$BIN" --model "$MODEL" --workload "$WLTOK" >/dev/null 2>&1
ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_DUMP_TOKENS=/tmp/i_fp32 "$BIN" --model "$MODEL" --workload "$WLTOK" >/dev/null 2>&1
ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_CPU_INT_DOT=on ECSIE_DUMP_TOKENS=/tmp/i_int "$BIN" --model "$MODEL" --workload "$WLTOK" >/dev/null 2>&1
T=$(wc -l < /tmp/i_gpu 2>/dev/null)
MG=$(paste /tmp/i_gpu  /tmp/i_int 2>/dev/null | awk '{if($1==$2)m++; else exit} END{print m+0}')
MF=$(paste /tmp/i_fp32 /tmp/i_int 2>/dev/null | awk '{if($1==$2)m++; else exit} END{print m+0}')
echo "int vs GPU : $MG / $T   |   int vs fp32-fused : $MF / $T" | tee -a "$OUT"
echo "first 16 GPU: $(head -16 /tmp/i_gpu | tr '\n' ' ')" | tee -a "$OUT"
echo "first 16 int: $(head -16 /tmp/i_int | tr '\n' ' ')" | tee -a "$OUT"
echo "#### SPEED (warm_step_rate = tok/s, spec off) ####" | tee -a "$OUT"
arm(){ echo "## $1 ##" | tee -a "$OUT"; env "${@:2}" "$BIN" --model "$MODEL" --workload "$WLSPD" 2>&1 | grep -aE "^\[latency\]" | head -1 | tee -a "$OUT"; }
arm "all-CPU fp32"      ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on
arm "all-CPU int"       ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_CPU_INT_DOT=on
arm "all-CPU int r2"    ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_CPU_INT_DOT=on
arm "hybrid N=3 int"    ECSIE_SPEC=off ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3 ECSIE_CPU_INT_DOT=on
echo "DONE" | tee -a "$OUT"
