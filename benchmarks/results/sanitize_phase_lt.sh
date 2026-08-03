#!/usr/bin/env bash
# compute-sanitizer trace of Phase LT to find why the captured ternary graph
# produces wrong output.  Use --tool=memcheck for OOB reads/writes; if that's
# clean try --tool=synccheck and --tool=racecheck.
set -u
cd $SRC_DIR/repos/ECSIE
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
BIN=./bin/ecsie_bench_measure_tps
M=$HOME/models/qwen3-30b/Qwen3-30B-A3B.ecsie
W=$SRC_DIR/repos/ECSIE/benchmarks/results/wl_bench80.json
SANI=/usr/local/cuda-13.2/bin/compute-sanitizer
OUT=$SRC_DIR/repos/ECSIE/benchmarks/results/sani_phase_lt

# Use a very short prompt so we don't run for hours.  Limit decode tokens to
# ~5 via the workload (assumed; if not, we'll trim manually below).
ECSIE_PHASE_LT=on \
"$SANI" --tool=memcheck \
  --launch-skip 100 --launch-count 30 \
  --print-limit 30 \
  "$BIN" --model "$M" --workload "$W" --out-tps /dev/null \
  > "${OUT}.stdout" 2> "${OUT}.stderr"

echo "rc=$?"
echo "--- sanitizer report (stdout) ---"
head -150 "${OUT}.stdout"
echo "--- bench stderr tail ---"
tail -30 "${OUT}.stderr"
