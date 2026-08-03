#!/usr/bin/env bash
# M3 validation: device-draft must reproduce host-draft accept (~0.818) — proves
# no cross-stream race / eps match — and report the tps delta.
set -u
B=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
M="${MODEL:-$HOME/ecsie_models/q4.gguf}"
W=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_200.json
LENS="ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46 ECSIE_CPU_EXPERTS=on"
echo "## HOST draft (baseline) ##"
env $LENS "$B" --model "$M" --workload "$W" 2>&1 | grep -aE "^\[latency\]|spec:"
echo "## M3 DEVICE draft (ECSIE_LENS_DEVICE_DRAFT=1) ##"
env $LENS ECSIE_LENS_DEVICE_DRAFT=1 "$B" --model "$M" --workload "$W" 2>&1 | grep -aE "^\[latency\]|spec:"
echo DONE
