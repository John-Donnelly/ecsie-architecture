#!/usr/bin/env bash
# Phase 1 correctness: GPU-expert vs CPU-expert generated token IDs (greedy,
# spec off). Same dequant math => tokens should match (modulo fp-order ties).
set -u
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_bench_measure_tps
MODEL=$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
WL=$SRC_DIR/repos/ECSIE/benchmarks/workloads/long_instruct_60.json

export ECSIE_SPEC=off
ECSIE_DUMP_TOKENS=/tmp/gpu_tok "$BIN" --model "$MODEL" --workload "$WL" > /tmp/gpu_gen.log 2>&1
ECSIE_CPU_EXPERTS=on ECSIE_DUMP_TOKENS=/tmp/cpu_tok "$BIN" --model "$MODEL" --workload "$WL" > /tmp/cpu_gen2.log 2>&1

echo "=== GPU first 20 token ids ==="; head -20 /tmp/gpu_tok 2>/dev/null | tr '\n' ' '; echo
echo "=== CPU first 20 token ids ==="; head -20 /tmp/cpu_tok 2>/dev/null | tr '\n' ' '; echo
echo "=== matching-prefix length (of $(wc -l < /tmp/gpu_tok 2>/dev/null) gpu / $(wc -l < /tmp/cpu_tok 2>/dev/null) cpu) ==="
paste /tmp/gpu_tok /tmp/cpu_tok 2>/dev/null | awk '{if($1==$2)m++; else exit} END{print m}'
echo "=== CPU text (if detokenized) ==="; head -c 400 /tmp/cpu_tok.text 2>/dev/null; echo
