#!/usr/bin/env bash
set -uo pipefail
MODEL="${1:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie}"
K="${2:-0}"
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_test_forward_equivalence
echo "=== $MODEL (K=$K, FUSED_MULTI_TERNARY=0 FUSED_MULTI_Q4K=0) ==="
ECSIE_FUSED_MULTI_TERNARY=0 ECSIE_FUSED_MULTI_Q4K=0 \
    "$BIN" --model "$MODEL" --k "$K" 2>&1 | tail -28
