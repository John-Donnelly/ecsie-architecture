#!/usr/bin/env bash
# All-fast-paths-off forward-equivalence test.  Forces pure legacy
# per-expert serial dispatch (no fused fast paths, no GPU router).
# Used during the v1.0.179-189 diagnostic chain to confirm the drift
# survives all dispatch tiers — which it did until v1.0.190's
# residency-sort fix.  Kept as a regression check.
set -uo pipefail
MODEL="${1:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie}"
K="${2:-0}"
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_test_forward_equivalence
echo "=== $MODEL (K=$K, all fast paths off) ==="
ECSIE_PHASE_F=off ECSIE_PHASE_H=off ECSIE_PHASE_J=off ECSIE_PHASE_I=off \
ECSIE_PHASE_K=off ECSIE_FUSED_MULTI_Q4K=0 ECSIE_FUSED_MULTI_TERNARY=0 \
    "$BIN" --model "$MODEL" --k "$K" 2>&1 | tail -28
