#!/usr/bin/env bash
set -uo pipefail
MODEL="${1:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie}"
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_test_ssm_roundtrip
N_WARM="${2:-8}"
"$BIN" --model "$MODEL" --n-warm "$N_WARM" 2>&1 | tail -40
