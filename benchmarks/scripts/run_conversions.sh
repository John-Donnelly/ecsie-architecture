#!/usr/bin/env bash
# Pass 2 A/B conversion runner: produces baseline (type-100) and entropy (type-101)
# .ecsie files from a shared GGUF input.  Logs to stdout.
set -euo pipefail

GGUF="${GGUF:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF}"
TOP_N="${TOP_N:-8}"
CONVERT="$SRC_DIR/repos/ECSIE/bin/ecsie_convert"

echo "=== baseline (type-100) ==="
date '+%H:%M:%S'
"$CONVERT" --in "$GGUF" \
           --out "$OUT_DIR/Qwen3-30B-A3B-Instruct-2507-baseline.ecsie" \
           --top-n "$TOP_N" \
           --verbose 2>&1 | tail -40
date '+%H:%M:%S'

echo
echo "=== entropy (type-101) ==="
date '+%H:%M:%S'
"$CONVERT" --in "$GGUF" \
           --out "$OUT_DIR/Qwen3-30B-A3B-Instruct-2507-entropy.ecsie" \
           --top-n "$TOP_N" \
           --entropy \
           --verbose 2>&1 | tail -40
date '+%H:%M:%S'

echo
echo "=== sizes ==="
ls -lh "$OUT_DIR"/*.ecsie
