#!/usr/bin/env bash
# Three-arm equivalence comparison: off, on (batched), on (sequential).
# Each arm runs in a FRESH PROCESS to defeat warm-cache numerics drift.
set -u
MODEL="${1:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie}"
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_test_spec_equivalence

run() {
    local label="$1"; shift
    local spec_arg="$1"; shift
    env "$@" "$BIN" --model "$MODEL" --spec "$spec_arg" 2>/dev/null \
        | grep '^TOKENS:' | head -n1 | sed "s/^TOKENS:/$label/"
}

# --spec sets ECSIE_SPEC inside the binary AFTER its env reset.
# ECSIE_SPEC_BATCHED is read via getenv() in the engine ctor lambda, so it
# survives the spec reset.
run "OFF       :" off
run "ON_BATCHED:" on ECSIE_SPEC_BATCHED=1
run "ON_SEQ    :" on ECSIE_SPEC_BATCHED=0
