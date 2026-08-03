#!/usr/bin/env bash
# Longer-output 3-arm equivalence check.  Uses --tokens 256 (4× the default
# 64) to stress losslessness over longer decode sequences.  Each arm runs in
# a fresh process so warm-cache numerics drift can't cross-contaminate.
set -u
MODEL="${1:-$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-hybrid-n8h8.ecsie}"
TOKENS="${2:-256}"
BIN=$SRC_DIR/repos/ECSIE/bin/ecsie_test_spec_equivalence

run() {
    local label="$1"; shift
    local spec_arg="$1"; shift
    env "$@" "$BIN" --model "$MODEL" --spec "$spec_arg" --tokens "$TOKENS" 2>/dev/null \
        | grep '^TOKENS:' | head -n1 | sed "s/^TOKENS:/$label/"
}

echo "=== $MODEL  ($TOKENS tokens, 3-arm) ==="
A=$(run "OFF       :" off)
B=$(run "ON_BATCHED:" on ECSIE_SPEC_BATCHED=1)
C=$(run "ON_SEQ    :" on ECSIE_SPEC_BATCHED=0)

# Print labels with their tokens elided to just first 8 and last 4 for clarity
preview() {
    local line="$1"
    local label_and_first8
    label_and_first8=$(echo "$line" | awk '{for(i=1;i<=9;++i) printf "%s ", $i; print ""}')
    local last4
    last4=$(echo "$line" | awk '{for(i=NF-3;i<=NF;++i) printf "%s ", $i; print ""}')
    echo "${label_and_first8}... ${last4}"
}
preview "$A"
preview "$B"
preview "$C"

# Quick equality verdict over the FULL line (not just preview).
a_tail=$(echo "$A" | sed 's/^[^:]*: *//')
b_tail=$(echo "$B" | sed 's/^[^:]*: *//')
c_tail=$(echo "$C" | sed 's/^[^:]*: *//')
echo
if [ "$a_tail" = "$b_tail" ] && [ "$b_tail" = "$c_tail" ]; then
    echo "VERDICT: PASS  (OFF == ON_BATCHED == ON_SEQ across $TOKENS tokens)"
    exit 0
else
    echo "VERDICT: FAIL"
    [ "$a_tail" != "$b_tail" ] && echo "  OFF != ON_BATCHED"
    [ "$a_tail" != "$c_tail" ] && echo "  OFF != ON_SEQ"
    [ "$b_tail" != "$c_tail" ] && echo "  ON_BATCHED != ON_SEQ"
    exit 1
fi
