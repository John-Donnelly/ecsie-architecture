#!/usr/bin/env bash
# ECSIE — benchmarks/spec_equivalence_check.sh
#
# Speculative-decoding output-equivalence check.
#
# Speculative decoding must be LOSSLESS: the tokens it accepts must be
# byte-identical to plain greedy decoding.  This is awkward to verify in a
# single process because the tiered streaming architecture keeps a warm
# expert cache that PERSISTS across generate() calls — tier residency shifts
# numerics, so two generate() calls in one process are not guaranteed to
# agree even with identical settings.  That cross-call non-determinism is a
# real, inherent engine property, not a bug.
#
# This script sidesteps it by running each arm in a FRESH PROCESS: each
# `ecsie_test_spec_equivalence` invocation starts from an identical
# empty-cache state and is internally deterministic.  It runs the binary
# twice — once `--spec off`, once `--spec on` — captures each `TOKENS:` line,
# and diffs them.
#
# Usage:
#   benchmarks/spec_equivalence_check.sh <model.gguf>
#
# Output:
#   EQUIVALENCE: PASS
#   EQUIVALENCE: FAIL at index N (off=X on=Y)
#
# Exit code: 0 on PASS, 1 on FAIL or any setup error.
#
# NOTE (v1.0.114): the off-vs-on comparison is EXPECTED to FAIL right now.
# Speculative decoding is known-broken — on a rejected draft the verifier
# emits the *rejected* draft token, corrupting output — so it is disabled by
# default and is opt-in only.  `--spec on` deliberately exercises that broken
# path, so this gate FAILS until the batched-verification rewrite lands
# (see SPEC-DECODE-PLAN.md).  A FAIL here is therefore the correct,
# documented state, not a regression.

set -u

MODEL="${1:-}"
if [ -z "${MODEL}" ]; then
    echo "usage: $0 <model.gguf>" >&2
    exit 1
fi
if [ ! -f "${MODEL}" ]; then
    echo "error: model not found: ${MODEL}" >&2
    exit 1
fi

# Locate the binary relative to this script (build_linux is the default
# out-of-tree build dir on the bench host).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN=""
for cand in \
    "${REPO_ROOT}/bin/ecsie_test_spec_equivalence" \
    "${REPO_ROOT}/build_linux/ecsie_test_spec_equivalence" \
    "${REPO_ROOT}/build/ecsie_test_spec_equivalence" \
    "${REPO_ROOT}/build_linux/benchmarks/ecsie_test_spec_equivalence" \
    "${REPO_ROOT}/build/benchmarks/ecsie_test_spec_equivalence"
do
    if [ -x "${cand}" ]; then BIN="${cand}"; break; fi
done
if [ -z "${BIN}" ]; then
    echo "error: ecsie_test_spec_equivalence binary not found; build it first" >&2
    exit 1
fi

# Run one arm in its own process; echo only the TOKENS: line to stdout.
run_arm() {
    local spec="$1"
    "${BIN}" --model "${MODEL}" --spec "${spec}" 2>/dev/null \
        | grep '^TOKENS:' | head -n1
}

# Printed whenever the off-vs-on comparison FAILs.  A FAIL is the expected,
# documented state until speculative decoding is fixed — make that explicit so
# nobody mistakes it for a fresh regression.
print_known_broken_note() {
    echo "NOTE: speculative decoding is known-broken (emits rejected draft" \
         "tokens) and is disabled by default; this gate will PASS once the" \
         "batched-verification rewrite lands (see SPEC-DECODE-PLAN.md)."
}

echo "[spec_equiv_check] arm 1/2: --spec off (fresh process)" >&2
OFF_LINE="$(run_arm off)"
echo "[spec_equiv_check] arm 2/2: --spec on  (fresh process)" >&2
ON_LINE="$(run_arm on)"

if [ -z "${OFF_LINE}" ]; then
    echo "error: --spec off arm produced no TOKENS: line" >&2
    exit 1
fi
if [ -z "${ON_LINE}" ]; then
    echo "error: --spec on arm produced no TOKENS: line" >&2
    exit 1
fi

# Strip the "TOKENS:" prefix → space-separated token IDs.
read -r -a OFF_TOK <<< "${OFF_LINE#TOKENS:}"
read -r -a ON_TOK  <<< "${ON_LINE#TOKENS:}"

OFF_N=${#OFF_TOK[@]}
ON_N=${#ON_TOK[@]}
N=${OFF_N}
[ "${ON_N}" -lt "${N}" ] && N=${ON_N}

i=0
while [ "${i}" -lt "${N}" ]; do
    if [ "${OFF_TOK[$i]}" != "${ON_TOK[$i]}" ]; then
        echo "EQUIVALENCE: FAIL at index ${i} (off=${OFF_TOK[$i]} on=${ON_TOK[$i]})"
        print_known_broken_note
        exit 1
    fi
    i=$((i + 1))
done

if [ "${OFF_N}" -ne "${ON_N}" ]; then
    echo "EQUIVALENCE: FAIL at index ${N} (length mismatch: off=${OFF_N} on=${ON_N})"
    print_known_broken_note
    exit 1
fi

echo "EQUIVALENCE: PASS"
exit 0
