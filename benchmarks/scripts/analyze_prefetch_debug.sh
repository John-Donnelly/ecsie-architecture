#!/usr/bin/env bash
# Aggregate the per-layer prefetch[L=N] debug log into totals.
# Usage: analyze_prefetch_debug.sh <stderr_file>
set -uo pipefail  # no -e: greps with no matches mustn't kill the script
F="$1"

echo "=== prefetch debug aggregate (${F##*/}) ==="
echo "total prefetch calls: $(grep -c 'prefetch\[L=' "$F" || true)"
echo
echo "first 5 lines:"
grep 'prefetch\[L=' "$F" | head -5
echo
for field in predicted skip_vram skip_pool evicted promote create staged; do
    total=$(grep 'prefetch\[L=' "$F" | \
            awk -v fld="$field" '{
                for (i=1; i<=NF; i++) {
                    if (match($i, "^"fld"=[0-9]+")) {
                        sub(fld"=", "", $i); s += $i
                    }
                }
            } END { print s+0 }')
    printf "  %-12s = %s\n" "$field" "$total"
done
echo
echo "predict() empty count: $(grep 'predict\[' "$F" | grep -c 'returned 0 ' || echo 0)"
echo "predict() nonempty count: $(grep 'predict\[' "$F" | grep -cv 'returned 0 ' || echo 0)"
echo
echo "warm_step_rate + tiers:"
grep -E 'warm_step_rate|dispatch tiers|vram_prefetch staged' "$F" | tail -3
