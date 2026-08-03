#!/usr/bin/env bash
# Phase 1 residency setup (WSL-specific).  The Q4 model lives on /mnt/a (drvfs,
# the Windows drive), which does NOT populate the Linux page cache — every CPU
# expert read re-faults at ~160 MB/s instead of DDR4 ~51 GB/s.  Two steps make
# the model DRAM-resident so the CPU expert path is read-fast:
#
#  1. Raise the WSL2 RAM cap above the model size.  In %USERPROFILE%\.wslconfig:
#         [wsl2]
#         memory=24GB
#         processors=16
#         swap=8GB
#     then `wsl --shutdown` (host 31.9 GB; model 18 GB; 24 GB leaves ~8 GB Win).
#
#  2. Copy the model from drvfs to the WSL ext4 filesystem (this script).  ext4
#     mmap pages cache normally; `cat model >/dev/null` then shows Cached ~= 18 GB.
#
# Bench then points --model at the ext4 copy.  Measured: CPU experts 5.76 (drvfs)
# -> 7.88 tok/s (ext4-resident), p99 137 ms vs GPU PCIe 1467 ms.
set -u
SRC="${1:-$MODELS_DIR/Models/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
DST="${2:-$HOME/ecsie_models/q4.gguf}"
mkdir -p "$(dirname "$DST")"
echo "[setup] free RAM:"; free -g | head -2
echo "[setup] copying $SRC -> $DST (one-time)"
time cp "$SRC" "$DST"
echo "[setup] warming page cache"; cat "$DST" > /dev/null
echo "[setup] cached:"; grep Cached /proc/meminfo
echo "[setup] done — bench with: MODEL=$DST"
