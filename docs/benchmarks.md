# Benchmark Reproducibility Guide

## Overview

All benchmark results reported in the paper are produced by `benchmarks/run_all.sh`.
This guide describes how to reproduce them on a clean system.

## Requirements

- NVIDIA GPU with 8 GB+ VRAM
- CUDA 12+ (CPU-only path available for routing/entropy benchmarks without GPU)
- Model: Qwen 3.6 35B A3B — `Qwen3.6-35B-A3B-Q4_K_M.gguf`
- Built benchmark binaries (see Quick Start)

## Quick Start

```bash
# 1. Build (with benchmarks enabled)
cmake --preset ci -B build/ci -DECSIE_BUILD_BENCHMARKS=ON
cmake --build build/ci -j$(nproc)

# 2. Run full benchmark suite (auto-detects model at default path)
MODEL_PATH=$MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  ./benchmarks/run_all.sh

# Or with explicit flags:
./benchmarks/run_all.sh \
  --model $MODELS_DIR/Models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --build-dir build/ci

# 3. Results will appear in benchmarks/results/
```

## Workload Profiles

| File | Description |
|------|-------------|
| `workloads/stable.json` | Low-variance prompts, consistent output length |
| `workloads/mixed.json` | Mixed prompt lengths and output distributions |
| `workloads/high_entropy.json` | High routing entropy, adversarial batch composition |

## Benchmark Binaries

| Binary | Flags | Output |
|--------|-------|--------|
| `ecsie_bench_measure_tps` | `--model --workload --out-tps` | `step,timestamp_ms,tokens,tps` |
| `ecsie_bench_latency_variance` | `--model --workload --out-latency` | `request_id,ttft_ms,mean_itl_ms,p50_itl_ms,p99_itl_ms` |
| `ecsie_bench_gpu_utilisation` | `--model --workload --out-gpu [--poll-ms]` | `timestamp_ms,gpu_util_pct,mem_used_mb,H_t` |

## Output Files

| File | Description |
|------|-------------|
| `results/tps_<workload>.csv` | Tokens-per-second per step |
| `results/latency_<workload>.csv` | Per-request latency distributions |
| `results/gpu_util_<workload>.csv` | GPU utilisation + entropy traces |
| `results/tps.csv` | Merged TPS summary across all workloads |

## Ablation Configs

Ablations are run via `experiments/ablations/`. Each script disables one ECSIE component
and records the performance delta.

## Determinism

Set `ECSIE_SEED=42` environment variable to enable deterministic routing for reproducible
benchmarks. Note: this disables entropy-adaptive behaviour.

```bash
ECSIE_SEED=42 ./benchmarks/run_all.sh
```

