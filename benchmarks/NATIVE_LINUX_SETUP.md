# Native-Linux validation of the 30-tok/s goal

**Why:** On the WSL2 dev box the CPU-MoE path (Phase 1) is hard-capped by memory
bandwidth — `membw.cpp` measures only **~16 GB/s** for a plain sequential DRAM
read, vs the Ryzen 5800X / DDR4-3200's **~40-50 GB/s native**. WSL2's memory
virtualization roughly **halves** bandwidth. Since the CPU-MoE is purely
DRAM-bandwidth bound (proven: integer kernel + SW prefetch both failed to help),
native Linux should ~2.5× it with **zero code changes**:

| | DRAM BW | CPU-MoE/step | raw tok/s | user-facing (lens spec) |
|---|---|---|---|---|
| WSL2 (measured) | ~16 GB/s | ~74 ms | ~9-10 | **~17** |
| native (predicted) | ~40 GB/s | ~29 ms | ~22-25 | **~30-33** |

The committed engine (fused CPU kernel v1.0.225, the +9.6% win) is unchanged —
this is an **environment** test, not a code change.

## Prerequisites (bare-metal Linux, NOT WSL)
- NVIDIA driver + **CUDA toolkit** (`nvcc` in PATH; match your driver — 12.x or 13.x)
- `cmake` ≥ 3.18, `g++` ≥ 11 (C++17), `make`
- ≥ 32 GB RAM (model is ~18 GB; must be page-cache resident)
- The **Q4_K_M model**: copy `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` to local
  disk, e.g. `~/ecsie_models/q4.gguf` (NOT a network/9p mount).

## Steps
```bash
# 1. clone (or copy) the repo to native Linux local disk
git clone <repo> ECSIE && cd ECSIE        # or rsync from the Windows drive

# 2. configure + build the bench binary (fresh build dir — do NOT reuse build_linux)
cmake -B build-native -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target ecsie_bench_measure_tps -j$(nproc)
#   -> produces ./bin/ecsie_bench_measure_tps

# 3. run the validation harness
MODEL=~/ecsie_models/q4.gguf bash benchmarks/native_linux_bench.sh
```

## How to read the result
1. **Step 1 (membw)** is decisive. If it prints **≥ 35 GB/s**, the prediction
   holds and the engine numbers should follow. If it's still **~16 GB/s**, the
   limit is the *box* (not WSL) — then 30 tok/s needs DDR5/AVX-512 or Phase 2,
   and the WSL result was already representative.
2. **Step 3** — the "spec lens" line is the user-facing number. Compute
   `user_tok/s = warm_step_rate × (200 / spec.steps)`. Target: **~30-33**.
3. Also check `moe_dispatch=` (per-layer CPU-MoE ms): WSL was ~1.56 ms/layer
   clean; native should be **~halved** if bandwidth doubled.

## If native confirms ~30
The fused CPU kernel + lens spec already in `main` (v1.0.225) get you there on
native hardware with no further work. If you then want **higher** (40+ tok/s),
that's where the deferred Phase-2 smart-hybrid (hot experts in VRAM) compounds.
