# Native-Windows decode tuning (Qwen3-30B-A3B, RTX 4060 / DDR4-2666)

Measured 2026-05-30 on the native clang-cl build (WSL retired). Hardware: RTX 4060
8 GB (sm_89), Ryzen 5800X (8 cores / 16 threads), DDR4-2666 **balanced dual-channel**
(~28 GB/s measured by `benchmarks/membw`), Windows 11 + HAGS On.

Model: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (48 layers, all-MoE, 128 experts,
top-8). Workload: `benchmarks/workloads/long_instruct_200.json` (200-token decode).

## TL;DR — recommended configs

**Byte-exact (lossless greedy) — ~27–28.5 tok/s:**
```
ECSIE_SPEC=on ECSIE_SPEC_DEPTH=1 ECSIE_SPEC_DRAFT=lens ECSIE_LENS_DRAFT_LAYER=46
ECSIE_CPU_EXPERTS=on ECSIE_MOE_HOTCOLD=3 ECSIE_CPU_MOE_THREADS=8
ECSIE_VRAM_POOL_MB=4500      # or leave unset: the adaptive formula auto-picks ~4.4 GB
```
(+ HAGS On in Windows Graphics settings; + balanced 16/16 dual-channel RAM at 2666.)

**Max throughput (lossy opt-in) — ~30 tok/s (28.6–31):**
```
... all of the above, plus:
ECSIE_SPEC_ACCEPT=typical:0.3,0.09     # Medusa-style entropy-gated acceptance
# optional, marginal: ECSIE_TOPK_CUM_THRESHOLD=0.90
```
`typical` acceptance raises accept 0.818 → 0.942 (more tokens/step). Output is NOT
byte-identical to strict greedy, but verified coherent + technically accurate (no
perplexity harness exists; quality judged by generated-text inspection).

## Env var reference (re-validated on native Windows)

| var | optimum | notes |
|-----|---------|-------|
| `ECSIE_CPU_MOE_THREADS` | **8** | physical cores. Sweep: 8 (best) ≈ 16 > 10 > 12 > 6. Native handles full-SMT (16) better than WSL did, but 8 still wins; 10/12 collide with GPU/spec helper threads. |
| `ECSIE_MOE_HOTCOLD` | **3** | hottest N experts on GPU (VRAM-resident), cold tail on CPU, overlapped. N3 (15.0 steps/s) > N2 (14.9) > N0 all-CPU (12.3) at pool 4500 → **+22% under spec**. N≥4 regresses (M=1 WDDM launch cost). Needs the big pool to compose with spec. |
| `ECSIE_VRAM_POOL_MB` | **4500** | spec peak on the 8 GB card; 4600 cliffs (working set crosses VRAM). Leave unset to use the model-aware adaptive formula (engine.cpp) → ~4.4 GB when ≥7 GB free. Conditional on free VRAM (desktop-heavy → OOM; adaptive falls back safe). |
| `ECSIE_SPEC_DRAFT` / `_DEPTH` / `_LENS_DRAFT_LAYER` | `lens` / `1` / `46` | logit-lens self-spec, depth-1 (structural — eagle pos-2 accept ~0.05). Layer 46 is quant-independent; not HW-sensitive. accept 0.818 strict. |
| `ECSIE_SPEC_ACCEPT` | `strict` (lossless) / `typical:0.3,0.09` (30-tps) | typical = entropy-gated acceptance, lossy, +~10% user-facing. |
| `ECSIE_TOPK_CUM_THRESHOLD` | `1.0` off (or `0.90` lossy) | skip low-prob experts on confident tokens. +6% standalone at 0.90 on the CPU-MoE path (was null on the old GPU path); Qwen3's flat routing limits firing; stacks only marginally with `typical`. |
| `ECSIE_CPU_INT_DOT` | off | int8 q8×q4 dot — null at 28 GB/s + lossy. Wins only on DDR5/AVX-512. |

## The bandwidth ceiling

The decode step (~80 ms) is **~71% CPU-MoE DRAM byte-streaming at ~28 GB/s**. GPU work
(attention via the fast device path ~2 ms, lm_head ~2.6 ms ×3, lens draft) is fully
**overlapped — off the critical path** (verified: lm_head fusion moved the step 78.0→77.8
ms ≈ 0%). Lens spec is structurally pinned at depth-1. Therefore the **only** wall-clock
lever is *reducing expert bytes read per token without adding compute*.

- **Lossless ceiling ≈ 28.5 tok/s.** The only remaining lossless code lever is
  **Q4_K_S routed experts** (keep `ffn_down` Q4_K instead of the Q6_K promotion on the
  24/48 promoted layers → ~6.6% fewer MoE bytes → ~1–2% user-facing; reuses
  `q4k_dot_fused`, zero added compute). Requires a requantized Q4_K_S GGUF.
- **Lossy → ~30 tok/s** via `ECSIE_SPEC_ACCEPT=typical` (and/or `ECSIE_TOPK_CUM_THRESHOLD`).
- **A genuinely lossless 30 needs hardware** (matched 2×16 GB DDR4-3200 → ~40 GB/s),
  which would also re-open the deferred compute-side levers (int-dot, etc.).

## Out of scope (Phase 2/3/4 — do not pull forward)

KTransformers expert deferral · dynamic GPU↔CPU expert migration · adaptive-depth ·
multi-request batching · KV/sliding-window compression · sparse-tree attention kernel ·
per-expert slab pool · full-step CUDA-graph capture · batched dequant · AVX-512/AMX kernels.

## Diagnostics

`ECSIE_PROFILE_SUMMARY=1` (per-timer dump) · `ECSIE_MOE_TIMING=1` (MoE % of step) ·
`ECSIE_SPEC_DIAG=1` (accept stats) · `ECSIE_ROUTE_HIST=1` (expert concentration) ·
`ECSIE_VRAM_PROBE=1` (free-VRAM headroom) · `ECSIE_DUMP_TOKENS=<path>` (output capture).
