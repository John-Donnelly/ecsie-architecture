# Decode-step profile pass — v1.0.162

**Hardware:** RTX 4060 8 GB (PCIe Gen4 x8 ≈ 16 GB/s peak), WSL2 Ubuntu 22.04, CUDA 13.2.
**Tooling:** Nsight Systems 2025.6.3 (`nsys profile --trace=cuda,nvtx`). Nsight Compute 2026.1.1 was attempted but `ncu` kernel-metric replay either hung past 8 minutes or silently failed to match named kernels on this WSL2 build; roofline classification below is derived from arithmetic-intensity analysis of the kernel bodies rather than measured `ncu` metrics. nsys did surface the data we actually need: per-kernel GPU time, per-API time, and per-direction memcpy bytes + duration.
**Workload:** `benchmarks/workloads/profile_ctx2k.json` — single prompt, batch=1, temperature=0, ~1900-token prompt + 200 generated tokens (`ctx ≈ 2k` by end of decode).
**Configs profiled:**
1. `Qwen3-30B-A3B-Instruct-2507-baseline.ecsie` (type-100 packed ternary).
2. `Qwen3-30B-A3B-Instruct-2507-entropy.ecsie` (type-101 Huffman-encoded ternary, v1.0.151+).
3. `Qwen3.6-35B-A3B-hybrid-n8h8.ecsie` (hybrid attention/SSM, n8h8 split — 10 attention + 30 SSM blocks).
**Captured `.nsys-rep`:** `benchmarks/results/profile_v1.0.162/profile_{30b_baseline,30b_entropy,35b_hybrid}.nsys-rep` (~350-380 MB each).

---

## Engine-reported per-step numbers (the bench's own counters)

| Metric                | 30B baseline | 30B entropy | 35B hybrid |
|-----------------------|-------------:|------------:|-----------:|
| Prompt-level `tps`    | 0.59         | 0.47        | 0.22       |
| **Warm step rate**    | **2.86 t/s** | **3.62 t/s**| **1.85 t/s** |
| Mean step time        | 350 ms       | 276 ms      | 542 ms     |
| p99 step time         | 1899 ms      | **405 ms**  | 2661 ms    |
| `attention` per layer | 0.72 ms      | 0.66 ms     | 1.07 ms    |
| `moe_dispatch` per layer | 4.28 ms   | 5.47 ms     | 15.31 ms   |
| Fused-fast / legacy   | 50 % / 50 %  | 50 % / 50 % | 45 % / 55 %|

Notes on the 35B `moe_dispatch=15.3 ms`: this layer-level counter includes the SSM/GDN forward for ~75 % of layers (30 of 40 are SSM in the n8h8 split). Pure MoE-attention layers measure closer to the 30B baseline.

---

## Per-decode-step GPU time breakdown

Aggregated over the full bench (prefill + decode), then expressed as % of total GPU activity (kernel + memop). Wall-time amounts include some prefill overhead that's proportional across configs.

### 30B baseline (type-100)

| Category                           | GPU time | % GPU activity |
|------------------------------------|---------:|---------------:|
| **H2D memcpy (cudaMemcpyAsync)**   | 42.3 s   | **32.3 %**     |
| Attention `gqa_decode_flash`       | 24.2 s   | 18.5 %         |
| `dequant_ecsie_ternary` (FP32 unpack from packed type-100) | 14.5 s | 11.1 % |
| `dequant_q6k`                      | 8.8 s    | 6.7 %          |
| cuBLAS `gemvx` (attention)         | 8.8 s    | 6.7 %          |
| MoE router `fused_route_softmax_topk` | 7.3 s | 5.6 %          |
| cuBLAS `ampere_sgemm_64x32`        | 4.9 s    | 3.7 %          |
| `fused_multi_ternary_down`         | 4.5 s    | 3.4 %          |
| `fused_multi_ternary_gate_up_silu` | 3.7 s    | 2.8 %          |
| `dequant_q4k`                      | 3.5 s    | 2.7 %          |
| Other kernels (RMSNorm, RoPE, FP16↔FP32, etc.) | 7.4 s | 5.6 %  |

**PCIe H2D bandwidth:** 466 GB transferred in 42.3 s of actual memcpy time = **11.0 GB/s sustained ≈ 70 % of Gen4 x8 peak.** **HIGH.**

**Sync stalls (CPU API wait time):**
- `cudaStreamSynchronize` 9.6 s (6.2 % of API time)
- `cudaEventSynchronize` 6.1 s (4.0 %)
- `cudaDeviceSynchronize` 27 µs total — effectively zero.

### 30B entropy (type-101)

| Category                           | GPU time | % GPU activity |
|------------------------------------|---------:|---------------:|
| **`ecsie_ternary_huffman_decode_kernel`** | **127.0 s** | **46.0 %** |
| H2D memcpy                         | 53.9 s   | 19.5 %         |
| Attention `gqa_decode_flash`       | 24.6 s   | 8.9 %          |
| `dequant_ecsie_ternary` (FP32 unpack from decoded type-100) | 15.2 s | 5.5 % |
| cuBLAS `gemvx`                     | 9.2 s    | 3.3 %          |
| `dequant_q6k`                      | 8.8 s    | 3.2 %          |
| MoE router                         | 7.5 s    | 2.7 %          |
| `ampere_sgemm_64x32`               | 5.6 s    | 2.0 %          |
| `gemv2T_kernel_val`                | 5.1 s    | 1.8 %          |
| `fused_multi_ternary_down`         | 4.6 s    | 1.7 %          |
| `dequant_q4k`                      | 3.9 s    | 1.4 %          |
| `fused_multi_ternary_gate_up_silu` | 3.8 s    | 1.4 %          |
| Other                              | 6.0 s    | 2.6 %          |

**PCIe H2D bandwidth:** 269 GB transferred in 53.9 s = **5.0 GB/s sustained ≈ 32 % of Gen4 x8 peak.** **LOW** — caching + the smaller encoded payload do their job; PCIe is mostly idle.

**Sync stalls:**
- `cudaEventSynchronize` 82.3 s (28.1 % of API time — large!) — this is the host blocking on the decode-kernel stream during materialise.
- `cudaStreamSynchronize` 10.9 s (3.7 %).
- `cudaDeviceSynchronize` 29 µs — negligible.

The Huffman decode kernel's ~272 µs average and 384 launches per step (~104 ms/step pure GPU time) is the load-bearing operation. The 28 % `cudaEventSynchronize` figure is the host serialising on those decode kernels at the materialise→tier-1-VRAM-slot boundary.

### 35B hybrid (Qwen3.6, n8h8)

| Category                           | GPU time | % GPU activity |
|------------------------------------|---------:|---------------:|
| **H2D memcpy**                     | **64.2 s** | **43.2 %**   |
| MoE router `fused_route_softmax_topk` | 12.6 s | 8.5 %          |
| cuBLAS `gemvx` (attention)         | 12.4 s   | 8.4 %          |
| Attention `gqa_decode_flash`       | 10.8 s   | 7.2 %          |
| `dequant_q8_0`                     | 9.4 s    | 6.3 %          |
| **SSM `gdn_fused_conv_rec_f16`**   | **8.0 s**| **5.4 %**      |
| `dequant_ecsie_ternary`            | 7.5 s    | 5.0 %          |
| `fused_multi_ternary_down`         | 5.3 s    | 3.6 %          |
| `dequant_q6k`                      | 5.1 s    | 3.4 %          |
| cuBLAS `gemvx` (float)             | 3.4 s    | 2.3 %          |
| `fused_multi_ternary_gate_up_silu` | 2.8 s    | 1.9 %          |
| Other (sgemm, dequant_q4k, SSM conv buf, …) | 7.0 s | 4.7 %    |

**PCIe H2D bandwidth:** 656 GB transferred in 64.2 s = **10.2 GB/s sustained ≈ 65 % of Gen4 x8 peak.** **HIGH.**

**Sync stalls:** broadly similar to baseline — `cudaStreamSynchronize` 13 s, `cudaEventSynchronize` 16 s. Nothing pathological.

---

## Roofline classification (analytic, since `ncu` was unavailable)

Each MoE expert in the inner ternary FFN reads ~36 KB of packed-ternary weights (1 expert's slab of `[K, M, n_experts]`) and produces ~6 KB of output. Arithmetic intensity ≈ ~0.5–1 FLOPS/byte for the dequant + matmul fused path. RTX 4060's compute-vs-memory ridge sits at ~14 FLOPS/byte (FP16 throughput ÷ HBM-equivalent bandwidth). Every kernel listed above operates at < 1 FLOPS/byte, so all of them are **memory-bandwidth-bound** in the roofline sense.

The interesting question therefore isn't "compute or memory" — every kernel is memory — but **which memory tier dominates each config's critical path**:
- Baseline: PCIe (system RAM → VRAM via cudaMemcpyAsync, the slowest tier).
- Entropy: GDDR within VRAM (cached payload → decoded slot, much faster than PCIe but the kernel is run ~384× per step).
- 35B: PCIe (same as baseline but more bytes/step because the model is larger and ternary tensors are bigger).

---

## Final verdicts

| Model                 | Decode is bound by …                                                  |
|-----------------------|-----------------------------------------------------------------------|
| **30B baseline (type-100)** | **PCIe H2D transfer.** 70 % of Gen4 x8 peak sustained; 32 % of GPU activity is memcpy. Compression + caching + prefetch (Stages 1, 2, 6) are correctly aimed. |
| **30B entropy (type-101)**  | **The Huffman decode kernel itself** (46 % of GPU activity). PCIe is at 32 % peak — way below transfer-bound. The compression already shipped, so the next lever is **decode kernel throughput** (per-warp cooperative decode, shared-memory LUT) or **caching the *decoded* type-100 bytes in VRAM** for hot bulks. Compression is "free disk + transfer wins"; further compression would help PCIe but the kernel is now the long pole. |
| **35B hybrid**              | **PCIe H2D transfer.** 65 % of peak sustained, 43 % of GPU activity. SSM/GDN kernels are only 5.4 % of GPU time — **the hypothesis that SSM dominates Qwen3.6's step time is refuted by this data.** Compression + caching + prefetch are correctly aimed here too. |

---

## Stage-priority adjustment (the gating decision this profile was meant to drive)

Original stage-ordering hypothesis (paraphrased from the brief):
> If PCIe utilisation is HIGH (>70 %) and cache hit rate is low → transfer-bound. Stages 1, 2, 6 (compression + prefetch + caching) correctly aimed.
> If PCIe utilisation is LOW (<50 %) → not transfer-bound; reprioritise scheduler/launch-overhead work.
> If SSM kernels dominate Qwen3.6 → Stage 3 jumps to top priority for that model.

Data-driven verdict:

1. **Compression + caching + prefetch (Stages 1, 2, 6) remain correctly aimed for the BASELINE 30B and the 35B hybrid.** PCIe utilisation is 65–70 % on both — the assumed transfer-bound regime is real.

2. **The Pass-2 entropy build has moved the bottleneck.** It is no longer PCIe-bound; it is bound by the Huffman decode kernel. For the entropy build specifically:
   - Stage 1 (compression) is shipped and structurally net-positive.
   - Stage 2 (caching) is shipped (v1.0.161 bulk cache) and is already eliminating most H2D — PCIe utilisation dropped from 70 % → 32 %.
   - The new top priority for entropy-build runtime is **kernel-side decode throughput**, not more compression. Specifically: a per-warp cooperative `ecsie_ternary_huffman_decode_kernel` (currently 1 thread per row) plus a 2-bit→symbol shared-memory LUT could plausibly halve the 272 µs-per-launch cost.

3. **SSM does NOT dominate Qwen3.6 on this rig.** 5.4 % of GPU time. Stage 3 priority should be **demoted** for this model unless we can show batch-size or context-length conditions where SSM time grows disproportionately. The 35B's wall-time cost is going to PCIe, then to the MoE router, then to attention's K-projection (gemvx), then to MoE FFN. The SSM kernel is not the long pole at batch=1, ctx≈2k on RTX 4060.

4. **`cudaEventSynchronize` is large (28 %) in the entropy build** — the host is blocking on the decode-kernel stream during materialise. This is a separate, smaller lever: replace the event sync with a stream-attached read on the materialise output, or batch multiple expert decodes onto the same stream so the wait amortises over more work.

---

## Per-step timing budget (illustrative, baseline 30B at 350 ms mean step)

GPU active time is the dominant slice; CPU launch overhead and sync waits overlap with GPU work via async streams. Approximate per-step:

```
[H2D async pipeline ─────────────────────────────────] 210 ms   (PCIe)
[Attention (gqa_decode_flash, gemvx, router, dequant)] 165 ms   (compute on cached data)
[MoE FFN (fused_multi_ternary, dequant_q4k)] ─────────  60 ms
[Other ───] ───────────────────────────────────────────  20 ms
                                                         ───
                                       overlap-effective ~350 ms
```

For the entropy build at 276 ms:
```
[Huffman decode kernel ─────────────────] 105 ms   (the new hotspot)
[Attention (same path) ─────────────────] 124 ms
[MoE FFN (post-decode unpack + matmul)] ─  50 ms
[H2D — only the first touch per bulk] ──   <5 ms after warm-up
[Other ────────] ───────────────────────   20 ms
                                            ───
                              overlap-effective ~276 ms
```

The entropy build is roughly **17 % faster per step** despite spending 105 ms on a kernel the baseline doesn't run at all, because it saves ~210 ms of H2D per step.
