# Phase-0 spec-decoding backlog (missed roadmap optimizations)

Source: adversarial survey workflow (59 agents, 51 candidates → 13 kept). All
training-free, Phase-0-only (spec mechanism), hardware-appropriate (RTX 4060 8GB,
PCIe-bound MoE). Baseline already shipped: logit-lens self-draft + relaxed accept.

Honest framing: at the K=1 throughput optimum, tok/s levers must (a) accept more
per verify, (b) avoid a forward, or (c) turn spec off where net-negative.

## Ranked

> UPDATE (v1.0.214c): #1 self-verify harvest INVESTIGATED & SKIPPED. 5-agent
> adversarial design found it does NOT eliminate a forward at K=1: the next-step
> input is always the BONUS token (an unforwarded prediction), not d0 (whose hidden
> the verify computed) — so the harvest match never fires. And the "2 sequential
> forwards" baseline was already false for pure-attention Qwen3-MoE (commit re-forward
> already skipped via kv_already_extended truncate). Real K=1 floor = 1 main + 1
> intrinsic verify forward, neither harvestable. DO NOT IMPLEMENT.
>
> UPDATE (v1.0.214c): #3 and #4 SHIPPED. #3 → `ECSIE_SPEC_ACCEPT=typical:eps,dm`
> (Medusa entropy-gated, eps=0.3/dm=0.09; lossy). #4 → `ECSIE_LENS_MARGIN_GATE`
> (skip verify on low lens top1-top2 margin; lossless). Both opt-in, default-OFF,
> wired into the real accept/verify path (`src/core/execution_loop.cpp`). Still
> OPEN: #2 (goodput depth gate), #5 (token recycling), #7 (adaptive lens layer) —
> all *accept-more-per-verify* levers, physics-capped near break-even on 8GB MoE.

1. **Self-verify token harvesting** (HIGH ROI / M) — reuse the verify forward's
   accepted-token layer-46 hidden to seed the next draft, eliminating the next
   step's redundant main forward. ECSIE currently does main-forward + verify =
   2 cold-expert forwards/step (~1.17 fwd/token) → spec is net-NEGATIVE on 30B
   (−19.7% code). Harvest = 1 forward/step → spec net-POSITIVE. **The structural
   tok/s lever toward 13-15.** Lossless. Touches spec_verify_draft + SpecVerifyResult
   + commit path. Capture post-block-46, NOT post-final-norm.

2. **Wall-time goodput depth gate** (HIGH / M) — online bandit argmax_K accept/ms
   over K∈{0,2,4,6} with ε-greedy; K=0 auto-disables spec where it loses. Replaces
   open-loop entropy buckets. Measures on-device (cost curve is non-monotonic).

3. ✅ **SHIPPED** (v1.0.214c, `ECSIE_SPEC_ACCEPT=typical:eps,dm`). **Entropy-gated
   typical acceptance** (HIGH-MED / S) — Medusa min(eps,delta·exp(-H))
   per-slot on the verifier softmax. diag path already computes Z/p1/pd; un-gate it.
   ~30 lines. Highest ROI-per-effort accept lever. eps=0.3 delta=0.09 Medusa defaults.

4. ✅ **SHIPPED** (v1.0.214c, `ECSIE_LENS_MARGIN_GATE`, lossless). **Drafter-confidence
   verify-skip gate** (MED / S) — skip verify when lens
   top1-top2 margin < threshold (draft would be rejected anyway). ~15 lines, free
   signal from the lens argmax scan (track runner-up).

5. **Token Recycling adjacency-matrix self-drafter** (MED / M) — |V|×8 matrix
   M[t]=argtopk(target logits after t), harvested free per verify; BFS into the
   tree verifier. Complementary to lens. ~4MB host, zero VRAM/PCIe.

6. AdaEDL target-entropy-modulated relaxed delta (MED / S) — redundant with #3.

7. **Adaptive per-position lens-layer selection** (MED / M-L) — pick exit layer
   per step from online EWMA agreement table over {38,42,46}. The one lever that
   raises pos-1 accept directly (could exceed 0.71). Risk: extra blocking-copy
   syncs per candidate layer. Do only if #1-#4 don't clear 0.65 reliably.

## Marginal (skip unless needed)
8. AdaEDL draft early-stop — no-op at K=1.
9. Anisotropic depth budget — overlaps #4.
10. Suffix-automaton PLD — thin upside on Qwen3 BPE.
11. Jacobi n-gram harvest — duplicative of bonus-token observe().

## Sequencing
#3 + #4 (DONE, v1.0.214c) → #1 (SKIPPED — see UPDATE) → #2 (goodput) → #7 (if gate
not cleared) → #5 (second drafter). **Remaining open: #2, #5, #7** — all physics-
capped near break-even on 8GB MoE; worth at most low effort until >8GB hardware.
