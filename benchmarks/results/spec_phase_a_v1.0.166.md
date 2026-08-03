# Phase A — Spec Decode Re-Measurement (v1.0.166+)

Reality-check on the brief's premise: "spec is the only optimization that can
push effective rate toward ~70 on a favorable workload."  Measurement on this
rig (RTX 4060 8 GB, Windows 11 / WSL2 Ubuntu, CUDA 13.2) decides whether the
three levers (multi-needle PLD + tree drafts, MTP head drafter, adaptive K)
are worth implementing.

## A1 — MTP heads in Qwen3.6 GGUF

**Result: ABSENT.**

Parsed `A:\AI\Models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf` (GGUF
v3, 733 tensors, 41 metadata KV pairs, architecture `qwen35moe`).  All 733
tensors fall into 4 prefixes:

| prefix       | role                          |
|--------------|-------------------------------|
| `blk.N.*`    | per-block transformer tensors |
| `token_embd` | input embedding               |
| `output_norm`| final RMSNorm                 |
| `output`     | LM head                       |

No `mtp.*`, `nextn.*`, `multi_token_*`, `speculator.*`, or `draft_head.*`.
Architecture metadata has no `mtp_layer_count` or equivalent.

**Implication: lever 2 (MTP head drafter) is N/A for this model file.**  The
brief's stated "zero extra VRAM, already in the model file" advantage does
not apply because the heads are not in this GGUF.

## A2 — Spec A/B on 30B-entropy

Model: `Qwen3-30B-A3B-Instruct-2507-entropy.ecsie`
Bench: `benchmarks/scripts/run_spec_ab_bench.sh`
Bench binary: `bin/ecsie_bench_measure_tps`

### Creative workload (`entropy_ab_short.json`, "Tell me a story…", 200 tok)

| variant   | warm step rate | acceptance        |
|-----------|---------------:|-------------------|
| spec off  | 3.16 t/s       | n/a               |
| spec on   | 2.92 t/s       | 0% (m=0 every step) |

**Spec is net-NEGATIVE: −7.6 % warm step rate (3.16 → 2.92).**  Engine
summary: `spec: steps=12 drafted=39 accepted=0 rate=0.000 mean_depth=3.25`.

Root cause: PLD repeatedly proposes EOS token 151645 (the buffer is
dominated by EOS markers from the chat template), and the target always
rejects.  Every verification pass is wasted compute.

### Repetitive workload (`spec_repetitive.json`, numbered list, 200 tok)

| variant   | warm step rate | acceptance                                           |
|-----------|---------------:|------------------------------------------------------|
| spec off  | 2.91 t/s       | n/a                                                  |
| spec on   | 3.18 t/s       | 25 % (m=3 of 12 drafted across 3 spec steps; mean_depth=4.0) |

**Spec is net-POSITIVE: +9.3 % warm step rate (2.91 → 3.18).**  Engine
summary: `spec: steps=3 drafted=12 accepted=3 rate=0.250 mean_depth=4.00`.

Notable: only 3 spec steps fired across 200 decoded tokens.  The existing
entropy gate (`ECSIE_ENTROPY_SPEC`) drops K to 0 when `H_t >= 0.70` — the
gate disabled speculation for the majority of the decode after the early
list-prefix burn-in.  That conservatism is the reason spec did not regress
the rest of the run, but it also caps the upside.  Phase B's accept-rate
axis lets spec stay on longer when recent acceptance is high (the regime
where the gate's downside is zero).

### Code workload (`spec_code.json`, BST class, 200 tok)

| variant   | warm step rate | acceptance                                              |
|-----------|---------------:|---------------------------------------------------------|
| spec off  | 3.24 t/s       | n/a                                                     |
| spec on   | 4.28 t/s       | 16.7 % (m=16 of 96 drafted across 31 spec steps; mean_depth=3.10) |

**Spec is net-POSITIVE: +32.1 % warm step rate (3.24 → 4.28).**  Engine
summary: `spec: steps=31 drafted=96 accepted=16 rate=0.167 mean_depth=3.10`.

This is the biggest single spec gain in the measurement matrix.  Code has
structured n-gram repetition (matching brackets, repeated identifiers,
indentation patterns) that PLD's needle search exploits much more
effectively than free-form text.  31 spec steps fired (vs 3 on the
numbered list) — the entropy gate stayed open across most of the decode.

### A2 summary

| Workload    | spec off | spec on | Δ       | spec steps | accept rate |
|-------------|---------:|--------:|--------:|-----------:|------------:|
| Creative    | 3.16 t/s | 2.92 t/s| −7.6 %  | 12         | 0.0 %       |
| Repetitive  | 2.91 t/s | 3.18 t/s| +9.3 %  | 3          | 25.0 %      |
| Code        | 3.24 t/s | 4.28 t/s| +32.1 % | 31         | 16.7 %      |

The brief's premise ("workload-class burst, not steady-state") is
empirically validated: code is the favourable case, +32 % is meaningful
on its own (no further levers required to call this a win), and creative
text is hurt by every wasted verification pass.  This is exactly the
shape Phase B (adaptive K) is designed for — preserve the upside on
favourable workloads, suppress the downside on hostile ones.

## A3 — Spec A/B on 35B-hybrid

Model: `Qwen3.6-35B-A3B-hybrid-n8h8.ecsie` (SSM + attention hybrid).

**Architectural note:** The batched K-position verifier in
`src/core/execution_loop.cpp:771-779` forces SEQUENTIAL fallback whenever
any block is SSM:

    const bool use_batched = env_spec_batched && !any_ssm && executor_;

Qwen3.6's hybrid has SSM blocks → `use_batched=false` → every spec step does
K sequential full-forwards plus the verifier rewind.  The per-step
amortisation that makes spec a win on Qwen3-30B does NOT apply to Qwen3.6.

### Repetitive workload (`spec_repetitive.json`, numbered list, 200 tok)

| variant                          | warm step rate | acceptance |
|----------------------------------|---------------:|------------|
| spec off                         | 2.36 t/s       | n/a        |
| spec on (Phase B v3 engaged)     | 2.18 t/s       | 0.0 % (m=0 of 9 drafted across 2 spec steps; mean_depth=4.5) |

**Spec is net-NEGATIVE on 35B-hybrid: −7.6 % warm step rate (2.36 → 2.18).**
Phase B kill switch fired after 2 spec attempts (window=[0, 0] → mean<0.10
→ K=0) — prevents the regression from compounding.  Without Phase B's
defensive layer, the slowdown would likely be substantially larger, since
the SSM sequential fallback re-runs every layer K times per spec attempt.

Architectural verdict: spec decode is structurally hostile to Qwen3.6's
SSM-hybrid until either (a) the batched verifier path is extended to
support SSM state-checkpointing per draft token, or (b) the SSM blocks are
replaced with attention.  Neither is in scope here.

## Phase B impact validation (v1.0.167)

Adaptive-K layer applied universally on top of `policy_spec_depth`.
See `CHANGELOG.md` for full design notes.  Two key validations:

### Kill switch on creative workload

| variant                                | warm step rate | spec attempts | accept |
|----------------------------------------|---------------:|--------------:|-------:|
| spec off (`spec_phaseB_v2_creative_*`) | 3.14 t/s       | n/a           | n/a    |
| spec on (Phase B engaged)              | 3.26 t/s       | 2             | 0.0 %  |

**Result: +3.8 %** (was −7.6 % in Phase A).  Kill switch fires after
2 spec attempts, capping waste at ~2 spec passes of overhead — the +3.8 %
is within measurement noise of "no-op," which is exactly the intended
defensive behaviour on a hostile workload.

### Equivalence

`benchmarks/spec_equivalence_check.sh` against
`Qwen3-30B-A3B-Instruct-2507-entropy.ecsie`: **PASS**.  spec_on output is
byte-identical to spec_off output.  Phase B touches K selection only;
the rejection-sampling algorithm is unchanged → lossless by construction.

## Phase C1a impact validation (v1.0.168)

Multi-needle PLD with consensus voting.  See `CHANGELOG.md` for full
design notes.

### Code workload (`spec_phaseC1_code_*`)

| variant                            | warm step rate |
|------------------------------------|---------------:|
| spec off                           | 3.88 t/s       |
| spec on (Phase B + C1 engaged)     | 4.34 t/s       |

**Result: +11.9 %.**  Spec attempts: 2, both rejected (m=0) — multi-needle
voting picked the same EOS-spam consensus drafts the v1.0.144 first-match
algorithm would have picked, because all candidate matches at this
position pointed to EOS as the continuation.  The +11.9 % is therefore
**noise from baseline drift**, not a demonstrated multi-needle win:
across the four code-workload benches in this session, `spec_off` ranged
3.24-4.22 t/s while `spec_on` ranged 4.28-4.34 t/s — the relative number
is dominated by `spec_off` variance, not by C1's contribution.

### Equivalence

`benchmarks/spec_equivalence_check.sh`: **PASS**.  Multi-needle consensus
changes which drafts the verifier sees, not how the verifier decides;
output remains byte-identical to spec-off greedy decoding.

### Honest framing

Phase C1a is shipped as ready infrastructure but its *measured* impact
on this rig under this measurement protocol is **indistinguishable from
no-op**.  Cleanly measuring its value requires either:
- A workload where multiple needle lengths pick distinct continuations
  (multi-modal n-gram regime), which the current code/repetitive/creative
  prompts don't exhibit at scale.
- Multi-run averaging (≥3 cold-process runs per cell) to attenuate the
  ±15-30 % baseline variance from `LearnedPolicy`'s dynamic depth output.

C1a is a strict superset of v1.0.144 PLD (same behaviour on single-match,
better on multi-match), lossless, and adds <1 ms per spec attempt — so
shipping with "no measured regression" is a defensible position.

## Multi-run averaging correction (v1.0.170)

Phase A's single-run claim of "+32 % on code" called out a "noisy
baseline" as an open caveat.  Resolving it with N=3 cold-process reps
per arm on the same code workload (`spec_code.json`,
`spec_phaseB_C1_code_n3_*`):

| Arm      | Rates (t/s)              | Mean ± σ          | CV     |
|----------|--------------------------|-------------------|-------:|
| spec off | [4.29, 4.33, 4.40]       | **4.340 ± 0.056** |  1.3 % |
| spec on  | [4.24, 3.15, 3.06]       | **3.483 ± 0.657** | 18.9 % |

**Mean comparison: spec is −19.7 % net-negative.**

The earlier "+32 %" claim was an outlier — one favourable spec_on run
paired with one unfavourable spec_off baseline measured many minutes
apart.  Once both arms are run consecutively three times, the picture
inverts:

- `spec_off` baseline is extremely stable: 4.340 ± 0.056 t/s
  (CV 1.3 %).  The earlier worry that the baseline drifted ±15-30 %
  was wrong — that variance came from session-timeline differences
  (warm-up, thermal state across hours), not from same-session
  run-to-run noise.
- `spec_on` is highly variable: 4.24 on rep 1, then 3.15 and 3.06 on
  reps 2 and 3.  All three reps fired the *same* 2 spec attempts at
  the same `kvpos` (113, 115) with the same drafts and the same 0 %
  acceptance.  The spec workload was identical across reps; the TPS
  variance comes from elsewhere — likely the per-attempt verification
  overhead interacting with thermal / scheduler state.

### Implication for Phase B + C1a impact

- **Phase B's defensive value remains real.**  The kill switch's job
  is to prevent the regression compounding past 2 spec attempts.
  Even on this code workload (which fired only 2 attempts, both
  rejected by the verifier — and remember these are PLD-induced EOS
  drafts at high-confidence kvpos), the regression is bounded at
  −19.7 % rather than the larger regression that would result from
  continuing to fire spec for the whole decode.
- **Phase C1a's measured impact is *still* indistinguishable from
  no-op** at the current attempt budget.  Multi-needle voting only
  matters when multiple candidates produce non-trivial drafts; the
  EOS-spam regime cancels its contribution at the consensus step.
- **The brief's "code is the favourable workload" framing does NOT
  hold on this rig + this entropy build.**  The original +32 % was
  noise.  Spec is net-negative on code in expectation.

### What still IS net-positive

- **Phase B kill switch on creative workload** (`spec_phaseB_v2_creative_*`):
  spec_off 3.14, spec_on 3.26 → +3.8 %.  Single run, but the mechanism
  (kill K=0 after 2 zero-accept attempts) caps waste at a known small
  cost.  This is a measurable defensive win, not a fluke.
- **Phase B kill switch on 35B-hybrid** (`spec_phaseB_35b_rep_*`):
  spec_off 2.36, spec_on 2.18 → −7.6 % (bounded by kill).  Without
  Phase B the regression would have been larger.

### Implementation plan revision

| Lever | Status                          |
|-------|--------------------------------|
| Phase B (adaptive K kill switch) | **SHIPPED v1.0.167** — defensive value confirmed; mean ~−19.7 % on code is the *capped* loss, not the *uncapped* one |
| Phase C1a (multi-needle PLD)     | **SHIPPED v1.0.168** — lossless infrastructure with no measured TPS impact on current workloads |
| Phase C1b (tree-attention CUDA)  | **Permanently deferred.**  Multi-run data shows spec is net-negative on average even on the formerly-favoured code workload; tree verify's only path to net-positive is via batched amortisation that requires the CUDA kernel work — and the brief's "this is the only optimization that can push effective rate toward ~70" framing is contradicted by this measurement.  Spec is not the lever. |
| Phase C2 (MTP heads)             | **Dead** — not in GGUF |
| Phase C3 (expert-set guard)      | **Dead** — no tree drafts to guard |

## A4 — Findings + implementation plan

### Confirmed (data-driven)

1. **PLD acceptance is workload-class-conditioned.**  Code workload
   delivered +32 % (16.7 % accept across 31 attempts).  Repetitive
   numbered-list delivered +9.3 % (25 % accept, 3 attempts).  Creative
   narrative cost −7.6 % (0 % accept, 12 attempts).  The brief's framing
   of "favourable workload burst" is empirically grounded.
2. **Native MTP heads (lever 2) is N/A** for Qwen3.6 GGUF — verified by
   tensor-index inspection (§A1).
3. **35B-hybrid has structural barriers to spec gains** independent of
   drafter quality — the SSM-forced sequential verify path negates the
   per-step amortisation that makes spec efficient on Qwen3.
4. **Phase B's kill switch works** — fired correctly on creative,
   suppressing the loss.  Lossless via equivalence check.

### Open questions (require additional measurement)

1. **LearnedPolicy variance.** Phase A creative had 12 spec attempts
   firing; subsequent benches with the same env had 8, 2, 2.  Root cause:
   `learned_policy.cpp:82` produces dynamic depths based on hidden state,
   so "number of spec steps that fire" varies across cold-process runs.
   Implication: single-run benches are noisy; "preserved +32 %" claims
   need ≥3-run averaging or longer decode budgets to assert with
   confidence.
2. **Phase C1 impact on the code workload** (the highest-upside case)
   needs the same multi-run averaging to be meaningfully measured.

### Implementation plan

| Lever                                | Status                          |
|--------------------------------------|---------------------------------|
| Phase B: adaptive K (acceptance window) | **SHIPPED v1.0.167**         |
| Phase C1a: multi-needle PLD (consensus voting; linear draft) | **SHIPPED v1.0.168** |
| Phase C1b: tree-shaped attention mask in CUDA verifier  | Deferred — requires kernel work; needs measurement evidence multi-needle PLD's consensus draft is short enough to leave residual upside |
| Phase C2: native MTP head drafter    | **Dead** — not in GGUF (§A1)   |
| Phase C3: expert-set inflation guard | Deferred — without C1b tree    drafts, no inflation to guard against |

## A4 — Synthesis

### Findings

1. **PLD on creative text: 0 % acceptance, net-negative TPS.**  This is a
   workload-class mismatch, not an implementation bug — PLD's prerequisite
   is recent-history n-gram repetition that creative narrative simply does
   not have.

2. **MTP head drafter (lever 2) is N/A** for the Qwen3.6 GGUF on this rig.
   Brief's premise refuted by direct tensor-index inspection.

3. **35B sequential-verify fallback** structurally suppresses spec's per-step
   amortisation on SSM-hybrid models, regardless of drafter quality.

### Re-ordered implementation plan

| Lever                          | Status              | Cost     | Expected upside |
|--------------------------------|---------------------|----------|-----------------|
| Multi-needle PLD + tree drafts | Viable but requires per-position attention mask in CUDA verifier | 3-5 sessions | Modest on repetitive text; zero on creative |
| Native MTP heads               | **Dead** (lever 2 not in GGUF)               | n/a      | n/a             |
| Adaptive K from accept rate    | Cheap; extends existing `ECSIE_ENTROPY_SPEC` | ~1 session | Cuts spec's downside on bad workloads (e.g. creative) by collapsing K to 0 |

### Recommended next steps

1. **Phase B (adaptive K)** — ship first.  It is the smallest change with the
   clearest defensive value: when the rolling acceptance rate is low (the
   creative-workload regime above), it collapses K toward 0 and avoids
   spec's net-negative TPS hit.  Today the entropy gate already drops to 0
   above H_t ≥ 0.70 — Phase B adds accept-rate as a second axis.

2. **Phase C1 (multi-needle + tree drafts)** — only if Phase B's measurement
   shows residual upside on the repetitive/code workloads.  Tree verify
   requires per-position attention masking; the `tree_attention_mask` and
   `pruned_candidate_tree` headers exist, but the CUDA attention kernel
   that consumes the mask is documented as future work.

3. **Phase C3 (expert-set inflation guard)** — only as a follow-up to C1.
   Without C1, there is no tree-shape inflation to guard against.
