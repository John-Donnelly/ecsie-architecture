// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/test_forward_equivalence.cpp
//
// Forward-equivalence diagnostic: does a synthetic spec_verify-rewind cycle
// perturb engine-wide state (VRAM pool / GPU MoE allocator / kernel
// scheduling history) in a way that mutates subsequent forward output?
//
// v1.0.179 + v1.0.180 already proved that per-block GPU persistent state
// (SsmBlock::d_h_state, d_conv_buf; Attention's d_k_cache_f16 at slots
// [0, p)) is byte-faithfully restored across the verify-rewind cycle on
// 35B-hybrid.  Yet OFF != ON outputs diverge from position 5.  Suspect:
// the v1.0.117 "execution-history-dependent numerical effect in the GPU
// MoE dispatch" lives below per-block state.
//
// This binary isolates that question.  It runs:
//   1. prefill(prompt[:-1])  → state at position p
//   2. snapshot, forward(prompt[-1]) → hidden_A
//   3. rewind to (kv_pos=p, ssm=snap)
//   4. synthetic verify-rewind: save SSM, forward K drafts, truncate,
//      restore SSM
//   5. forward(prompt[-1]) → hidden_B
//
// If hidden_A == hidden_B byte-for-byte, the engine is fully deterministic
// under verify-rewind.  If they differ, engine-wide state was mutated.
//
// Usage:
//   ecsie_test_forward_equivalence --model <path.ecsie>
//                                   [--prompt S] [--k N]
//
// Exit 0 PASS (identical), 1 FAIL (differ), 2 setup error.

#include "ecsie/engine.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt =
        "The quick brown fox jumps over the lazy dog. In a few sentences, "
        "explain why deterministic decoding matters for testing.";
    int K_drafts = 6;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt     = argv[++i];
        else if (a == "--k"      && i + 1 < argc) K_drafts   = std::atoi(argv[++i]);
    }
    if (model_path.empty()) {
        std::fprintf(stderr,
            "[forward_equiv] error: --model <path.ecsie> is required\n");
        return 2;
    }

    ecsie::Engine engine;
    try {
        engine.load_model(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[forward_equiv] model load error: %s\n", e.what());
        return 2;
    }
    if (!engine.is_loaded()) {
        std::fprintf(stderr, "[forward_equiv] model failed to load\n");
        return 2;
    }

    ecsie::Engine::ForwardEquivResult r;
    try {
        r = engine.forward_equivalence_diagnostic(prompt, K_drafts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[forward_equiv] diagnostic error: %s\n", e.what());
        return 2;
    }

    if (!r.ok) {
        std::fprintf(stderr, "[forward_equiv] diagnostic returned ok=false\n");
        return 2;
    }

    std::printf("prompt_len            = %d\n",  r.prompt_len);
    std::printf("kv_pos_at_test        = %d\n",  r.kv_pos);
    std::printf("test_token            = %d\n",  r.test_token);
    std::printf("K_drafts              = %d\n",  r.K_drafts);
    std::printf("hidden_dim            = %d\n",  r.hidden_dim);
    std::printf("fp_hidden_a           = %016" PRIx64 "\n", r.fp_hidden_a);
    std::printf("fp_hidden_b           = %016" PRIx64 "\n", r.fp_hidden_b);
    std::printf("n_diff_elements       = %d / %d\n", r.n_diff_elements, r.hidden_dim);
    std::printf("max_abs_diff          = %.6e\n", r.max_abs_diff);
    // v1.0.185: trace[0] = INPUT fingerprint (d_h before any block runs);
    // trace[i+1] = OUTPUT of block i.  first_diverging_block == 0 now means
    // the input itself differs (severe — means d_h is corrupted between
    // forward A and forward B); first_diverging_block == i+1 with i >= 0
    // means block i is where divergence appears.
    std::printf("first_diverging_index = %d (of %zu entries; index 0 = input, 1+ = block 0+)\n",
                r.first_diverging_block, r.trace_a.size());

    int show_until = static_cast<int>(r.trace_a.size());
    if (r.first_diverging_block >= 0) {
        show_until = std::min(show_until, r.first_diverging_block + 4);
    }
    std::printf("--- hidden trace (first %d entries; idx 0 = input) ---\n", show_until);
    std::printf(" idx  label         fp_A                fp_B           match\n");
    for (int i = 0; i < show_until; ++i) {
        const bool match = (r.trace_a[static_cast<std::size_t>(i)] ==
                            r.trace_b[static_cast<std::size_t>(i)]);
        char label[16];
        if (i == 0) std::snprintf(label, sizeof(label), "input  ");
        else        std::snprintf(label, sizeof(label), "blk%3d ", i - 1);
        std::printf("%4d  %s  %016" PRIx64 "  %016" PRIx64 "  %s\n",
                    i, label,
                    r.trace_a[static_cast<std::size_t>(i)],
                    r.trace_b[static_cast<std::size_t>(i)],
                    match ? "ok" : "DIFFER");
    }
    // v1.0.186: sub-block trace (post-SSM/attn d_subout, post-MoE d_subout).
    // Index 2*i = block i post-SSM/attn, index 2*i+1 = block i post-MoE.
    std::printf("\nfirst_diverging_sub_index = %d (of %zu entries; 2*i = blk i post-SSM/attn, 2*i+1 = blk i post-MoE)\n",
                r.first_diverging_sub_index, r.sub_trace_a.size());
    int sub_show_until = static_cast<int>(r.sub_trace_a.size());
    if (r.first_diverging_sub_index >= 0) {
        sub_show_until = std::min(sub_show_until, r.first_diverging_sub_index + 4);
    } else {
        sub_show_until = std::min(sub_show_until, 6);
    }
    std::printf("--- sub-block trace (first %d entries) ---\n", sub_show_until);
    std::printf(" idx  label             fp_A                fp_B           match\n");
    for (int i = 0; i < sub_show_until; ++i) {
        const bool match = (r.sub_trace_a[static_cast<std::size_t>(i)] ==
                            r.sub_trace_b[static_cast<std::size_t>(i)]);
        char label[24];
        const int blk_idx = i / 2;
        const bool is_moe = (i % 2 == 1);
        std::snprintf(label, sizeof(label), "blk%3d %s",
                      blk_idx, is_moe ? "post-MoE " : "post-attn");
        std::printf("%4d  %s  %016" PRIx64 "  %016" PRIx64 "  %s\n",
                    i, label,
                    r.sub_trace_a[static_cast<std::size_t>(i)],
                    r.sub_trace_b[static_cast<std::size_t>(i)],
                    match ? "ok" : "DIFFER");
    }
    // v1.0.188 per-expert trace.  Only populated when the legacy
    // per-expert MoE FFN path actually ran (use ECSIE_PHASE_*=off +
    // ECSIE_FUSED_MULTI_*=0 to force it).  Otherwise this section
    // shows zero entries.
    std::printf("\nfirst_diverging_expert_index = %d (of %zu per-expert d_out fingerprints)\n",
                r.first_diverging_expert_index, r.expert_trace_a.size());
    if (!r.expert_trace_a.empty()) {
        int exp_show = static_cast<int>(r.expert_trace_a.size());
        if (r.first_diverging_expert_index >= 0) {
            exp_show = std::min(exp_show, r.first_diverging_expert_index + 4);
        } else {
            exp_show = std::min(exp_show, 8);
        }
        std::printf("--- per-expert d_out trace (first %d) ---\n", exp_show);
        std::printf(" idx       fp_A                fp_B           match\n");
        for (int i = 0; i < exp_show; ++i) {
            const bool match = (r.expert_trace_a[static_cast<std::size_t>(i)] ==
                                r.expert_trace_b[static_cast<std::size_t>(i)]);
            std::printf("%4d  %016" PRIx64 "  %016" PRIx64 "  %s\n",
                        i,
                        r.expert_trace_a[static_cast<std::size_t>(i)],
                        r.expert_trace_b[static_cast<std::size_t>(i)],
                        match ? "ok" : "DIFFER");
        }
    } else {
        std::printf("(legacy per-expert path did not run; set ECSIE_PHASE_*=off + "
                    "ECSIE_FUSED_MULTI_*=0 to force it)\n");
    }
    std::printf("---\n");
    std::printf("VERDICT: %s\n", r.identical ? "PASS (deterministic)" : "FAIL (drift detected)");
    return r.identical ? 0 : 1;
}
