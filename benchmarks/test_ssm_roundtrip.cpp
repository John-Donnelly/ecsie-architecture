// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/test_ssm_roundtrip.cpp
//
// SSM state save/restore round-trip diagnostic.
//
// The pre-existing 35B-hybrid OFF≠ON divergence (surfaced by v1.0.178's
// 3-arm equivalence check) is post-spec numerical drift: tokens 0..4 match
// the spec-off arm, then position 5 diverges and snowballs.  Spec fires only
// at positions 0 and 1, so something perturbed by spec_verify_draft is not
// being restored by truncate(p) + restore_ssm_state().
//
// This binary tests one specific hypothesis: is SsmBlock::save_gpu_state()
// → reset_gpu_state() → restore_gpu_state() byte-faithful?  If save/restore
// is broken (e.g., silent cudaMalloc failure noted in v1.0.117), the
// snapshot would not survive a perturbation and fp_post != fp_pre on at
// least one block.  If the round-trip passes on every SSM block, save/restore
// is correctly accounting for every byte of d_h_state + d_conv_buf — the
// drift root cause is elsewhere (CUDA graph capture, VRAM-pool tier
// residency, GPU MoE allocator state, etc.).
//
// Usage:
//   ecsie_test_ssm_roundtrip --model <path.ecsie>
//                            [--prompt S] [--n-warm N]
//
// Exit 0 PASS (every block restored byte-faithfully).
// Exit 1 FAIL (at least one block's fp_post != fp_pre).
// Exit 2 setup error.

#include "ecsie/engine.hpp"

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
    int n_warm = 8;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt     = argv[++i];
        else if (a == "--n-warm" && i + 1 < argc) n_warm     = std::atoi(argv[++i]);
    }
    if (model_path.empty()) {
        std::fprintf(stderr,
            "[ssm_roundtrip] error: --model <path.ecsie> is required\n");
        return 2;
    }

    ecsie::Engine engine;
    try {
        engine.load_model(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ssm_roundtrip] model load error: %s\n", e.what());
        return 2;
    }
    if (!engine.is_loaded()) {
        std::fprintf(stderr, "[ssm_roundtrip] model failed to load\n");
        return 2;
    }

    ecsie::Engine::SsmRoundtripResult r;
    try {
        r = engine.ssm_state_roundtrip_diagnostic(prompt, n_warm, /*n_perturb=*/0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ssm_roundtrip] diagnostic error: %s\n", e.what());
        return 2;
    }

    if (r.total_ssm_blocks == 0) {
        std::fprintf(stderr,
            "[ssm_roundtrip] model has no SSM blocks — nothing to test\n");
        return 2;
    }

    int n_pass = 0, n_fail_restore = 0, n_fail_perturb = 0;
    for (const auto& b : r.blocks) {
        const char* verdict = b.restored_matches ? "PASS" : "FAIL_RESTORE";
        if (!b.perturbed_differs) verdict = "FAIL_PERTURB";

        std::printf("block %3d  fp_pre=%016" PRIx64
                    "  fp_perturbed=%016" PRIx64
                    "  fp_post=%016" PRIx64
                    "  %s\n",
                    b.block_idx, b.fp_pre, b.fp_perturbed, b.fp_post, verdict);

        if (!b.perturbed_differs) ++n_fail_perturb;
        else if (b.restored_matches) ++n_pass;
        else ++n_fail_restore;
    }

    std::printf("---\n");
    std::printf("VERDICT: %s   total=%d  pass=%d  fail_restore=%d  fail_perturb=%d\n",
                r.all_ok ? "PASS" : "FAIL",
                r.total_ssm_blocks, n_pass, n_fail_restore, n_fail_perturb);
    return r.all_ok ? 0 : 1;
}
