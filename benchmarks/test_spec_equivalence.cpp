// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/test_spec_equivalence.cpp
//
// Single-arm generator for the speculative-decoding output-equivalence check.
//
// Speculative decoding must be LOSSLESS: the tokens it accepts must be
// byte-identical to what plain greedy decoding produces.  Verifying that
// invariant is complicated by a real engine property — the tiered streaming
// architecture keeps a warm expert cache that PERSISTS across generate()
// calls, and tier residency shifts numerics.  Two generate() calls in the
// SAME process therefore are not guaranteed to agree even with identical
// settings: the second call starts from a different (warm) cache state.
//
// That cross-call non-determinism is inherent to the architecture, not a bug
// to fix here.  The harness works around it by running each comparison arm
// in a FRESH PROCESS: every arm starts from an identical empty-cache state
// and is internally deterministic.
//
// This binary is therefore ONE arm: it does exactly one generate() (greedy,
// temperature 0, fixed prompt, default 64 tokens) and prints the resulting
// token IDs as a single line:
//
//   TOKENS: <id> <id> <id> ...
//
// then exits 0.  The two-arm comparison (spec off vs spec on) is driven by
// benchmarks/spec_equivalence_check.sh, which runs this binary twice as
// separate processes and diffs the two TOKENS: lines.
//
// Speculation is toggled via the ECSIE_SPEC env var.  As of v1.0.114
// speculative decoding is DISABLED BY DEFAULT — it is opt-in only — so the
// harness must set ECSIE_SPEC=on explicitly to exercise the speculative path:
//   --spec off  → sets ECSIE_SPEC=off → speculative path forced off
//   --spec on   → sets ECSIE_SPEC=on  → speculative path opted in
//   (no --spec) → ECSIE_SPEC unset    → DEFAULT == spec off (correct path)
// Because each arm is its own process, ECSIE_SPEC is read once per process
// (see env_spec_depth in src/core/execution_loop.cpp).
//
// Usage:
//   ecsie_test_spec_equivalence --model <path.gguf> [--spec on|off]
//                               [--tokens N] [--prompt S]
//
// With no --spec argument the binary runs the true default (speculative
// decoding OFF), which must be byte-identical to an explicit --spec off run.
//
// Exit code: 0 on success (TOKENS: line printed), 1 on any setup error.

#include "ecsie/engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Portable env helpers: POSIX has setenv/unsetenv; this harness only builds on
// the Linux bench host, but guard anyway so it is obvious what is required.
//
// spec_mode: -1 = leave ECSIE_SPEC unset (the true engine default — spec off),
//             0 = ECSIE_SPEC=off (explicit kill-switch),
//             1 = ECSIE_SPEC=on  (opt in to the speculative path).
#if defined(_WIN32)
#  include <cstdlib>
static void set_spec(int spec_mode) {
    if      (spec_mode < 0) _putenv_s("ECSIE_SPEC", "");      // unset
    else if (spec_mode == 0) _putenv_s("ECSIE_SPEC", "off");  // kill-switch
    else                     _putenv_s("ECSIE_SPEC", "on");   // opt in
}
#else
#  include <cstdlib>
static void set_spec(int spec_mode) {
    if      (spec_mode < 0)  ::unsetenv("ECSIE_SPEC");          // engine default
    else if (spec_mode == 0) ::setenv("ECSIE_SPEC", "off", 1);  // kill-switch
    else                     ::setenv("ECSIE_SPEC", "on", 1);   // opt in
}
#endif

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt =
        "The quick brown fox jumps over the lazy dog. In a few sentences, "
        "explain why deterministic decoding matters for testing.";
    int  n_tokens = 64;
    int  spec     = -1;   // -1 = no --spec given (engine default), 0 = off, 1 = on

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) n_tokens   = std::atoi(argv[++i]);
        else if (a == "--prompt" && i + 1 < argc) prompt     = argv[++i];
        else if (a == "--spec"   && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "off") spec = 0;
            else if (v == "on")  spec = 1;
            else {
                std::fprintf(stderr,
                    "[spec_equiv] error: --spec must be 'on' or 'off'\n");
                return 1;
            }
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr,
            "[spec_equiv] error: --model <path.gguf> is required\n");
        return 1;
    }
    if (n_tokens < 1) n_tokens = 1;

    // Toggle speculation BEFORE the engine is constructed so the env var is
    // already in place when ExecutionLoop::step() reads it.  With no --spec
    // argument (spec == -1) ECSIE_SPEC is left unset — exercising the true
    // engine default, which (v1.0.114+) is speculative decoding OFF.
    set_spec(spec);

    const char* spec_label = (spec < 0) ? "default" : (spec ? "on" : "off");

    // ── Load model ────────────────────────────────────────────────────────────
    ecsie::Engine engine;
    try {
        engine.load_model(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[spec_equiv] model load error: %s\n", e.what());
        return 1;
    }
    if (!engine.is_loaded()) {
        std::fprintf(stderr, "[spec_equiv] model failed to load: %s\n",
                     model_path.c_str());
        return 1;
    }

    std::fprintf(stderr,
        "[spec_equiv] model=%s tokens=%d temp=0 (greedy) spec=%s\n",
        model_path.c_str(), n_tokens, spec_label);

    // ── One generate() — the single arm ───────────────────────────────────────
    ecsie::GenerateConfig cfg;
    cfg.max_new_tokens = n_tokens;
    cfg.temperature    = 0.0f;   // greedy / argmax → deterministic, no RNG
    cfg.top_p          = 1.0f;
    cfg.seed           = 1234;   // irrelevant at temp 0, fixed for clarity

    std::vector<int> tokens;
    try {
        tokens = engine.generate(prompt, cfg).tokens;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[spec_equiv] generate error: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "[spec_equiv] produced %zu tokens\n", tokens.size());

    // ── Emit the token IDs as a single TOKENS: line ───────────────────────────
    std::printf("TOKENS:");
    for (int t : tokens) std::printf(" %d", t);
    std::printf("\n");
    return 0;
}
