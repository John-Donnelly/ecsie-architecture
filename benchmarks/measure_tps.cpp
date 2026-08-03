// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/measure_tps.cpp
//
// Tokens-per-second throughput benchmark.
// Drives the engine with a workload file and records TPS at each step.
// After every generate() call a per-phase latency breakdown is printed to
// stderr so it can be captured independently from the CSV stream.
//
// Usage:
//   ecsie_bench_measure_tps [--model <path>] [--workload <path>]
//                            [--out-tps <csv>]

#include "ecsie/engine.hpp"
#include "ecsie/profiler.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
// POSIX setenv/unsetenv are absent on Windows; map onto the CRT's _putenv_s.
// These per-arm env toggles are re-read by the engine each call (see below).
static int setenv(const char* name, const char* value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
static int unsetenv(const char* name) {
    return _putenv_s(name, "");  // empty value removes the variable on MSVC CRT
}
#endif

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Ms        = std::chrono::duration<double, std::milli>;

static double elapsed_ms(TimePoint t0) {
    return Ms(Clock::now() - t0).count();
}

static double epoch_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        system_clock::now().time_since_epoch()).count();
}

#ifdef ECSIE_CUDA_ENABLED
extern "C" int ecsie_q4k_batched_selftest(int M, int K, int N);
extern "C" int ecsie_cublas_batched_bench(int Wrows, int K, int N);
#endif

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string workload_path;
    std::string out_tps = "-";   // "-" = stdout

#ifdef ECSIE_CUDA_ENABLED
    // Gated correctness check for the batched Q4_K prefill GEMM (no model
    // needed): ECSIE_Q4K_BATCHED_TEST=1 → run a few shapes vs the M=1 kernel.
    if (const char* e = std::getenv("ECSIE_Q4K_BATCHED_TEST"); e && *e && *e != '0') {
        int rc = 0;
        rc |= ecsie_q4k_batched_selftest(64,  256, 1);
        rc |= ecsie_q4k_batched_selftest(64,  256, 7);
        rc |= ecsie_q4k_batched_selftest(512, 4096, 33);
        rc |= ecsie_q4k_batched_selftest(4096, 4096, 100);
        rc |= ecsie_q4k_batched_selftest(4096, 4096, 2789);   // real SSM gate/out @ pf_n60 len
        std::fprintf(stderr, "[q4k_batched_selftest] overall %s\n", rc == 0 ? "PASS" : "FAIL");
        // Is cuBLAS tiled GEMM the answer where custom GEMV kernels fail?
        ecsie_cublas_batched_bench(4096, 4096, 2789);   // SSM gate/out (Q4K→FP16)
        ecsie_cublas_batched_bench(8192, 4096, 2789);   // SSM qkv (Q6K→FP16)
        return rc;
    }
#endif

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model"    && i + 1 < argc) model_path    = argv[++i];
        else if (arg == "--workload" && i + 1 < argc) workload_path = argv[++i];
        else if (arg == "--out-tps"  && i + 1 < argc) out_tps       = argv[++i];
    }

    // ── Load workload ─────────────────────────────────────────────────────────
    struct Prompt { std::string text; int max_tokens; };
    std::vector<Prompt> prompts;
    int repetitions = 1;
    int batch_size  = 1;
    double temperature = 1.0;

    if (!workload_path.empty()) {
        try {
            std::ifstream f(workload_path);
            if (!f) throw std::runtime_error("cannot open " + workload_path);
            nlohmann::json wl = nlohmann::json::parse(f);
            repetitions  = wl.value("repetitions", 1);
            batch_size   = wl.value("batch_size", 1);
            temperature  = wl.value("temperature", 1.0);
            for (auto& p : wl["prompts"]) {
                prompts.push_back({p["text"].get<std::string>(),
                                   p.value("max_tokens", 256)});
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[measure_tps] workload error: %s\n", e.what());
            return 1;
        }
    }
    // Allow CLI override (Phase E benchmarking).
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
    }
    if (batch_size < 1) batch_size = 1;

    if (prompts.empty()) {
        prompts.push_back({"Hello, world!", 16});
        prompts.push_back({"What is 2+2?", 8});
    }

    // ── Load engine ───────────────────────────────────────────────────────────
    ecsie::Engine engine;
    if (!model_path.empty()) {
        try {
            engine.load_model(model_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[measure_tps] model load error: %s\n", e.what());
            return 1;
        }
    }

    // ── Perplexity mode (accuracy harness) ─────────────────────────────────────
    // ECSIE_PERPLEXITY_CORPUS=<text file> → compute the loaded model's teacher-
    // forced perplexity over the corpus and exit (before the throughput bench).
    // ECSIE_PERPLEXITY_CTX sets the chunk size (default 512).  Quant/spec-config-
    // agnostic — it gates lossy levers (n6h8 quant, ternary-KV, typical-accept).
    // Run WITHOUT ECSIE_CPU_EXPERTS for a faster GPU prefill (PPL is path-agnostic).
    if (const char* cpath = std::getenv("ECSIE_PERPLEXITY_CORPUS")) {
        if (!engine.is_loaded()) {
            std::fprintf(stderr, "[perplexity] model not loaded\n");
            return 1;
        }
        std::ifstream cf(cpath, std::ios::binary);
        if (!cf) {
            std::fprintf(stderr, "[perplexity] cannot open corpus %s\n", cpath);
            return 1;
        }
        cf.seekg(0, std::ios::end);
        const std::streamoff sz = cf.tellg();
        cf.seekg(0, std::ios::beg);
        std::string corpus(static_cast<std::size_t>(sz > 0 ? sz : 0), '\0');
        if (sz > 0) cf.read(&corpus[0], sz);
        int ctx = 512;
        if (const char* c = std::getenv("ECSIE_PERPLEXITY_CTX")) {
            const int v = std::atoi(c);
            if (v >= 2) ctx = v;
        }
        const TimePoint t0 = Clock::now();
        long ntok = 0;
        const double ppl = engine.perplexity(corpus, ctx, &ntok);
        const double secs = elapsed_ms(t0) / 1000.0;
        std::fprintf(stderr,
            "[perplexity] model=%s ctx=%d scored_tokens=%ld  PPL=%.4f  (%.1fs)\n",
            model_path.c_str(), ctx, ntok, ppl, secs);
        std::printf("PPL=%.4f scored_tokens=%ld ctx=%d\n", ppl, ntok, ctx);
        return 0;
    }

    // ── Open output ───────────────────────────────────────────────────────────
    std::FILE* out = (out_tps == "-") ? stdout : std::fopen(out_tps.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "[measure_tps] cannot open output %s\n", out_tps.c_str());
        return 1;
    }

    std::fprintf(out, "step,timestamp_ms,tokens,tps\n");

    std::fprintf(stderr, "[measure_tps] batch_size=%d (Phase E if >1)\n", batch_size);

    // ── In-process relaxed-acceptance sweep (v1.0.214) ─────────────────────────
    // ECSIE_SPEC_ACCEPT_SWEEP="strict,ratio:0.5,ratio:0.3,..." runs each accept
    // mode back-to-back on the SAME loaded model (one slow drvfs load instead of
    // one per arm).  spec_accept_cfg() re-reads the env per call, so setenv()
    // here switches the accept rule between generate() runs.  Per-mode spec stats
    // come from delta-ing the engine's cumulative drafted/accepted counters.
    // Keep ECSIE_SPEC / ECSIE_SPEC_DEPTH / ECSIE_SPEC_BATCHED fixed in the
    // environment (they are read once at process start); only the accept rule
    // varies across the sweep.
    if (const char* sw = std::getenv("ECSIE_SPEC_ACCEPT_SWEEP")) {
        std::vector<std::string> modes;
        { std::string cur; for (const char* c = sw; *c; ++c) {
            if (*c == ',') { if (!cur.empty()) modes.push_back(cur); cur.clear(); }
            else cur.push_back(*c);
          } if (!cur.empty()) modes.push_back(cur); }
        if (modes.empty()) modes.push_back("strict");
        if (prompts.empty() || !engine.is_loaded()) {
            std::fprintf(stderr, "[sweep] no prompt or model not loaded\n");
            return 1;
        }
        const Prompt& P = prompts[0];
        int prev_drafted = 0, prev_accepted = 0;
        std::fprintf(stderr, "[sweep] %zu modes, prompt max_tokens=%d temp=%.2f\n",
                     modes.size(), P.max_tokens, temperature);
        for (const auto& m : modes) {
#ifdef _WIN32
            _putenv_s("ECSIE_SPEC_ACCEPT", m.c_str());
#else
            setenv("ECSIE_SPEC_ACCEPT", m.c_str(), 1);
#endif
            ecsie::GenerateConfig cfg;
            cfg.max_new_tokens = P.max_tokens;
            cfg.temperature    = static_cast<float>(temperature);
            const auto t0 = Clock::now();
            auto r         = engine.generate(P.text, cfg);
            const double wall = elapsed_ms(t0);
            auto lr        = engine.latency_report(wall, r.completion_tokens);
            const int dd   = lr.speculative_total_drafted  - prev_drafted;
            const int da   = lr.speculative_total_accepted - prev_accepted;
            prev_drafted   = lr.speculative_total_drafted;
            prev_accepted  = lr.speculative_total_accepted;
            const double rate    = (dd > 0) ? static_cast<double>(da) / dd : 0.0;
            const double eff_tps = (wall > 0.0)
                ? r.completion_tokens * 1000.0 / wall : 0.0;
            std::fprintf(stderr,
                "[sweep] mode=%-12s drafted=%d accepted=%d rate=%.3f "
                "tokens=%d wall_ms=%.0f eff_tps=%.2f mean_step_ms=%.2f\n",
                m.c_str(), dd, da, rate, r.completion_tokens, wall, eff_tps,
                lr.mean_step_ms);
        }
        if (out != stdout) std::fclose(out);
        return 0;
    }

    // ── In-process head-norm placement sweep (v1.0.214) ────────────────────────
    // ECSIE_EAGLE3_NORM_SWEEP="default,no_fc,no_post,fusion,hidden_first,
    // no_fc+no_post" runs each norm-placement config back-to-back on ONE model
    // load, at STRICT acceptance, to find a placement that raises the head's TRUE
    // greedy-match rate (lossless).  eagle3_draft.cpp re-reads these flags per
    // step, so setenv() here switches placement between generate() runs.  The
    // engine's drafted/accepted counters are per-generate (reset each call), so
    // read them directly.
    if (const char* nsw = std::getenv("ECSIE_EAGLE3_NORM_SWEEP")) {
        const char* const NORM_VARS[] = {
            "ECSIE_EAGLE3_NO_FC_NORM",    "ECSIE_EAGLE3_NO_POST_ATTN",
            "ECSIE_EAGLE3_NO_FINAL_NORM", "ECSIE_EAGLE3_FUSION_NORM",
            "ECSIE_EAGLE3_CONCAT_ORDER" };
        auto apply_key = [&](const std::string& k) {
            if      (k == "no_fc")        setenv("ECSIE_EAGLE3_NO_FC_NORM",    "1",  1);
            else if (k == "no_post")      setenv("ECSIE_EAGLE3_NO_POST_ATTN",  "1",  1);
            else if (k == "no_final")     setenv("ECSIE_EAGLE3_NO_FINAL_NORM", "1",  1);
            else if (k == "fusion")       setenv("ECSIE_EAGLE3_FUSION_NORM",   "on", 1);
            else if (k == "hidden_first") setenv("ECSIE_EAGLE3_CONCAT_ORDER",
                                                 "hidden_first", 1);
            // "default" / unknown → leave cleared (no-op)
        };
        setenv("ECSIE_SPEC_ACCEPT", "strict", 1);   // measure the lossless rate
        std::vector<std::string> combos;
        { std::string cur; for (const char* c = nsw; *c; ++c) {
            if (*c == ',') { if (!cur.empty()) combos.push_back(cur); cur.clear(); }
            else cur.push_back(*c);
          } if (!cur.empty()) combos.push_back(cur); }
        if (combos.empty() || prompts.empty() || !engine.is_loaded()) {
            std::fprintf(stderr, "[normsweep] nothing to do\n");
            return 1;
        }
        const Prompt& P = prompts[0];
        std::fprintf(stderr, "[normsweep] %zu combos, max_tokens=%d temp=%.2f\n",
                     combos.size(), P.max_tokens, temperature);
        for (const auto& combo : combos) {
            for (const char* v : NORM_VARS) unsetenv(v);
            { std::string cur; const std::string s = combo + "+";
              for (char c : s) {
                if (c == '+') { if (!cur.empty()) apply_key(cur); cur.clear(); }
                else cur.push_back(c);
              } }
            ecsie::GenerateConfig cfg;
            cfg.max_new_tokens = P.max_tokens;
            cfg.temperature    = static_cast<float>(temperature);
            const auto t0 = Clock::now();
            auto r         = engine.generate(P.text, cfg);
            const double wall = elapsed_ms(t0);
            auto lr        = engine.latency_report(wall, r.completion_tokens);
            const int dd   = lr.speculative_total_drafted;
            const int da   = lr.speculative_total_accepted;
            const double rate = (dd > 0) ? static_cast<double>(da) / dd : 0.0;
            std::fprintf(stderr,
                "[normsweep] combo=%-16s drafted=%d accepted=%d rate=%.3f "
                "tokens=%d mean_step_ms=%.1f\n",
                combo.c_str(), dd, da, rate, r.completion_tokens, lr.mean_step_ms);
        }
        if (out != stdout) std::fclose(out);
        return 0;
    }

    // ── Measurement loop ──────────────────────────────────────────────────────
    int step = 0;
    for (int rep = 0; rep < repetitions; ++rep) {
        // Chunk the prompts into batches of size `batch_size`.  Each chunk
        // becomes one Engine::generate_batch() call when batch_size > 1.
        for (std::size_t base = 0; base < prompts.size(); base += static_cast<std::size_t>(batch_size)) {
            const std::size_t end = std::min(prompts.size(),
                                              base + static_cast<std::size_t>(batch_size));
            const int N = static_cast<int>(end - base);

            double ts = epoch_ms();
            int    n_tokens_total = 0;
            double tps      = 0.0;
            double wall_ms  = 0.0;

            if (engine.is_loaded()) {
                const auto t0 = Clock::now();
                if (N == 1) {
                    ecsie::GenerateConfig cfg;
                    cfg.max_new_tokens = prompts[base].max_tokens;
                    cfg.temperature    = static_cast<float>(temperature);
                    auto result    = engine.generate(prompts[base].text, cfg);
                    n_tokens_total = result.completion_tokens;
                    // Optional: dump generated token IDs for equivalence
                    // checks (e.g. fused-path-on vs -off correctness).
                    if (const char* dp = std::getenv("ECSIE_DUMP_TOKENS")) {
                        if (std::FILE* tf = std::fopen(dp, "w")) {
                            for (int tk : result.tokens)
                                std::fprintf(tf, "%d\n", tk);
                            std::fclose(tf);
                        }
                        const std::string tp = std::string(dp) + ".text";
                        if (std::FILE* xf = std::fopen(tp.c_str(), "w")) {
                            std::fprintf(xf, "%s\n", result.text.c_str());
                            std::fclose(xf);
                        }
                    }
                } else {
                    std::vector<ecsie::BatchGenerateRequest> batch;
                    batch.reserve(static_cast<std::size_t>(N));
                    for (std::size_t i = base; i < end; ++i) {
                        ecsie::BatchGenerateRequest bgr;
                        bgr.prompt = prompts[i].text;
                        bgr.cfg.max_new_tokens = prompts[i].max_tokens;
                        bgr.cfg.temperature    = static_cast<float>(temperature);
                        batch.push_back(std::move(bgr));
                    }
                    auto results = engine.generate_batch(batch);
                    for (const auto& r : results) n_tokens_total += r.completion_tokens;
                    // Optional per-sequence token/text dump for batched-path
                    // correctness checks (compare default vs ECSIE_BATCHED_FFN=1).
                    if (const char* dp = std::getenv("ECSIE_DUMP_TOKENS")) {
                        for (std::size_t i = 0; i < results.size(); ++i) {
                            const std::string fp =
                                std::string(dp) + ".seq" + std::to_string(i);
                            if (std::FILE* tf = std::fopen(fp.c_str(), "w")) {
                                for (int tk : results[i].tokens)
                                    std::fprintf(tf, "%d\n", tk);
                                std::fclose(tf);
                            }
                            if (std::FILE* xf = std::fopen((fp + ".text").c_str(), "w")) {
                                std::fprintf(xf, "%s\n", results[i].text.c_str());
                                std::fclose(xf);
                            }
                        }
                    }
                }
                wall_ms = elapsed_ms(t0);
                tps     = (wall_ms > 0.0) ? (n_tokens_total * 1000.0 / wall_ms) : 0.0;
            } else {
                for (std::size_t i = base; i < end; ++i)
                    n_tokens_total += prompts[i].max_tokens;
                tps = 0.0;
            }

            std::fprintf(out, "%d,%.3f,%d,%.2f\n", step, ts, n_tokens_total, tps);

            if (engine.is_loaded()) {
                auto lr = engine.latency_report(wall_ms, n_tokens_total);
                // warm_step_rate = 1000 / mean_step_ms — the steady-state
                // throughput at batch=1 measured per decode step, independent
                // of cold-start / prefill cost amortisation.  This is the
                // metric to compare against goals like "30 TPS at batch=1".
                const double warm_step_rate =
                    (lr.mean_step_ms > 0.0) ? (1000.0 / lr.mean_step_ms) : 0.0;
                std::fprintf(stderr,
                    "[latency] step=%d (batch=%d)  tps=%.2f  warm_step_rate=%.2f  "
                    "steps=%zu  mean_step=%.3f ms  p99_step=%.3f ms\n"
                    "          attn_norm=%.3f ms  attention=%.3f ms  "
                    "ffn_norm=%.3f ms  moe_dispatch=%.3f ms\n",
                    step, N, lr.tps, warm_step_rate,
                    lr.step_count,
                    lr.mean_step_ms, lr.p99_step_ms,
                    lr.mean_attn_norm_ms, lr.mean_attention_ms,
                    lr.mean_ffn_norm_ms, lr.mean_moe_dispatch_ms);
                if (lr.speculative_steps_with_draft > 0) {
                    std::fprintf(stderr,
                        "          spec: steps=%d drafted=%d accepted=%d "
                        "rate=%.3f mean_depth=%.2f\n",
                        lr.speculative_steps_with_draft,
                        lr.speculative_total_drafted,
                        lr.speculative_total_accepted,
                        lr.speculative_accept_rate,
                        lr.speculative_mean_depth);
                }
                if (lr.fused_multi_q4k_layers_seen > 0) {
                    std::fprintf(stderr,
                        "          fused_multi_q4k: layers_used=%llu/%llu\n",
                        static_cast<unsigned long long>(lr.fused_multi_q4k_layers_used),
                        static_cast<unsigned long long>(lr.fused_multi_q4k_layers_seen));
                }
                if (lr.fused_multi_ternary_layers_seen > 0) {
                    std::fprintf(stderr,
                        "          fused_multi_ternary: layers_used=%llu/%llu\n",
                        static_cast<unsigned long long>(lr.fused_multi_ternary_layers_used),
                        static_cast<unsigned long long>(lr.fused_multi_ternary_layers_seen));
                }
                if (lr.lm_head_sparse_calls_seen > 0) {
                    std::fprintf(stderr,
                        "          lm_head_sparse: taken=%llu/%llu\n",
                        static_cast<unsigned long long>(lr.lm_head_sparse_calls_taken),
                        static_cast<unsigned long long>(lr.lm_head_sparse_calls_seen));
                }
                // ECSIE_PROFILE_SUMMARY=1: dump every profiler timer (name, n,
                // mean, p99) for the just-finished generate before the reset.
                // Used to attribute the step's GPU cost (attention vs moe vs
                // lm_head/draft residual) and confirm forwards-per-step via n.
                if (std::getenv("ECSIE_PROFILE_SUMMARY"))
                    ecsie::Profiler::instance().summarise();
                engine.reset_profiler();
            }
            ++step;
        }
    }

    if (out != stdout) std::fclose(out);
    std::fprintf(stderr, "[measure_tps] wrote %d rows\n", step);
    return 0;
}

