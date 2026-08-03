// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/latency_variance.cpp
//
// Per-token latency distribution benchmark.
// Records time-to-first-token (TTFT) and inter-token latency (ITL).
//
// Usage:
//   ecsie_bench_latency_variance [--model <path>] [--workload <path>]
//                                 [--out-latency <csv>]

#include "ecsie/engine.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Ms        = std::chrono::duration<double, std::milli>;

static double percentile(std::vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t idx = static_cast<std::size_t>(pct / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string workload_path;
    std::string out_latency = "-";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"       && i + 1 < argc) model_path    = argv[++i];
        else if (arg == "--workload"    && i + 1 < argc) workload_path = argv[++i];
        else if (arg == "--out-latency" && i + 1 < argc) out_latency   = argv[++i];
    }

    // ── Load workload ─────────────────────────────────────────────────────────
    struct Prompt { std::string text; int max_tokens; };
    std::vector<Prompt> prompts;
    int repetitions = 1;
    double temperature = 1.0;

    if (!workload_path.empty()) {
        try {
            std::ifstream f(workload_path);
            if (!f) throw std::runtime_error("cannot open " + workload_path);
            nlohmann::json wl = nlohmann::json::parse(f);
            repetitions = wl.value("repetitions", 1);
            temperature = wl.value("temperature", 1.0);
            for (auto& p : wl["prompts"]) {
                prompts.push_back({p["text"].get<std::string>(),
                                   p.value("max_tokens", 256)});
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[latency_variance] workload error: %s\n", e.what());
            return 1;
        }
    }

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
            std::fprintf(stderr, "[latency_variance] model load error: %s\n", e.what());
            return 1;
        }
    }

    // ── Open output ───────────────────────────────────────────────────────────
    std::FILE* out = (out_latency == "-") ? stdout : std::fopen(out_latency.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "[latency_variance] cannot open output %s\n", out_latency.c_str());
        return 1;
    }

    std::fprintf(out, "request_id,ttft_ms,mean_itl_ms,p50_itl_ms,p99_itl_ms\n");

    // ── Measurement loop ──────────────────────────────────────────────────────
    int request_id = 0;
    for (int rep = 0; rep < repetitions; ++rep) {
        for (const auto& p : prompts) {
            ecsie::GenerateConfig cfg;
            cfg.max_new_tokens = p.max_tokens;
            cfg.temperature    = static_cast<float>(temperature);

            double                ttft_ms  = 0.0;
            std::vector<double>   itl_ms_v;

            if (engine.is_loaded()) {
                const auto t_start  = Clock::now();
                bool       got_first = false;
                TimePoint  t_last;

                engine.generate_stream(p.text, [&](int /*tok*/, const std::string&) {
                    const auto now = Clock::now();
                    if (!got_first) {
                        ttft_ms   = Ms(now - t_start).count();
                        got_first = true;
                    } else {
                        itl_ms_v.push_back(Ms(now - t_last).count());
                    }
                    t_last = now;
                }, cfg);
            } else {
                // Synthetic: no model available — emit zero latencies.
                ttft_ms = 0.0;
            }

            const double mean_itl = itl_ms_v.empty() ? 0.0
                : [&] {
                    double s = 0.0;
                    for (double v : itl_ms_v) s += v;
                    return s / itl_ms_v.size();
                  }();
            const double p50 = percentile(itl_ms_v, 50.0);
            const double p99 = percentile(itl_ms_v, 99.0);

            std::fprintf(out, "%d,%.3f,%.3f,%.3f,%.3f\n",
                         request_id, ttft_ms, mean_itl, p50, p99);
            ++request_id;
        }
    }

    if (out != stdout) std::fclose(out);
    std::fprintf(stderr, "[latency_variance] wrote %d rows\n", request_id);
    return 0;
}

