// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/gpu_utilisation.cpp
//
// GPU utilisation sampler. Polls nvidia-smi at 100ms intervals during
// a benchmark run and correlates utilisation with entropy signal.
//
// Usage:
//   ecsie_bench_gpu_utilisation [--model <path>] [--workload <path>]
//                                [--out-gpu <csv>] [--poll-ms <int>]

#include "ecsie/engine.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock  = std::chrono::steady_clock;
using Ms     = std::chrono::duration<double, std::milli>;

static double epoch_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        system_clock::now().time_since_epoch()).count();
}

// ── nvidia-smi polling thread ─────────────────────────────────────────────────

struct GpuSample { double timestamp_ms; double util_pct; double mem_used_mb; };

struct GpuPoller {
    std::vector<GpuSample>  samples;
    std::atomic<bool>       stop_flag{false};
    int                     poll_interval_ms{100};

    void run() {
        while (!stop_flag.load()) {
            GpuSample s;
            s.timestamp_ms = epoch_ms();
            s.util_pct     = 0.0;
            s.mem_used_mb  = 0.0;

            // Try nvidia-smi; gracefully handle absence.
            FILE* pipe = popen(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                "--format=csv,noheader,nounits 2>/dev/null",
                "r");
            if (pipe) {
                double util = 0.0, mem = 0.0;
                if (std::fscanf(pipe, "%lf , %lf", &util, &mem) == 2) {
                    s.util_pct    = util;
                    s.mem_used_mb = mem;
                }
                pclose(pipe);
            }

            samples.push_back(s);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(poll_interval_ms));
        }
    }
};

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string workload_path;
    std::string out_gpu    = "-";
    int         poll_ms    = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"    && i + 1 < argc) model_path    = argv[++i];
        else if (arg == "--workload" && i + 1 < argc) workload_path = argv[++i];
        else if (arg == "--out-gpu"  && i + 1 < argc) out_gpu       = argv[++i];
        else if (arg == "--poll-ms"  && i + 1 < argc) poll_ms       = std::stoi(argv[++i]);
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
            std::fprintf(stderr, "[gpu_utilisation] workload error: %s\n", e.what());
            return 1;
        }
    }

    if (prompts.empty()) {
        prompts.push_back({"Hello, world!", 16});
    }

    // ── Load engine ───────────────────────────────────────────────────────────
    ecsie::Engine engine;
    if (!model_path.empty()) {
        try {
            engine.load_model(model_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[gpu_utilisation] model load error: %s\n", e.what());
            return 1;
        }
    }

    // ── Start GPU poller ──────────────────────────────────────────────────────
    GpuPoller poller;
    poller.poll_interval_ms = poll_ms;
    std::thread poll_thread([&poller] { poller.run(); });

    // ── Inference loop ────────────────────────────────────────────────────────
    for (int rep = 0; rep < repetitions; ++rep) {
        for (const auto& p : prompts) {
            ecsie::GenerateConfig cfg;
            cfg.max_new_tokens = p.max_tokens;
            cfg.temperature    = static_cast<float>(temperature);
            if (engine.is_loaded()) {
                engine.generate(p.text, cfg);
            }
        }
    }

    // ── Stop poller ───────────────────────────────────────────────────────────
    poller.stop_flag.store(true);
    poll_thread.join();

    // ── Write output ──────────────────────────────────────────────────────────
    std::FILE* out = (out_gpu == "-") ? stdout : std::fopen(out_gpu.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "[gpu_utilisation] cannot open output %s\n", out_gpu.c_str());
        return 1;
    }

    std::fprintf(out, "timestamp_ms,gpu_util_pct,mem_used_mb,H_t\n");
    for (const auto& s : poller.samples) {
        // H_t is not currently exposed per-sample; emit 0.0 as placeholder.
        std::fprintf(out, "%.3f,%.1f,%.1f,0.0\n",
                     s.timestamp_ms, s.util_pct, s.mem_used_mb);
    }

    if (out != stdout) std::fclose(out);
    std::fprintf(stderr, "[gpu_utilisation] wrote %zu samples\n", poller.samples.size());
    return 0;
}

