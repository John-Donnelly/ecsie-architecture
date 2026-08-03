// ECSIE — Entropy-Controlled Sparse Inference Engine
// benchmarks/ternary_recon_bench.cpp
//
// Standalone bench for the ternary + top-2 Q4 KV codec.  Generates synthetic
// row distributions matching what attention sees in practice (sparse outliers
// on a roughly Gaussian background) and reports reconstruction error +
// cosine similarity at the codec's chosen compression ratio.

#include "ecsie/ternary_kv.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    constexpr int    kv_dim    = 512;   // Qwen3.6 head_dim × num_kv_heads
    constexpr int    num_rows  = 1024;
    float  noise_std   = 0.2f;
    float  outlier_p   = 0.004f; // 0.4 % → ~2 outliers per row
    float  outlier_std = 2.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--outlier-p"  && i + 1 < argc) outlier_p   = static_cast<float>(std::stof(argv[++i]));
        else if (a == "--outlier-std" && i + 1 < argc) outlier_std = static_cast<float>(std::stof(argv[++i]));
        else if (a == "--noise-std"   && i + 1 < argc) noise_std   = static_cast<float>(std::stof(argv[++i]));
    }

    std::mt19937 rng(12345);
    std::normal_distribution<float>       noise(0.0f, noise_std);
    std::normal_distribution<float>       outlier(0.0f, outlier_std);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    double sum_relerr = 0.0, sum_cos = 0.0;
    double min_cos = 1.0, max_relerr = 0.0;

    std::vector<float> row(kv_dim);
    for (int r = 0; r < num_rows; ++r) {
        // Generate row: Gaussian background with sparse outliers.
        for (int i = 0; i < kv_dim; ++i) {
            row[i] = (u01(rng) < outlier_p) ? outlier(rng) : noise(rng);
        }
        const auto s = ecsie::ternary_top2_roundtrip_stats(row.data(), kv_dim);
        sum_relerr += s.relative_error;
        sum_cos    += s.cosine_sim;
        if (s.cosine_sim < min_cos)        min_cos    = s.cosine_sim;
        if (s.relative_error > max_relerr) max_relerr = s.relative_error;
    }

    const double avg_relerr = sum_relerr / num_rows;
    const double avg_cos    = sum_cos / num_rows;

    const std::size_t bytes_fp16 = static_cast<std::size_t>(kv_dim) * 2;
    const std::size_t bytes_tk   = ecsie::ternary_top2_encoded_bytes(kv_dim);
    const double ratio = static_cast<double>(bytes_fp16) / bytes_tk;

    std::printf("rows=%d  dim=%d  outlier_p=%.2f  outlier_std=%.1f  noise_std=%.2f\n",
                num_rows, kv_dim, outlier_p, outlier_std, noise_std);
    std::printf("avg_rel_l2_err = %.4f  worst = %.4f\n", avg_relerr, max_relerr);
    std::printf("avg_cos_sim    = %.4f  worst = %.4f\n", avg_cos, min_cos);
    std::printf("bytes/row: fp16 = %zu, ternary+top2 = %zu  (compression %.2fx)\n",
                bytes_fp16, bytes_tk, ratio);
    return 0;
}
