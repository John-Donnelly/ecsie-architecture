// Minimal 8-thread sequential DRAM read-bandwidth probe.
// Decides whether the CPU-MoE's ~15.6 GB/s effective is the memory wall or an
// access-pattern (24-scattered-stream) artifact that a row-parallel layout could
// fix. Allocates BUF bytes, touches them, then N threads each sum a contiguous
// chunk; reports aggregate GB/s. Build: g++ -O3 -march=native -pthread.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <chrono>
int main(int argc, char** argv) {
    const std::size_t GB = 4;                       // buffer size
    const int NT = (argc > 1) ? std::atoi(argv[1]) : 8;
    const std::size_t n = GB * (1ull << 30) / sizeof(std::uint64_t);
    std::vector<std::uint64_t> buf(n);
    for (std::size_t i = 0; i < n; ++i) buf[i] = i;  // touch (fault in)
    const std::size_t per = n / NT;
    volatile std::uint64_t sink = 0;
    // warm + 3 timed passes
    for (int pass = 0; pass < 4; ++pass) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        std::vector<std::uint64_t> partial(NT, 0);
        for (int t = 0; t < NT; ++t)
            th.emplace_back([&, t] {
                std::uint64_t s = 0;
                const std::size_t b = (std::size_t)t * per, e = (t == NT-1) ? n : b + per;
                for (std::size_t i = b; i < e; ++i) s += buf[i];
                partial[t] = s;
            });
        for (auto& x : th) x.join();
        for (auto p : partial) sink += p;
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (pass > 0)
            std::printf("NT=%d  read %.1f GB in %.3f s  => %.1f GB/s\n",
                        NT, (double)GB, sec, GB / sec);
    }
    return (int)(sink & 1);
}
