# ECSIE System Architecture

## Overview

ECSIE is structured as a layered execution engine. Each layer has a single responsibility and communicates through well-defined interfaces.

## Execution Model

ECSIE does not execute models layer-by-layer in a fixed sequence. Instead, it compiles a loaded model into an **execution graph** and runs inference as a series of scheduled graph traversals.

### Execution Graph: G = (L, E, R, M)

| Symbol | Description |
|--------|-------------|
| `L` | Ordered set of transformer layers |
| `E` | Expert compute nodes (sparse MoE units) |
| `R` | Routing function (pluggable policy) |
| `M` | Memory residency state (VRAM / RAM / mmap) |

The graph is inferred at model load time by `src/loader/graph_inference.cpp` using tensor structure metadata from `src/loader/tensor_scanner.cpp`.

## Layer Responsibilities

| Layer | Path | Responsibility |
|-------|------|----------------|
| Runtime Core | `src/core/` | Execution loop, scheduler, policy dispatch |
| MoE Graph | `src/moe/` | Expert nodes, router logic, graph traversal |
| Memory | `src/memory/` | VRAM/RAM/mmap allocation and movement |
| Entropy Controller | `src/entropy/` | H_t computation, policy transitions |
| Batching | `src/batching/` | Continuous batching, token queue |
| Speculative | `src/speculative/` | Draft model, verifier, acceptance |
| Loader | `src/loader/` | GGUF parsing, tensor scanning, graph build |
| Backend | `src/backend/` | CUDA kernel dispatch |
| API | `src/api/` | HTTP server, request/response handling |

## Entropy Signal

The entropy controller (`src/entropy/controller.cpp`) computes:

```
H_t = alpha * R_t + beta * L_t + gamma * (1 - A_t) + delta * (1 - C_t)
```

This signal drives policy decisions in `src/entropy/policy.cpp`, which feeds back into the scheduler and speculative decoder.

### Entropy subsystem headers (Phase 3)

| Header | Class / API | Purpose |
|--------|-------------|---------|
| `include/ecsie/entropy.hpp` | `EntropyState` | Holds `R_t`, `L_t`, `A_t`, `C_t`, `H_t` |
| `include/ecsie/controller.hpp` | `EntropyController` | `step(R_t,L_t,A_t,C_t)→H_t`; caches `EntropyState` |
| `include/ecsie/metrics.hpp` | `LatencyWindow`, `routing_entropy()`, `latency_variance_normalised()` | Fixed-capacity `std::deque` latency window; free metric functions |

`LatencyWindow::variance_normalised()` returns `σ²/σ²_max` clamped to `[0, 1]`, used directly as `L_t`.

### Router factory pattern (Phase 3)

`include/ecsie/router.hpp` exposes two factory functions:

```cpp
std::unique_ptr<IRouter> make_topk_router(int top_k, int num_experts);
std::unique_ptr<IRouter> make_cached_router(int top_k, int num_experts,
                                            std::size_t cache_capacity);
```

All router types share the `IRouter` interface (`route(RouterInput)`, `last_routing_entropy()`).

## Memory Hierarchy

```
VRAM  ── hot experts (currently executing or recently active)
 RAM  ── warm experts (prefetched, likely next)
mmap  ── cold experts (on-disk, loaded on demand)
```

Movement between tiers is triggered by the entropy policy, not by a static LRU.

### Phase 3 Memory API (public headers, v0.3.x)

| Header | Class | Purpose |
|--------|-------|---------|
| `include/ecsie/vram_pool.hpp` | `VramPool` | Pimpl facade; CPU stub + CUDA bump-allocator behind `ECSIE_CUDA_ENABLED` |
| `include/ecsie/ram_pool.hpp` | `RamPool` | 64-byte-aligned slab with address-sorted free-list coalescing |
| `include/ecsie/expert_cache.hpp` | `ExpertCache` | LRU `std::list` + `std::unordered_map`; tracks VRAM/RAM bytes; `snapshot()→MemoryStats` |
| `include/ecsie/memory.hpp` | `MemoryAllocator`, `MemoryStats` | Unified allocator; RAM tier delegates to `RamPool`; `stats()` returns aggregate snapshot |

The `RamPool` slab is allocated with `std::aligned_alloc(64, …)` so every pointer returned satisfies 64-byte alignment regardless of request size.

## Data Flow

```
Request
  → API layer (server.cpp)
  → Batch scheduler (batch_scheduler.cpp)
  → Entropy controller (controller.cpp)
  → Scheduler (scheduler.cpp)
  → MoE router (router_*.cpp)
  → Expert dispatch (expert_dispatch.cu)
  → CUDA backend (cuda_kernels.cu)
  → Speculative verifier (verifier.cpp)
  → Response
```

## Phase 4 — Public API & Dispatch Chain (v0.4.x)

Phase 4 promotes the batching and speculative-decoding subsystems to first-class public headers and wires them through the `Engine` API.

### Public headers added in Phase 4

| Header | Class / API | Purpose |
|--------|-------------|---------|
| `include/ecsie/batch_scheduler.hpp` | `BatchScheduler`, `Request`, `Batch` | Priority-sorted request queue; FIFO within equal priority (`std::upper_bound`); `next_batch()` assembles a flat token tensor |
| `include/ecsie/token_queue.hpp` | `TokenQueue`, `SequenceState` | Per-sequence KV-cache position tracker; `append_token()`, `finish()`, `remove()` lifecycle |

### Engine dispatch chain

`Engine::generate()` and `Engine::generate_stream()` (in `src/core/engine.cpp`) follow this dispatch path:

1. **Tokenise** — whitespace-hash tokeniser maps prompt words to integer IDs.
2. **Enqueue** — `BatchScheduler::enqueue(Request)` inserts with FIFO priority ordering.
3. **Schedule** — `BatchScheduler::next_batch()` assembles the next flat batch.
4. **Track sequences** — each sequence is registered in `TokenQueue`.
5. **Decode** — one token per sequence is produced (EOS stub in CI; real forward pass when GPU is present).
6. **Stream** — `generate_stream()` invokes the `TokenCallback` for each produced token.

### DraftModel n-gram strategy

`DraftModel` (in `src/speculative/draft_model.cpp`) maintains an n-gram frequency table trained via `observe(tokens)`. The `generate(context, depth)` method returns the most-frequent continuation for the last context token; when no n-gram is available it falls back to `(prev+1) % 32000`. All returned log-probabilities are set to `ln(0.5) ≈ -0.693`.

### Verifier rejection sampling

`Verifier` (in `src/speculative/verifier.cpp`) implements the standard speculative-decoding acceptance rule:

```
accept token i  if  U[0,1) < min(1,  p_target[i] / p_draft[i])
```

When `target_log_probs` is empty the verifier synthesises target probabilities by adding uniform noise `U[-0.3, 0.3)` to the draft log-probabilities. The first rejected draft token is saved as `bonus_token`.

### Profiler wiring

`ExecutionLoop::step()` (in `src/core/execution_loop.cpp`) wraps the router dispatch in a `ScopedProfile{"routing_step"}` region. The resulting timing is retrievable via `Profiler::instance().last_ms("routing_step")` and is copied into `StepResult::routing_step_ms`.

The `Profiler::last_ms(std::string_view)` method (added in v0.4.13) returns the most-recently recorded duration for a named region, or `-1.0` if unmeasured.

