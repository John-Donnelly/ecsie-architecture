# ECSIE — Architecture & Benchmarks

**Entropy-Controlled Sparse Inference Engine.** A model-agnostic execution
runtime for sparse Mixture-of-Experts language models.

This repository is the **public design record**: the architecture, the public
API surface, the benchmark harness, and the measured results. The engine
implementation itself is not open source and is not included here.

---

## The idea

Most inference stacks were built for dense models and have had MoE support
grafted on. ECSIE starts from the opposite premise: **LLM execution is a
dynamic sparse computation graph, not a static tensor pipeline.**

Every model is represented as

```
G = (L, E, R, M)
```

| | |
| --- | --- |
| `L` | layers |
| `E` | experts — sparse compute units |
| `R` | routing functions |
| `M` | memory state — VRAM / RAM / mmap residency |

Decoupling weights from execution semantics behind this abstraction is what
makes the runtime model-agnostic: a new MoE architecture is a new graph
description, not a new code path.

The **entropy signal** is the control input. Routing entropy at each step is a
measurable proxy for how uncertain the model currently is, and the scheduler
uses it to decide how aggressively to speculate, how much to prefetch, and what
to keep resident. Scheduling reacts to what the model is actually doing rather
than to a fixed policy set at load time.

## What's here

| Path | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Execution model, layer responsibilities, entropy subsystem, memory hierarchy, dispatch chain |
| [`docs/api.md`](docs/api.md) | The public header surface |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Reproducibility guide — workload profiles, ablation configs, determinism |
| [`docs/native-windows-tuning.md`](docs/native-windows-tuning.md) | Platform tuning notes |
| [`benchmarks/`](benchmarks/) | The measurement harness: drivers, ablation manifests, A/B matrices |
| [`benchmarks/results/`](benchmarks/results/) | Raw measured output from the runs behind the numbers below |

The benchmark drivers link against ECSIE's public headers only. They are
published so the methodology can be inspected — they will not build without the
engine.

## Measured envelope

Consumer-grade 8 GB VRAM, Qwen 3.6 35B A3B:

| Mode | tokens/sec |
| --- | --- |
| Stable workload | 22–35 |
| Peak — high-acceptance speculative decoding | 38–55 |
| High-entropy stress | 15–22 |

The raw logs behind these figures are in `benchmarks/results/`, including the
ablation runs that isolate each contribution. Machine-local paths have been
replaced with environment variables; nothing else has been edited.

## Design claims worth arguing with

- **Sparsity is a scheduling problem, not a kernel problem.** Most of the win
  comes from deciding what to compute and what to keep resident, not from
  faster matmuls.
- **Entropy is a usable control signal at runtime.** It is cheap to compute
  from the router logits already being produced, and it correlates with the
  decisions the scheduler has to make anyway.
- **Memory residency belongs in the graph.** Treating VRAM/RAM/mmap placement
  as part of the execution description, rather than as an allocator detail,
  is what lets a 35B-parameter MoE run inside 8 GB.

## Licence

Documentation and measured results: [CC BY 4.0](LICENSE).
Benchmark harness source: MIT (see [LICENSE](LICENSE)).

The ECSIE engine itself is proprietary, is not contained in this repository,
and no licence to it is granted here.

© 2026 John Donnelly.
