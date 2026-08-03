# ECSIE Engine API Reference

## Overview

ECSIE exposes a REST-compatible HTTP API through `src/api/server.cpp`.

## Endpoints

### POST /v1/completions

Generate completions for a prompt.

**Request body:**
```json
{
  "model": "<model-path-or-alias>",
  "prompt": "<input text>",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.95
}
```

**Response:**
```json
{
  "id": "<request-id>",
  "choices": [
    { "text": "<completion>", "finish_reason": "stop" }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### GET /v1/entropy

Returns current entropy state for diagnostics.

```json
{
  "H_t": 0.32,
  "R_t": 0.41,
  "L_t": 0.18,
  "A_t": 0.87,
  "C_t": 0.76,
  "policy": "adaptive"
}
```

### GET /health

Health check endpoint.

## C++ Public API

See `include/ecsie/engine.hpp` for the programmatic C++ interface.

```cpp
#include <ecsie/engine.hpp>

ecsie::Engine engine;
engine.load_model("/path/to/model.gguf");
auto result = engine.generate(prompt, max_tokens);
```

## CLI

```bash
# Start inference server
./bin/ecsie_server --model /path/to/model.gguf --port 8080

# Run benchmark
./bin/ecsie_benchmark --model /path/to/model.gguf --workload benchmarks/workloads/stable.json
```
