# nnframe

A neural network framework built from scratch in C++17. No external math libraries — everything from tensors to autograd to layers is hand-rolled. Goal: understand every design decision deeply enough to defend it in a systems/ML-infra interview.

## Stack

- **Language:** C++17
- **Build system:** CMake
- **Compiler:** MSVC (via Build Tools for Visual Studio 2026)
- **Platform:** Windows, CPU first

## Rules

- No Eigen, Armadillo, libtorch, or any existing autograd/tensor library in the core
- C++ standard library only for core math (`<vector>`, `<cmath>`, `<memory>`, etc.)
- Readable over clever — short "why" comments on non-obvious lines
- One test file per milestone, all green before moving to the next

## Milestones

- **M0** — Project setup
- **M1** — Tensor (flat buffer + strides, element-wise ops, matmul, reshape, broadcasting)
- **M2** — Autograd (computation graph, backward pass, gradient checking)
- **M3** — Layers (Linear, ReLU, GELU, sequential container)
- **M4** — Loss + optimizer (MSE, plain gradient descent)
- **M5** — Training loop, proof target: XOR
- **M6** — Transformer support (attention, LayerNorm, positional encoding), proof target: tiny character-level model
- **M7** — Inference (lean): real multi-head attention (split/concat heads, currently cosmetic), KV cache, autoregressive generation with temperature/top-k sampling (current)
- **M8** — SIMD: profile the CPU loop, AVX2-vectorize the hot ops (expect matmul), benchmark before/after
- **M9** — CUDA backend: device memory management, custom matmul kernel, port training + inference to GPU, correctness-check against CPU reference. Go slow here — prioritize understanding GPU architecture (memory model, warps/threads/blocks) over speed of implementation
- **M10** — Inference + training revisited, target: GPT-2 small (124M params) on 6GB VRAM. Serving-style techniques (continuous batching, paged KV cache) plus the memory/throughput work needed to actually fit GPT-2 small: mixed precision (bf16 + loss scaling, highest priority), AdamW optimizer, a caching CUDA allocator (reuse freed device buffers instead of cudaMalloc/cudaFree per op), Flash Attention-style fused/tiled attention kernel (online softmax, no full S×S materialization), KV-cache int8 quantization, a CUDA path for `KVCache::grow()` (currently CPU-only via `at()`, no device check — blocks GPU inference), and fused kernels (residual+LayerNorm, bias+GELU) as a lower-priority throughput pass. Once things fit in memory, CUDA Graphs (capture the repeated per-step kernel-launch sequence once, replay it instead of re-issuing each launch) is a further lower-priority latency optimization — addresses launch overhead, not memory, so it comes after the memory work above. See `docs/mni-ml-framework-notes.md` for the research behind this list.

## Repo layout

```
include/    — public headers
src/        — implementations
tests/      — one file per milestone
examples/   — runnable proof targets (XOR, tiny transformer)
docs/       — decisions.md logs every real design fork
.claude/    — project-specific Claude context
```

## How we work

- Milestone by milestone — no skipping ahead until tests are green
- Before each milestone: sketch the approach, discuss design forks, let the user pick
- After each milestone: quiz on the core concept before moving on
