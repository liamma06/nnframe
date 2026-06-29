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

- **M0** — Project setup (current)
- **M1** — Tensor (flat buffer + strides, element-wise ops, matmul, reshape, broadcasting)
- **M2** — Autograd (computation graph, backward pass, gradient checking)
- **M3** — Layers (Linear, ReLU, sequential container)
- **M4** — Loss + optimizer (MSE, plain gradient descent)
- **M5** — Training loop, proof target: XOR
- **M6** — Transformer support (attention, LayerNorm, positional encoding), proof target: tiny character-level model

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
