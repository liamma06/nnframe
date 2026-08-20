# Build-vs-use: when to hand-roll vs call NVIDIA's own library

**Status**: accepted (decided for the caching allocator; recorded here as a general policy for future M10/M11 work)

## Context

While scoping M10's caching CUDA allocator, the obvious real-world answer was "don't write one — call `cudaMallocAsync`/`cudaMemPool`, NVIDIA already solved this." That's true, and raised the actual question: given `CLAUDE.md`'s goal ("understand every design decision deeply enough to defend it in a systems/ML-infra interview"), when does it make sense to use NVIDIA's own provided implementation instead of hand-rolling, given the project's blanket rule is "no Eigen/Armadillo/libtorch in the core" for tensors/autograd?

## Decision

Rule of thumb: hand-roll the op if writing it *is* the milestone's point; use NVIDIA's version if the milestone is about something else and the op is just infrastructure in the way.

Concretely for the caching allocator: **used `cudaMallocAsync` + `cudaMemPool` instead of hand-rolling a free-list.** Reasoning:
- Memory *management* (not memory *math*) was never what M9/M10 were testing understanding of — the interesting, defensible work in this project is the tensor/autograd/kernel math, not allocator internals.
- NVIDIA's version is strictly better engineered (real fragmentation handling via block splitting/coalescing, stream-safety, tunable release threshold) than anything hand-rolled here would be in reasonable time.
- The *configuration* surface (`cudaMemPoolSetAttribute`, release threshold, `cudaMemPoolTrimTo`) still had to be understood and reasoned about — see the session's discussion of `cudaMemPoolAttrReleaseThreshold` and its tradeoff (pool never shrinks below peak usage unless trimmed) — so this wasn't a zero-understanding shortcut, just a scoped one.
- Wired into `Tensor::from_device_ptr`'s deleter and every core `cudaMalloc`/`cudaFree` call site (`src/`, `include/`); see `docs/benchmarks.md`'s "Caching allocator via cudaMallocAsync/cudaMemPool" entry (2026-08-18) for the measured result (~55-60% faster full round trip, matmul benchmark).

## Other places this same choice exists in the codebase

Surveyed while making the allocator decision — recorded here so future milestones don't have to re-derive the tradeoff from scratch:

| Op | NVIDIA-provided alternative | Why we hand-rolled it anyway |
|---|---|---|
| `matmul_cuda` / `matmul_batched` | **cuBLAS** — tuned GEMM, uses Tensor Cores | Writing the matmul kernel *is* M9's point; a from-scratch tensor framework calling cuBLAS for its core op would defeat the project |
| `softmax_cuda`, `layernorm_cuda`, attention | **cuDNN** — fused, optimized primitives, includes flash-attention-style kernels already | Same reasoning; also would have made M10's Flash Attention item moot — the whole point is to build the online-softmax tiling by hand |
| `sum_reduce_cuda`, mean/variance tree-reduction in `layernorm_cuda` | **Thrust `reduce`** / **CUB** block/warp reductions | Parallel reduction is a core GPU-programming concept this project is meant to teach; using a library reduction hides exactly the warp/block mechanics `CLAUDE.md`'s M9 note ("prioritize understanding GPU architecture over speed of implementation") calls out |
| Weight initialization (currently CPU `std::mt19937`, copied to device) | **cuRAND** (`curandGenerateUniform`, etc.) | Not yet done on-device at all; if/when this becomes a bottleneck, cuRAND is the obvious fix and wouldn't cost any conceptual understanding — RNG-on-device isn't a concept this project is testing |
| Gradient sync (N/A — single GPU) | **NCCL** | Not applicable yet; the provided answer if this ever goes multi-GPU |
| Per-step kernel launch batching (M10, unstarted) | **CUDA Graphs' own capture/replay API** | Same choice will come up again here — capture/replay is itself the provided abstraction; there's little to "hand-roll" as an alternative (unlike matmul/attention, there's no meaningfully different from-scratch version of this to build), so this one will likely just use the provided API directly |

## Net result

The allocator is the first place this project deliberately chose "use NVIDIA's" over "build from scratch," breaking from the project's default hand-roll-everything posture. Documenting the reasoning here so it's defensible as a deliberate scoping decision, not an inconsistency — and so the same reasoning doesn't need to be re-litigated when cuBLAS/cuDNN/Thrust inevitably come up again in interviews or later milestones.
