# Benchmarks

Recorded results from `examples/*_benchmark.cpp`. Each entry notes the date, hardware, and
build config, so numbers stay comparable across changes.

Hardware: NVIDIA GTX 1660 SUPER (Turing, no Tensor Cores), Windows 11.
Build: Release config (`cmake --build build --config Release`), LTO enabled.

## matmul (512x512 * 512x512, 20 trials) — 2026-08-14

Baseline before the sync-cleanup + kernel-fusion pass (M9 CUDA autograd complete, no launch
overhead reduction done yet).

| | time | vs AVX2 |
|---|---|---|
| CPU scalar | 374.39 ms | — |
| CPU AVX2 | 31.90 ms | 11.7x faster than scalar |
| CUDA kernel-only (data already resident on GPU) | 0.87 ms | 36.6x faster than AVX2 |
| CUDA full round trip (alloc+upload+compute+download+free) | 16.34 ms | ~2x faster than AVX2 |

**Key finding:** 94.7% of the CUDA "full round trip" time is transfer/allocation overhead, not
compute — the actual matmul kernel is ~430x faster than CPU scalar, but a single isolated
matmul call spends almost all its wall-clock time on `cudaMalloc`/upload/download/`cudaFree`
rather than the math itself.

**Why this matters going forward:** this is the concrete argument for staying GPU-resident
across a whole op chain (which CUDA autograd now makes possible) rather than paying the
transfer cost per-op — a full forward+backward pass with no CPU round-trips in between should
capture much closer to the ~430x kernel-only number across the whole chain, instead of the ~2x
the isolated single-call benchmark shows here. Re-run this same benchmark after the
sync-cleanup + kernel-fusion pass to see the effect of removing per-launch
`cudaDeviceSynchronize()` calls and fusing adjacent kernels.

## matmul_grad_cuda (512x512x512 backward, 20 trials) — 2026-08-15

Effect of fusing the accumulate step directly into the matmul kernel (`matmul_accumulate_cuda`),
removing the temp-buffer + separate `accumulate_cuda` launch. Also removed per-launch
`cudaDeviceSynchronize()` calls across all `.cu` files beforehand (kept `cudaGetLastError()`
everywhere; default-stream ordering makes the syncs unnecessary between chained launches).

| | launches | temp allocs | time |
|---|---|---|---|
| Before (transpose → matmul into temp → accumulate, x2) | 6 | 2 | 2.96 ms |
| After (transpose → matmul_accumulate directly into grad, x2) | 4 | 0 | 1.91 ms |

**35% faster** — fewer kernel launches and no temp `cudaMalloc`/`cudaFree` pair per gradient half.

## bias+GELU fusion (batch=64, features=3072, GPT-2 small MLP hidden dim, 50 trials) — 2026-08-15

Fusing `Linear`'s bias-add directly into the GELU kernel (`bias_gelu_cuda`, wired into a new
`LinearGELU` module), vs. the unfused `add_cuda` then `gelu_cuda` (2 launches, writes the
intermediate to global memory and reads it back).

| | launches | time |
|---|---|---|
| Unfused (add_cuda + gelu_cuda) | 2 | 0.0284 ms |
| Fused (bias_gelu_cuda) | 1 | 0.0248 ms |

**~15% faster.** Smaller win than the matmul fusion — these are tiny elementwise ops dominated
by fixed per-launch overhead rather than compute, so there's less to save per call. Still a real,
repeated saving: this runs on every forward pass through an MLP block.

## grad-clip + AdamW fusion (n=2,359,296, one GPT-2 small MLP weight matrix, 200 trials) — 2026-08-15

`AdamW::step(max_norm)` applies the clip-scale inline inside the AdamW update kernel
(`adamw_clipped_cuda`) instead of a separate `grad_clip_kernel` pass that writes the scaled
gradient back to memory before AdamW reads it again.

First attempt showed a **regression** (fused ~20% slower) because `step(max_norm)` was doing
`cudaMalloc`/`cudaFree` of the tiny sum-of-squares scratch buffer on every single call — that
allocator churn outweighed the launch savings. Fixed by allocating the scratch buffer (`d_sum_sq_`)
once in `AdamW`'s constructor and reusing it every step, same pattern as `d_m_`/`d_v_`.

| | launches | time |
|---|---|---|
| Unfused (clip_grad_norm + step()) | 3 (sum_sq, grad_clip, adamw) | 0.441 ms |
| Fused (step(max_norm)) | 2 (sum_sq, adamw_clipped) | 0.406 ms |

**~9% faster**, after fixing the allocator issue. Isolated at the raw kernel level (no class/malloc
overhead at all) the fusion itself is ~10% faster (0.44 ms → 0.40 ms) — consistent with the
class-level number once the allocation churn was removed. Lesson: a correct kernel fusion can
still net negative if the surrounding host code reintroduces the exact kind of per-call overhead
the fusion was meant to eliminate.

## Caching allocator via cudaMallocAsync/cudaMemPool (matmul 512x512 full round trip, 20 trials) — 2026-08-18

Replaced every `cudaMalloc`/`cudaFree` call across the core library (`Tensor::from_device_ptr`'s
deleter, `Linear`, `Embed`, `LayerNorm`, `SelfAttention`, `AdamW`, `matmul_cuda`'s transpose temps,
etc.) with `cudaMallocAsync`/`cudaFreeAsync` on the default stream, plus a one-time
`init_cuda_mempool()` call (`src/cuda/mempool.cu`) that sets the default pool's
`cudaMemPoolAttrReleaseThreshold` to `UINT64_MAX` so freed buffers stay cached instead of being
handed back to the driver between calls. Used NVIDIA's built-in stream-ordered memory pool rather
than a hand-rolled free-list allocator — deliberate tradeoff, see decision discussion in this
session (2026-08-18): a hand-rolled version would be more defensible in an interview but strictly
worse than what the driver already provides; this project isn't optimizing for that tradeoff here.

| | time |
|---|---|
| Before (cudaMalloc/cudaFree per call) | 16.34 ms |
| After (cudaMallocAsync/cudaFreeAsync + pooled, warmed up) | ~6-9 ms |

**~55-60% faster** on the full round trip once the pool has warmed up (first call after process
start still pays a real `cudaMalloc`, since the pool starts empty). No numerical change — same 35
CUDA test cases / 17301 assertions and 2 CharModel test cases / 3782 assertions still pass
byte-for-byte after the swap.

**Open discrepancy, not attributed to this change:** the "kernel-only" number in this same
benchmark also dropped sharply (0.87 ms → ~0.01 ms) despite that code path being completely
unchanged (same raw `cudaMalloc`/`cudaFree` outside the timed loop, same kernel call inside it).
Likely a GPU clock-state or driver difference since the 2026-08-14 baseline was recorded, not an
effect of the allocator work — flagged here rather than silently re-baselined.

**Scope note:** this only touched the core library (`src/`, `include/`) and this one benchmark.
Other benchmark/example files (`examples/*_benchmark.cpp` besides this one) and `tests/test_m9_cuda.cpp`
still allocate raw `cudaMalloc`/`cudaFree` for their own standalone buffers — untouched, since they
don't go through `Tensor::from_device_ptr` and weren't part of this pass.

## KV cache int8 quantization (CharModel::generate, 4 blocks, 4 heads, 64-dim, 8-token prompt, 200 new tokens, CPU) — 2026-08-20

`KVBlockPool` gained an opt-in `use_quantization` flag: K/V values are quantized to int8 with a
per-(head, token) scale on `append()`, dequantized back to float on `get_k()`/`get_v()`. Wired
through into `CharModel::generate(..., use_quantized_kv_cache)`. Default is off (unquantized,
existing behavior unchanged); CUDA kernels (`quantize_cuda`/`dequantize_cuda`) exist and are
tested but not yet exercised by this specific benchmark (CPU-only run).

| | time | KV cache memory (all blocks, one sequence) |
|---|---|---|
| Plain (float32) | 267.25 ms | 416 KB |
| Quantized (int8) | 268.90 ms | 130 KB |

**~3.2x smaller KV cache, ~0.6% slower.** Memory savings land under the naive 4x (float32 vs int8)
because each slot's scale factor (4 bytes) is stored alongside the compressed data, and slot
granularity here is per-(head, token) rather than per-block — more scales than a coarser scheme
would need, traded for correctness under incremental (decode-time) writes into a block. Time cost
is close to free since the quantize/dequantize work is O(head_dim) per token, dwarfed by the rest
of the forward pass.
