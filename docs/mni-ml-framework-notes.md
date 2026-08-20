# Notes: mni-ml/framework — what's worth adopting toward the GPT-2 small goal

Reference: https://github.com/mni-ml/framework (TypeScript + Rust, CPU/CUDA/WebGPU backends,
built for the same "understand ML frameworks internally" reason as nnframe).

Their `README.md` undersells the project — it lists no transformer/attention/mixed-precision
support at all. The actual code (`src/native/kernels/`, `src/native/src/`) has substantially
more than the docs claim. Notes below are from reading the kernel and Rust source directly.

## What they have that we don't (yet)

### 1. Flash Attention (`kernels/flash_attention.cu`)
FlashAttention-2 style forward/backward. Tiles over the sequence dimension (`blockIdx.y` +
`blockDim.y`), keeps a running max/sum for online softmax so it never materializes the full
`S×S` score matrix (O(S) memory instead of O(S²)). Supports causal masking by shrinking the
inner loop bound (`col_end = causal ? row+1 : S`) rather than masking after the fact. No
`__shared__` tiling — relies on global memory with GPU-side caching, so it's a simpler/less
optimal implementation than the original FlashAttention paper's shared-memory version, but the
online-softmax trick (the actual hard part) is there.
**Relevance:** already on our own M10 roadmap ([[project_training_target]]). At GPT-2-small
context lengths (256–512 tokens) this matters less than at long context, but it's still the
right technique to lower peak VRAM during training, which is our binding constraint.

### 2. Mixed precision (`kernels/mixed_precision.cu`) — CORRECTED, see below
**Initial read of this file was too generous — corrected after tracing the actual call path.**
The `.cu` file has real f32→bf16 cast kernels (round-to-nearest-even) alongside `scale_f32` and
`check_inf_nan_f32`. But tracing how they're actually *used*, in `src/native/src/ops/mixed_precision.rs`
(the Rust glue that calls the kernels) — only `scale_f32` and `check_inf_nan_f32` are ever called,
both of which operate on plain f32 data. The bf16 cast kernels aren't wired into anything in the
ops layer. And their `GpuTensor`/`Tensor` struct (`src/native/src/tensor.rs`) is **hardcoded to
f32 everywhere** — `Vec<f32>` on CPU, `CudaSlice<f32>` on CUDA, no `DType` enum, no generic type
parameter, no `cast()`/`to_dtype()` method anywhere in the codebase.

So what they actually built is **loss scaling only** (scale gradients up before backward, check
for inf/nan) — a numerical-stability technique, useful on its own, but it does nothing for VRAM:
everything is still stored as full 32-bit floats. They did NOT build the actual memory-saving
half — a tensor type that can be bf16-resident. That's the harder problem (same one nnframe's
`scalar_t`-hardcoded-to-`float` design would have to solve too), and it's unsolved in their
codebase, not something we can copy.
**Relevance:** loss scaling itself is still worth adding (cheap, standalone, protects future bf16
work from day one). But the actual VRAM-halving technique — real bf16-resident tensors — is
something we'd be designing from scratch, not porting from them.

### 3. KV-cache int8 quantization (`kernels/kv_quant.cu`)
Row-wise int8 quant: per-row `scale = max_abs/127`, quantize/clamp to int8, dequant on read.
4x memory reduction on cached K/V. We already have a KV cache (M7); this is a cheap add-on once
inference at GPT-2-small scale needs longer generated sequences than fit comfortably in fp32.
**Relevance:** inference-side win, not training-side — lower priority than mixed precision for
reaching the training goal, but easy to justify adding to M10 alongside paged KV cache.

### 4. Fused kernels (`kernels/fused_ops.cu`)
Two fusions: (residual-add + LayerNorm) in one kernel, and (bias-add + GELU) forward+backward
in one kernel each. Point is avoiding extra kernel launches and extra global-memory round-trips
for the intermediate (unfused) result.
**Relevance:** real but secondary — a throughput win, not a memory-fit win. Worth doing once
correctness is solid and we're optimizing training speed, not before.

### 5. Caching allocator (`src/allocator.rs`)
Size-bucketed free-list pool (`HashMap<usize, Vec<buffer>>`) — reuses freed device buffers
instead of calling `cudaMalloc`/`cudaFree` every op, since `cudaMalloc` is a synchronizing,
relatively expensive call. CPU path zeroes buffers in place on reuse instead of reallocating.
Has a manual `clear_cache()` since pooled-but-unused buffers otherwise just sit resident.
**Relevance:** directly applicable to our `Device`/`to()` design. Right now every `.to(CUDA)`
call and every intermediate op result does a fresh `cudaMalloc`. At GPT-2-small training scale
(many ops per step, many steps) that allocator overhead adds up and also risks fragmenting a
6GB card faster than necessary. Worth a small ADR once matmul's CUDA dispatch is in.

### 6. AdamW (`kernels/adamw.cu`)
Decoupled weight decay (the "W" in AdamW) as its own fused kernel rather than adding decay into
the gradient before the Adam update. We currently only have plain SGD (M4 milestone scope).
**Relevance:** needed regardless of framework comparison — AdamW is the standard optimizer for
transformer training and is on our own roadmap already, just calling it out as confirmed-useful.

### 7. Device abstraction (`src/device.rs`)
Compile-time feature-flag backend selection (`#[cfg(feature = "cuda")]` etc.), single global
`GpuDevice` singleton via `OnceLock`, runtime-compiled kernels (NVRTC for CUDA, WGSL modules for
WebGPU) cached in a `HashMap`. Notably: error handling is `.expect()`/panic-on-init-failure, not
graceful — same "fail loudly" philosophy as our exception-based ADR-0003 design, just at a
different layer (init-time vs call-time).
**Relevance:** validates our own `#ifdef NNFRAME_WITH_CUDA` + `Device` enum approach — no
change needed, but confirms compile-time backend selection is the standard pattern, not
something we should replace with runtime detection.

### 8. Architecture layering (`kernels/*.cu` → `ops/*.rs` → `device.rs` → `lib.rs`)
Confirmed by reading multiple files: raw `.cu` kernels do all actual numeric work (nothing
touches data on the CPU side). A thin per-op Rust wrapper (`ops/mixed_precision.rs`,
`ops/matmul.rs`, etc.) takes tensor IDs / device pointers and launches the matching kernel.
`device.rs`'s `GpuDevice` singleton holds the compiled kernels (NVRTC at startup) and does the
actual launch. `lib.rs` exposes N-API functions to the TS side, which orchestrates the training
loop. Tensors are tracked in a central `TensorStore` by ID so ops never copy data back to the
host between calls — everything chains on-GPU until something explicitly needs the CPU.
**Relevance:** confirms the right target behavior (minimize CPU↔GPU round-trips, chain ops
GPU-resident) but NOT the mechanism. Their `TensorStore`+`TensorId` indirection exists because
Rust↔JS crosses a language boundary (N-API) that makes passing objects directly awkward — we
don't have that problem. nnframe already passes `TensorPtr` (`shared_ptr<Tensor>`) directly
within one C++ binary, so achieving the same "stay GPU-resident" goal just means giving every op
(`add`, `sub`, `mul`, `softmax`, `log`, `mean`, LayerNorm, embedding lookup, cross-entropy) a
CUDA-dispatch branch, same pattern already built for `matmul()` — no store/ID indirection needed.

### 9. Op coverage vs. nnframe (corrected — nnframe has more than a first pass assumed)
nnframe already has CPU implementations of LayerNorm (`include/modules/layernorm.h`, full
forward+hand-derived backward), CrossEntropy (`include/loss/cross_entrop.h`, built on
`softmax`/`log`/`mean`), and token embedding lookup (`include/modules/embed.h`, forward+backward)
— an earlier pass over this doc incorrectly claimed these were missing entirely; they exist and
are correct, they just have no CUDA path yet. `LayerNorm`/`Embed`/`CrossEntropy` index into
tensors directly via `at()`/`mutable_data()` rather than composing through `Tensor` ops, so they
need their own kernels (`layernorm.cu`, `embedding.cu`), not just CUDA branches on the ops they
call. AdamW is the one genuinely-missing piece (nnframe only has plain SGD, M4 scope).

### 10. Matmul is NOT a hand-rolled kernel — it's cuBLAS
Deep-dive finding: `src/native/kernels/` has no `matmul.cu` at all. `ops/matmul.rs` calls
`dev.blas.gemm(...)` / `dev.blas.gemm_strided_batched(...)` — NVIDIA's cuBLAS library, wired in via
the `cudarc` crate's `cublas` feature (`Cargo.toml`: `cudarc = { features = ["cublas", "nvrtc",
"curand", ...] }`). There is no shared-memory tiling, warp primitives, or tensor-core (WMMA) code
anywhere in their matmul path — they simply never wrote a custom matmul kernel; cuBLAS handles all
of that internally (tiling, register blocking, and tensor cores when the hardware has them).
For batched (rank-3) matmul specifically, they call `gemm_strided_batched` — cuBLAS's own batched
API — rather than a custom `blockIdx.z`-batched kernel like the one nnframe is building for M9.
**Relevance:** confirms there is genuinely nothing to "port" for matmul performance — cuBLAS is a
professionally hand-tuned, closed-source-internals library that nnframe's CLAUDE.md rules
explicitly exclude ("No Eigen, Armadillo, libtorch, or any existing autograd/tensor library in the
core"). This is worth stating plainly for the interview angle: matching cuBLAS's matmul throughput
by hand is not a realistic goal for this project, and mni-ml/framework didn't attempt it either —
they used the professional library exactly where "understand it deeply" wasn't the point. nnframe's
own hand-rolled `matmul_kernel` (naive one-thread-per-output-element, no shared memory) is already
past what mni-ml wrote for matmul, since they wrote none. **Also directly confirms the tensor-core
question raised in the research brief: the GTX 1660 SUPER (Turing TU116 die) has no Tensor Cores at
all** — Tensor Cores were reserved for the RTX Turing parts (2060/2070/2080); the GTX 16-series is
a cost-reduced die without them. So even if nnframe wanted WMMA-based tensor-core matmul, the
hardware can't run it — moot point, not just deprioritized.

### 11. Tree/shared-memory reduction pattern in `softmax.cu` and `layernorm.cu`
Both kernels use **one CUDA block per row**, with all threads in the block cooperating via a
`__shared__ float sdata[BLOCK_DIM]` buffer and classic tree reduction:
```
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```
Softmax does this twice (once for row-max, once for the exp-sum) — a genuine two-pass algorithm
over the row, each pass using the tree reduction to combine partial per-thread results down to one
value, broadcast back through shared memory. LayerNorm does the same shape: one reduction for mean,
`__syncthreads()`, a second reduction for variance using that mean, entirely within one block per
row (not Welford's single-pass algorithm — they chose the simpler two-pass approach).
**Relevance: directly actionable, and high-priority.** nnframe doesn't have CUDA kernels for
`softmax` or `LayerNorm` at all yet (§9) — when writing them, this one-block-per-row +
shared-memory-tree-reduction pattern is the correct baseline design, not something to reinvent.
Doing it naively (one thread computes the whole row's max/sum sequentially, like nnframe's current
CPU implementation) would leave the GPU almost entirely idle per row. This is worth prioritizing
into the "full CUDA op coverage" work already at the top of the priority list below — it's the
concrete technique that work needs, not a separate item.

### 12. Inconsistency worth learning from: `reduce.cu` itself uses naive reduction
In contrast to §11, the generic reduction kernels in `kernels/reduce.cu` (used for general
`sum`/`mean`/`max` over a dimension) do NOT use tree reduction — each thread just loops
sequentially: `for (int d = 0; d < dim_size; d++) sum += inp[...]`, one thread per *output* element,
no cooperation within a block, no `__syncthreads()`, no warp shuffles. This is a real
inefficiency in their own codebase — they clearly know the tree-reduction pattern (used it
correctly in softmax/layernorm) but didn't apply it consistently everywhere reduction happens.
There is one exception: `elementwise.cu`'s `sum_reduce_all_f32` (full-tensor sum to a scalar) DOES
use shared-memory tree reduction within each block, then `atomicAdd`s each block's partial result
into a single global accumulator — a correct hybrid pattern (intra-block tree reduction + inter-block
atomics) worth using as the template for nnframe's own future `mean()`/`sum()`-to-scalar CUDA kernel.
**Relevance:** two lessons. (1) The tree-reduction+atomicAdd hybrid in `sum_reduce_all_f32` is the
right pattern for nnframe's own `mean()` CUDA kernel (currently CPU-only, sequential sum in
`tensor.cpp`). (2) The inconsistency itself is a useful thing to watch for in our own code once we
write multiple CUDA kernels that each do "reduce along a dimension" (elementwise reduce, softmax's
internal reduce, layernorm's internal reduce, cross-entropy's internal reduce) — worth writing one
shared device-side reduction helper/pattern early rather than re-deriving it ad hoc per kernel and
risking the same drift they have.

### 13. Cross-entropy fuses softmax directly into the loss kernel
`kernels/cross_entropy.cu`'s forward kernel computes row-max, exp-sum, and the negative-log-
likelihood of the target class all within a single kernel launch — it does not call a separate
softmax kernel first and then index into the result the way nnframe's CPU `CrossEntropy::forward`
does (`logits->softmax(1)` as its own tensor op, then gathers the target-class probability).
Backward is the standard `(softmax_out - one_hot_target) * grad_scale` per logit.
**Relevance:** a real throughput idea for nnframe's eventual `cross_entropy.cu` — computing softmax
as a fully separate `Tensor` (materializing the whole `[seq_len, vocab_size]` probability matrix in
VRAM) costs extra global-memory writes/reads that a single fused forward kernel avoids. Lower
priority than getting *a* CUDA cross-entropy kernel working at all (§9's existing gap), but worth
doing the fused version directly rather than a naive softmax-then-gather CUDA port, since it's not
meaningfully more code either way.

### 14. Embedding backward needs atomics — a real correctness gotcha, not just a perf one
`kernels/embedding.cu`'s backward kernel: `atomicAdd(&dweight[indices[t] * embed_dim + d],
dout[t * embed_dim + d]);`. This isn't an optimization choice — it's required for correctness. When
the same vocabulary token appears more than once in a sequence (extremely common), multiple GPU
threads will try to scatter-add gradients into the *same* embedding row simultaneously. Without
`atomicAdd`, concurrent unsynchronized writes to the same memory location silently lose updates
(last write wins, others discarded) — not a crash, just quietly wrong gradients.
**Relevance: important gotcha to flag before nnframe writes `embedding.cu`.** nnframe's existing
*CPU* `Embed::forward`'s backward (`include/modules/embed.h`) loops sequentially over tokens, so
this race condition doesn't exist there — CPU execution is single-threaded per this loop. It's easy
to port that same loop structure to a naive CUDA kernel (one thread per token) and get silently
wrong gradients whenever a token repeats in the input, without any compiler or runtime error to
catch it. Needs `atomicAdd` from the start of the CUDA embedding backward, not added later as a fix.

### 15. Gradient clipping — a genuinely missing piece, not previously flagged
`kernels/grad_util.cu` has two kernels nnframe has no equivalent of at all, CPU or GPU:
`grad_norm_sq_partial_f32` (each block computes a partial sum of squared gradient values via tree
reduction, left for the host to finish summing into one L2 norm) and `grad_clip_f32` (multiplies
every gradient element by a scale factor once the norm is known). Together: compute the global
gradient L2 norm, and if it exceeds a threshold, rescale all gradients down proportionally —
standard gradient clipping, used to prevent occasional huge gradients from blowing up training.
**Relevance:** genuinely missing from nnframe's roadmap up to this point — not CPU, not GPU. This
is standard practice for transformer training specifically (exploding gradients are a known
failure mode for deep transformer stacks) and is cheap to add. Worth scoping alongside AdamW as
a training-stability essential, not just a CUDA-porting nice-to-have — arguably should exist on
the CPU path first, per the project's own "CPU path before GPU path" milestone convention.

### 16. Dropout mask generation via cuRAND, applied as a separate cheap kernel
`Cargo.toml` confirms the `curand` cudarc feature is enabled — `kernels/dropout.cu`'s forward/backward
kernels (`out[i] = x[i] * mask[i] * scale`) just consume a precomputed mask; the actual random mask
generation happens via NVIDIA's cuRAND library elsewhere, not in these kernels. The separation
(generate randomness once via a trusted, correct library primitive; apply the mask via a trivial
elementwise multiply kernel) avoids the well-known trap of writing your own GPU-side RNG (e.g.
naively seeding by thread index, which produces correlated "random" values across threads).
**Relevance:** if/when nnframe adds dropout (GPT-2 uses it during training; not strictly required
for a first successful GPT-2-small training run, lower priority than the items above), the lesson
is: don't hand-roll a GPU random number generator. Either use NVIDIA's cuRAND (a real external
library, would need the same "is this core math" judgment call CLAUDE.md's no-external-libraries
rule was written for — likely fine since it's RNG infrastructure, not autograd/tensor logic) or
generate the mask on the CPU and copy it over. Not urgent — GPT-2 small can train without dropout
first, add it once correctness on the core path is solid.

### 17. Release build optimization flags — a free win nnframe hasn't taken yet
`src/native/Cargo.toml`'s `[profile.release]`: `opt-level = 3` and `lto = true` (link-time
optimization). Simple, standard, high-leverage settings.
**Relevance: worth checking nnframe's own build configuration.** Every build command run so far in
this project has produced binaries under `build/Debug/` — meaning nnframe has been building and
benchmarking in an **unoptimized Debug configuration this entire time** (no `-DCMAKE_BUILD_TYPE`
has been passed to any `cmake -S . -B build` call seen in this project's history, and CMake/MSVC
defaults to Debug when unset). This is a bigger and easier win than anything else in this document:
before writing any new kernels, building with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
(plus optionally `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)` in `CMakeLists.txt` for LTO) would
speed up both the existing CPU AVX2 matmul and the CUDA kernels for free, with zero code changes,
and would make any future before/after benchmark numbers actually meaningful — benchmarking a Debug
build and drawing performance conclusions from it would be misleading.
**No CUDA-specific compile flags were found** in their build system (`build.rs` only sets up the
N-API bindings; nvcc flags like `--use_fast_math` or explicit `-arch`/`-gencode` targets are
handled internally by the `cudarc` crate, not visible in their repo) — so there's nothing further
to port on the CUDA-flags side specifically. nnframe's own `CMakeLists.txt` already sets
`CMAKE_CUDA_ARCHITECTURES native`, which is the reasonable equivalent.

### 18. No benchmark validation exists in their repo
Their `test/` directory (`autograd.test.ts`, `native.test.ts`, `nn.test.ts`, `tensor.test.ts`,
etc.) contains only correctness unit tests — no benchmark files, no recorded throughput/speedup
numbers anywhere in the repository.
**Relevance:** every performance-oriented claim in this document (tree reduction being faster than
naive, fused kernels reducing launch overhead, etc.) is standard, well-established CUDA practice,
but is NOT independently validated by mni-ml/framework's own test suite — they never measured it
themselves either. Treat these as "correct standard techniques worth applying," not as "proven to
help this specific codebase" — nnframe should benchmark its own before/after when applying any of
them (already the project's own habit per M8's SIMD milestone: "benchmark before/after").

## What's NOT worth adopting

- **WebGPU backend** — irrelevant to our goal (Windows CPU/CUDA only per CLAUDE.md stack).
- **Conv1d/Conv2d/pooling kernels** — not needed for a decoder-only transformer training run.
- Full TS/Rust N-API bridge architecture — that's solving a "ship this as an npm package"
  problem we don't have; not relevant to a single-binary C++ research project.

## Priority order for reaching GPT-2 small (124M params, 6GB VRAM)

0. **Build in Release, not Debug** — free win, do this immediately regardless of everything else.
   Every build in this project so far has been an unoptimized Debug build (§17). Switch to
   `-DCMAKE_BUILD_TYPE=Release` (+ interprocedural/LTO optimization) before writing new kernels or
   trusting any benchmark numbers — costs nothing, and an unoptimized baseline makes every future
   "before/after" comparison meaningless.
1. **Full CUDA op coverage + autograd** — the actual prerequisite for everything below. Elementwise
   ops (`add`/`sub`/`mul`), `softmax`, `log`, `mean`, plus real kernels for LayerNorm, embedding
   lookup, and cross-entropy all need CUDA-dispatch branches (matmul's pattern) before a full
   training step can stay GPU-resident at all. When writing the softmax/layernorm kernels
   specifically, use the one-block-per-row + shared-memory tree-reduction pattern from §11, not a
   naive per-thread sequential scan. When writing the embedding backward kernel, it needs
   `atomicAdd` for the scatter-add from day one (§14) — this is a correctness requirement, not an
   optimization. Matmul itself needs no further work here — cuBLAS-chasing isn't the goal, and the
   GTX 1660 SUPER has no Tensor Cores to exploit anyway (§10).
2. **AdamW + gradient clipping** — both training-stability essentials nnframe is missing entirely
   (CPU or GPU); AdamW was already known-missing, gradient clipping is a new finding (§15) worth
   scoping alongside it since both belong to "does training actually converge," not just "is it
   fast." Clipping should land on the CPU path first, per the project's own CPU-before-GPU
   convention, before a CUDA version.
3. **Real bf16-resident tensors + loss scaling** — this is the actual VRAM-halving technique;
   note it is NOT something to port from mni-ml/framework (see §2 correction above) — they never
   built the bf16-resident tensor type, only the loss-scaling safety net around it. This is
   original design work for nnframe, highest leverage once op coverage exists to run it through.
4. **Caching allocator for CUDA buffers** — reduces `cudaMalloc` overhead/fragmentation risk
   once training loops are doing many ops/step over many steps.
5. **Flash Attention (online softmax)** — lowers attention's peak memory; matters more as
   context length grows, still worth it at 256–512 tokens.
6. **KV-cache int8 quantization** — inference-time win, do alongside M10's paged KV cache work.
7. **Fused residual+LayerNorm / bias+GELU kernels, and fused softmax-in-cross-entropy (§13)** —
   throughput optimization, last priority, do once correctness and memory-fit are both solid.
8. **Dropout** — lowest priority; GPT-2 small can train without it initially, and if added later,
   use a real RNG source (cuRAND or CPU-generated mask) rather than hand-rolling GPU randomness (§16).
