# Four latent CUDA bugs found while writing the first real CUDA attention test

**Status**: accepted (all fixed, all 33 CUDA tests + 5 CPU tests green)

## Context

M9's CUDA-readiness audit had marked `matmul`, `reshape`, `permute`, `contiguous`, `softmax`, `add`, `mul` as "already CUDA-capable" based on reading the code and each op's own isolated unit tests. Writing a correctness test for `SelfAttention::forward()` on CUDA — the first thing in the codebase to actually run `reshape()`/`permute()`/`contiguous()` back-to-back on a tensor that started life on the GPU — immediately crashed with a segfault, then (after the crash was fixed) produced gradients that were numerically close but wrong. Chasing that down surfaced four separate, pre-existing bugs, none of them related to attention itself. Every existing CUDA test happened to avoid all four, which is why they went unnoticed until now.

## Bug 1 — `Tensor::contiguous()` was CPU-only

```cpp
TensorPtr Tensor::contiguous() const {
    auto result = Tensor::create(shape_);        // always allocates on CPU
    ...
    result->at(idx) = at(idx);                    // at() dereferences data_, which is null for a CUDA tensor
```

No device check at all. Calling `.contiguous()` on a CUDA tensor dereferenced a null `data_` pointer — a segfault, not a thrown error. This is the same "silently unsafe on CUDA" pattern already known from the `SelfAttention` masking / `KVCache::grow()` audit, just not caught for this op specifically.

**Fix**: added a real CUDA branch — a strided-gather kernel (`contiguous_cuda` in `matmul_cuda.cu`) that reads `device_data_` via the tensor's own `strides_`/`offset_` and writes a fresh contiguous device buffer. Supports rank ≤ 3 (this codebase's max), padding unused leading dims with shape=1/stride=0 so one kernel covers rank 1/2/3 uniformly.

## Bug 2 — `reshape()`/`permute()` silently dropped CUDA state

Both build their result via the same private constructor:

```cpp
Tensor::Tensor(std::shared_ptr<std::vector<scalar_t>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
    data_ = data; shape_ = shape; strides_ = strides; offset_ = offset;
    // device_ and device_data_ never touched -> default to CPU / nullptr
}
```

For a CUDA tensor, `data_` is already null (real data lives in `device_data_`). Since this constructor never copies `device_`/`device_data_` over, the "view" it produces silently becomes a broken CPU tensor claiming to hold data it doesn't have. Every subsequent `.at()` call on it segfaults. This is what actually crashed the attention test — at the very first `Q->reshape(...)` after a CUDA matmul.

Never caught before because every existing CUDA test does `reshape(...)->to(Device::CUDA)` — reshape on CPU, *then* move to CUDA — never the reverse order.

**Fix**: added `Device device = Device::CPU` and `scalar_t* device_data = nullptr` parameters to that constructor (defaulted, so every other call site is unaffected), and pass `device_, device_data_` through from both `reshape()` and `permute()`.

## Bug 3 — `reshape()`/`permute()` backward was also CPU-only

Even after fixing the constructor, `backward()` crashed the moment it reached either op's `grad_fn_`:

```cpp
self->grad().mutable_data()[i] += upstream.data()[i];   // reshape
self->grad().mutable_data()[i] += upstream.at(out_idx);  // permute
```

Both use CPU-only accessors on `self->grad()`, which is a CUDA tensor when `self` is.

**Fix**:
- `reshape()`'s backward is a flat elementwise accumulate (reshape requires a contiguous input, so there's no reordering) — reused the existing `accumulate_cuda` kernel, no new kernel needed.
- `permute()`'s backward genuinely reorders indices, so it needed a new kernel (`permute_grad_cuda`): for each flat index into `self`, unravel via `self`'s shape, remap through the permutation `axes`, and gather from `upstream` at the remapped strided position. Rank-3 specific, matching the only rank this codebase actually permutes.

## Bug 4 — `init_grad()` allocated CUDA grad buffers with the wrong strides

```cpp
grad_ = Tensor::from_device_ptr(d_grad, shape_, strides_);   // self's own strides_, not standard
```

The CPU branch (`Tensor::create(shape_)`) always allocates a standard-contiguous grad tensor, regardless of `self`'s own layout. The CUDA branch instead reused `self`'s own `strides_` — which, for a permuted (non-contiguous) `self`, are *not* standard. Every existing CUDA grad kernel before this only ever wrote/read grad buffers as flat arrays (`mutable_device_data()[i] +=`), so this mismatch never mattered. `permute_grad_cuda` (bug 3's fix) was the first kernel to actually read a grad tensor's `strides()` for addressing — which exposed the inconsistency as gradients that were close in magnitude but numerically wrong (not zero, not garbage — just addressed through the wrong strides).

This also silently broke `contiguous()`'s own backward (bug 1) a second time: its original backward wrote into `self_grad` using `self`'s real (permuted) strides, which was only "correct" by accident while `init_grad()` had the same bug. Fixing `init_grad()` to always allocate standard-contiguous grad buffers meant `contiguous()`'s backward also simplified down to a flat `accumulate_cuda` call, same as `reshape()`'s — no dedicated scatter kernel needed at all.

**Fix**: `init_grad()` and `backward()`'s grad-seeding both now compute standard row-major strides for `shape_` explicitly, instead of reusing `strides_`.

## Net result

Four bugs, one root cause each, discovered in this order because each fix exposed the next: constructor silently drops device → crash on first reshape; fixed, then backward crashes; fixed, then gradients are subtly wrong; fixed, and `contiguous()`'s own backward — written assuming the old (buggy) grad layout — needed simplifying to match. `SelfAttention::forward()` + `backward()` now runs correctly on CUDA end-to-end, verified against the CPU reference in `tests/test_m9_cuda.cpp`.

The general lesson: an op being individually unit-tested on CUDA (in isolation, always starting from a tensor freshly moved via `.to(Device::CUDA)`) is not the same as that op being safe to *chain* after another CUDA op. The gap only shows up when something actually exercises the chain — which nothing did, for `reshape`→`permute`→`contiguous`, until this test.
