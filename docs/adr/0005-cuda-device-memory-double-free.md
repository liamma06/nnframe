# CUDA device memory double-free found while writing the TransformerBlock test

**Status**: accepted (fixed, verified with compute-sanitizer, all 35 CUDA tests + 5 CPU tests green)

## Context

Writing the first end-to-end `TransformerBlock` CUDA-vs-CPU test — the first thing in the codebase to chain LayerNorm, SelfAttention, Linear, and GELU together in one pass, the way an actual model forward would — crashed with `CUDA error ... invalid argument`, reported from inside `layernorm_cuda`'s `cudaGetLastError()` check. That line was innocent; the real failure was upstream.

## The bug

`~Tensor()` unconditionally called `cudaFree(device_data_)` for every CUDA tensor:

```cpp
Tensor::~Tensor(){
    #ifdef NNFRAME_WITH_CUDA
        if (device_data_ != nullptr) {
            cudaFree(device_data_);
        }
    #endif
}
```

But `reshape()` and `permute()` build **views** — they reuse the source tensor's `device_data_` pointer directly (that's the point of a view: no copy, no new allocation) rather than allocating their own buffer. Only `.contiguous()` actually copies into a fresh buffer. So a chain like:

```cpp
TensorPtr K_head = K->reshape({...})->permute({...});   // no .contiguous()
```

leaves `K`, `K`'s reshape-view, and `K_head` all pointing at the **same** raw device buffer. When the autograd graph holding all of them finally unwinds (at scope exit, once the last shared_ptr reference drops), each one's destructor calls `cudaFree` on that same pointer — the first call succeeds, every one after it fails.

Crucially, `cudaFree`'s return value was never checked (no `CUDA_CHECK`), so the failure was silent. It didn't crash — it just left CUDA's sticky error state poisoned. The failure only surfaced later, at the next unrelated call anywhere in the program that actually checked for an error (`layernorm_cuda`'s `CUDA_CHECK(cudaGetLastError())`), which got blamed for a completely unrelated test's cleanup. Confirmed the real source with `compute-sanitizer --tool memcheck`, which caught four `cudaErrorInvalidValue` hits on `cudaFree` during the `SelfAttention` test's teardown, all before the reported `layernorm_cuda` line.

This is the same root shape as the bugs in ADR 0004: individually-tested CUDA ops that had never been *chained* together in a way that kept multiple views alive simultaneously, so the hazard was invisible until `TransformerBlock` did exactly that.

## The fix

`data_` (the CPU buffer) was already a `shared_ptr<vector<scalar_t>>`, so CPU views share ownership safely and free automatically once the last reference drops. `device_data_` had no equivalent — it was a bare raw pointer, freed unconditionally by whichever `Tensor` happened to die.

Changed `device_data_` from `scalar_t*` to `std::shared_ptr<scalar_t>`, with the actual `cudaFree` wired in as a custom deleter at the one place a device buffer is first wrapped (`Tensor::from_device_ptr`):

```cpp
result->device_data_ = std::shared_ptr<scalar_t>(d_ptr, [](scalar_t* p){ cudaFree(p); });
```

Views created by `reshape()`/`permute()` now just copy the `shared_ptr` (cheap, no new allocation, refcount++) instead of aliasing a raw pointer. The deleter only runs once the last `Tensor` sharing that buffer is destroyed — automatic, no manual ownership tracking needed. `~Tensor()`'s manual `cudaFree` call was removed entirely; it's now redundant. Every other place that used `device_data_` directly (kernel call sites, `device_data()`/`mutable_device_data()` accessors, the `to(CPU)` memcpy) switched to `.get()` to keep passing raw pointers to the CUDA kernels, unchanged.

## Net result

Same class of lesson as ADR 0004: an op being correct in isolation (or even correct when chained once) doesn't mean its *memory ownership* is correct once real model code keeps multiple views of the same buffer alive at once. `TransformerBlock::forward()` + `backward()` now runs correctly on CUDA end-to-end, verified against the CPU reference and confirmed error-free under `compute-sanitizer`.
