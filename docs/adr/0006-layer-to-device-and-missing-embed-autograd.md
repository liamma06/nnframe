# Layer::to(Device) + a missing autograd-wiring bug found while testing CharModel

**Status**: accepted (fixed, verified with compute-sanitizer, CharModel matches CPU on CUDA for both 1-block and multi-block stacking)

## Context

Every CUDA correctness test up through `TransformerBlock` had to manually re-type the target's entire `forward()` logic by hand — pull each weight out via `parameters()`, move it to CUDA individually, then replay the same sequence of ops `forward()` already does internally, just to get a CUDA-resident computation graph to compare against the CPU reference. This worked, but didn't scale: testing `CharModel` (embedding + positional encoding + N stacked transformer blocks + output head) would have meant hand-replicating an entire model's forward pass, growing linearly with the number of blocks.

The actual gap: no layer had any way to move its own internal weights to CUDA in place. `Tensor::to(Device)` moves a single tensor; nothing did the equivalent for a whole `Layer`.

## The fix: `Layer::to(Device)`

Added two things to the `Layer` base class (`include/modules/layer.h`):

```cpp
virtual void set_parameters(const std::vector<TensorPtr>& params) { (void)params; }

void to(Device device){
    std::vector<TensorPtr> params = parameters();
    std::vector<TensorPtr> moved;
    for (auto& p : params){
        TensorPtr m = p->to(device);
        m->set_requires_grad(p->requires_grad());
        moved.push_back(m);
    }
    set_parameters(moved);
}
```

`parameters()` already existed and could *read* a layer's weights out; `set_parameters()` is the missing *write* side — each subclass overrides it to reassign its own private members. `to()` is written once, generically, on top of both: move every parameter, then hand the moved copies back via `set_parameters()` (dispatched correctly at runtime via `virtual`, regardless of what concrete layer `to()` is called on).

Implemented `set_parameters()` for every layer with weights:
- `Linear`, `LayerNorm`, `Embed`, `SelfAttention` — reassign their own 1-4 private members directly.
- `TransformerBlock`, `CharModel` — composite layers; slice the incoming vector using each sub-layer's own `parameters().size()`, and delegate to that sub-layer's `set_parameters()`. `CharModel`'s version loops over `transformer_blocks_`, so it works for any number of stacked blocks with no code change.

With this in place, testing an entire model on CUDA collapsed to:

```cpp
TensorPtr output_cpu = model.forward(input_cpu);
output_cpu->backward();
std::vector<TensorPtr> params_cpu = model.parameters(); // capture before to() replaces them

model.to(Device::CUDA);
TensorPtr output_gpu = model.forward(input_cpu->to(Device::CUDA));
output_gpu->backward();
// compare output_gpu / params_gpu[i]->grad() against output_cpu / params_cpu[i]->grad()
```

No manual replication, no growing linearly with model depth — same ~35 lines regardless of `num_transformer_blocks`.

## The bug this immediately exposed: `Embed::forward()`'s CUDA branch had no autograd wiring

The first time this ran (`tests/test_m9_charmodel.cpp`, 1-block `CharModel`), it crashed with a segfault at the very first gradient comparison — `params_gpu[0]->grad()` (the embedding table) dereferenced a null `grad_`.

Root cause, in `include/modules/embed.h`:

```cpp
if (input->device() == Device::CUDA){
    ...
    embed_cuda(embedding_matrix_->device_data(), input->device_data(), d_out, seq_len, embedding_dim_);
    return Tensor::from_device_ptr(d_out, {seq_len, embedding_dim_}, {embedding_dim_, 1});   // <-- returns here
}
```

The CUDA branch computed the correct forward values and returned immediately — no `set_requires_grad`, no `set_inputs`, no `set_grad_fn`. Compare to the CPU branch a few lines below, which has all three. The embedding lookup was a dead end in the autograd graph on CUDA: `backward()`'s DFS never reached `embedding_matrix_` at all, so its gradient was never computed, and nothing anywhere caught this at forward time (there's no way to fail loudly for "this tensor has no `grad_fn_`" — it just silently doesn't participate in backprop).

This is exactly the same shape of bug as ADR 0004/0005: an op that looked "CUDA-capable" (it ran, and its output values were correct) but was never actually exercised as part of a *real trained model*, because every prior embedding test called `embed_cuda`/`embed_backward_cuda` directly as raw kernel functions, never through the real `Embed::forward()` chained inside an actual model's backward pass.

**Fix**: added the same wiring the CPU branch already has:

```cpp
auto output_tensor = Tensor::from_device_ptr(d_out, {seq_len, embedding_dim_}, {embedding_dim_, 1});
auto self = embedding_matrix_;
output_tensor->set_requires_grad(self->requires_grad());
output_tensor->set_inputs(std::vector<TensorPtr>{self});
size_t embedding_dim = embedding_dim_;
output_tensor->set_grad_fn([self, input, seq_len, embedding_dim](const Tensor& upstream){
    if (self->requires_grad()){
        self->init_grad();
        embed_backward_cuda(input->device_data(), upstream.device_data(), self->grad().mutable_device_data(), seq_len, embedding_dim);
    }
});
return output_tensor;
```

## Net result

`CharModel` (the mini-GPT: embedding -> positional encoding -> N transformer blocks -> output head) now runs correctly on CUDA end-to-end using its real, unmodified `forward()` -- verified against the CPU reference for both a single block and three stacked blocks, output and every parameter's gradient matching within `1e-3`. Confirmed clean under `compute-sanitizer`.

The recurring lesson across ADRs 0004-0006: this codebase's CUDA ops are consistently correct in isolation but the *integration points* (chaining views, sharing device memory, wiring autograd through a real model) are where the actual bugs hide -- and they only surface once something forces the integration to actually happen.
