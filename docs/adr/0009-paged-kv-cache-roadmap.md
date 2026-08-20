# PagedKVCache: staged roadmap toward the kv-aware-inference reference design

**Status**: steps 1-2 done (CPU+CUDA, tested). Step 3: LRU done (CPU+CUDA); sliding window,
LFU, cost-based deliberately not built — see note at the end.

## Context

`KVCache` (`include/infer/kv_cache.h`, `src/KVCache/KVCache.cpp`) works and is CUDA-correct (see the
`grow()` CUDA fix and its test, `tests/test_m9_cuda.cpp`), but reallocates and copies the entire
cache on every `append()` — total copying grows ~O(n²) over a full generation. This is a real,
recognized baseline design: https://github.com/qimcis/kv-aware-inference calls the equivalent
approach `NaiveCache` ("straight-line buffer management without eviction policies, storing
sequences contiguously and expanding as needed") — exactly what ours does.

That same reference repo's main `KVCache` class is the actual target: block-based GPU allocation,
multiple eviction policies (LRU, sliding window, LFU, cost-based), and explicit support for
concurrent multi-sequence inference workloads — the real modern paged-attention shape (same
family as vLLM's PagedAttention), reproduced as an educational "toy inference engine."

**Explicit decision**: the end goal is that full design, not a permanently-smaller substitute.
The existing `KVCache` (naive) is being kept as-is, unrenamed, specifically so it stays available
to reference/diff against as the new `PagedKVCache` (`include/infer/paged_kv_cache.h`,
currently empty) is built up in stages.

## Staged plan

1. **Single-sequence, fixed block size, no eviction** (current step). Pre-allocate one buffer
   sized for `max_seq_len` upfront (constructor-provided), rounded up to `block_size` chunks.
   `append()` writes new tokens into the next open position — no reallocation, no copying old
   data. `get_k()`/`get_v()` return a zero-copy strided view of just the valid `current_len_`
   portion (same mechanism `reshape()`/`permute()` already use: share the buffer, different
   shape/strides), not a fresh copy.

2. **Multi-sequence support**. Move from "one buffer per sequence" to a shared block pool;
   each sequence holds a block table (list of which blocks in the pool it owns) instead of
   owning a dedicated contiguous buffer. This is what actually unlocks continuous batching later.

3. **Eviction policies**, matching the reference's four: LRU first (simplest), then sliding
   window, LFU, cost-based — needed once the shared pool can fill up across multiple sequences.

## Integration into real generation (done)

`KVBlockPool` was correct and tested but completely disconnected from actual generation --
`SelfAttention::forward`, `TransformerBlock::forward`, and `CharModel::forward`/`generate()` all
still hard-required `KVCache&`. Swapped all three to `KVBlockPool&` (+ `sequence_id`), replacing
`KVCache` as what real generation actually runs on. `KVCache`/`PagedKVCache` stay in the codebase,
untouched, as standalone tested comparison classes -- they just don't drive generation anymore.

One API wrinkle: `KVBlockPool::get_k()` throws on an unregistered `sequence_id` rather than
returning null the way `KVCache::get_k()` did, so `CharModel::forward()`'s KVBlockPool overload
gained an explicit `start_pos` parameter (tracked by the caller, `generate()`) instead of trying
to infer position from cache state.

## Why the other three eviction policies weren't built

Re-reading the reference's actual `select_victim()`/`touch_block()` code (not just the class
description) showed their "four policies" reduce to two real mechanisms: LRU and sliding window
share the exact same branch (`lru_list_` front-pop) — sliding window adds no new code in their
implementation. LFU and cost-based also share one mechanism (a decayed counter, `x = x*decay + 1.0`,
evict the minimum) — as shown, `value_` (cost-based) updates with the identical formula as `freq_`
(LFU), so their "cost" isn't weighted by anything distinct from access frequency. Given that, LRU
(already built) covers the recency-list mechanism, and the decayed-counter mechanism (which would
give LFU + cost-based "for free," same as the reference) was deliberately left unbuilt as a
known, understood gap rather than busywork — not because it's hard, but because LRU alone is
enough to demonstrate and use the eviction system, and the marginal learning from adding a second,
structurally similar mechanism was judged not worth the time here. Revisit if a real reason to
compare LRU vs. frequency-based eviction on this project's actual workload ever comes up.

## Why staged instead of building the full design at once

Session pattern before this decision was jumping across many small M10 items (caching allocator,
loss scaling, bf16 primitives, `KVCache::grow()`'s CUDA fix) without going deep on any one. This
was called out directly and corrected: commit to one real systems project (paged KV cache) and
stay on it, rather than switching again mid-way. Staging is about sequencing that one project
correctly (get single-sequence paging correct and tested before adding the multi-sequence +
eviction complexity on top), not about scoping down the actual goal.
