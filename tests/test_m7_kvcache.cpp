#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "infer/kv_cache.h"
#include "infer/paged_kv_cache.h"
#include "infer/kv_block_pool.h"
#include "modules/attention.h"
#include "infer/sampler.h"
#include "modules/char_model.h"
#include <random>
#include <stdexcept>

namespace {
    // pulls out rows [start, start+count) as a fresh [count, embed_dim] tensor
    TensorPtr extract_rows(const TensorPtr& t, size_t start, size_t count, size_t embed_dim) {
        std::vector<scalar_t> vals;
        for (size_t i = 0; i < count; i++)
            for (size_t j = 0; j < embed_dim; j++)
                vals.push_back(t->at({start + i, j}));
        return std::make_shared<Tensor>(std::vector<size_t>{count, embed_dim}, vals);
    }
}

TEST_CASE("KVCache: first append stores directly, no growth") {
    KVCache cache;
    // num_heads=2, seq_len=1, head_dim=2
    auto k = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{
        1, 2,    // head0, token0
        10, 20   // head1, token0
    });
    auto v = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{
        100, 200,
        1000, 2000
    });

    cache.append(k, v);

    auto got_k = cache.get_k();
    CHECK(got_k->shape()[0] == 2);
    CHECK(got_k->shape()[1] == 1);
    CHECK(got_k->shape()[2] == 2);
    CHECK(got_k->at({0, 0, 0}) == 1.0f);
    CHECK(got_k->at({0, 0, 1}) == 2.0f);
    CHECK(got_k->at({1, 0, 0}) == 10.0f);
    CHECK(got_k->at({1, 0, 1}) == 20.0f);

    auto got_v = cache.get_v();
    CHECK(got_v->at({0, 0, 0}) == 100.0f);
    CHECK(got_v->at({1, 0, 1}) == 2000.0f);
}

TEST_CASE("KVCache: prefill (multi-token) then decode (single-token) grows correctly") {
    KVCache cache;

    // prefill: num_heads=2, seq_len=3, head_dim=2
    auto k_prefill = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 2}, std::vector<scalar_t>{
        1, 2,  3, 4,  5, 6,        // head0: token0, token1, token2
        10, 20,  30, 40,  50, 60   // head1: token0, token1, token2
    });
    auto v_prefill = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 2}, std::vector<scalar_t>{
        100, 200,  300, 400,  500, 600,
        1000, 2000,  3000, 4000,  5000, 6000
    });
    cache.append(k_prefill, v_prefill);

    auto k1 = cache.get_k();
    CHECK(k1->shape()[1] == 3);
    CHECK(k1->at({0, 2, 0}) == 5.0f);
    CHECK(k1->at({0, 2, 1}) == 6.0f);
    CHECK(k1->at({1, 0, 0}) == 10.0f);

    // decode: 1 new token
    auto k_decode = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{
        7, 8,    // head0, token3
        70, 80   // head1, token3
    });
    auto v_decode = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{
        700, 800,
        7000, 8000
    });
    cache.append(k_decode, v_decode);

    auto k2 = cache.get_k();
    CHECK(k2->shape()[0] == 2);
    CHECK(k2->shape()[1] == 4);
    CHECK(k2->shape()[2] == 2);

    // old prefill data must be unchanged
    CHECK(k2->at({0, 0, 0}) == 1.0f);
    CHECK(k2->at({0, 0, 1}) == 2.0f);
    CHECK(k2->at({0, 1, 0}) == 3.0f);
    CHECK(k2->at({0, 2, 0}) == 5.0f);
    CHECK(k2->at({1, 1, 1}) == 40.0f);

    // new decode token appended at the end, per head
    CHECK(k2->at({0, 3, 0}) == 7.0f);
    CHECK(k2->at({0, 3, 1}) == 8.0f);
    CHECK(k2->at({1, 3, 0}) == 70.0f);
    CHECK(k2->at({1, 3, 1}) == 80.0f);

    auto v2 = cache.get_v();
    CHECK(v2->shape()[1] == 4);
    CHECK(v2->at({0, 0, 0}) == 100.0f);   // old v preserved
    CHECK(v2->at({1, 2, 1}) == 6000.0f);  // old v preserved
    CHECK(v2->at({0, 3, 0}) == 700.0f);   // new v appended
    CHECK(v2->at({1, 3, 1}) == 8000.0f);  // new v appended
}

TEST_CASE("PagedKVCache matches naive KVCache across prefill + multiple decode steps") {
    size_t heads = 2, head_dim = 3;
    std::vector<size_t> chunk_seq_lens = {4, 1, 1, 1}; // prefill of 4, then 3 decode steps

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<TensorPtr> k_chunks, v_chunks;
    for (size_t seq_len : chunk_seq_lens) {
        std::vector<scalar_t> k_data(heads * seq_len * head_dim);
        std::vector<scalar_t> v_data(heads * seq_len * head_dim);
        for (auto& x : k_data) x = dist(rng);
        for (auto& x : v_data) x = dist(rng);
        k_chunks.push_back(std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, k_data));
        v_chunks.push_back(std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, v_data));
    }

    KVCache naive_cache;
    for (size_t i = 0; i < k_chunks.size(); i++) naive_cache.append(k_chunks[i], v_chunks[i]);

    PagedKVCache paged_cache(heads, head_dim, /*max_seq_len=*/16, /*block_size=*/4);
    for (size_t i = 0; i < k_chunks.size(); i++) paged_cache.append(k_chunks[i], v_chunks[i]);

    TensorPtr k_naive = naive_cache.get_k();
    TensorPtr v_naive = naive_cache.get_v();
    TensorPtr k_paged = paged_cache.get_k();
    TensorPtr v_paged = paged_cache.get_v();

    REQUIRE(k_paged->shape() == k_naive->shape());
    REQUIRE(v_paged->shape() == v_naive->shape());

    for (size_t h = 0; h < heads; h++) {
        for (size_t i = 0; i < k_naive->shape()[1]; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                CHECK(k_paged->at({h, i, j}) == doctest::Approx(k_naive->at({h, i, j})));
                CHECK(v_paged->at({h, i, j}) == doctest::Approx(v_naive->at({h, i, j})));
            }
        }
    }
}

TEST_CASE("PagedKVCache throws when appending past capacity") {
    PagedKVCache cache(/*num_heads=*/1, /*head_dim=*/2, /*max_seq_len=*/4, /*block_size=*/4);

    auto k = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 2}, std::vector<scalar_t>{1,2, 3,4, 5,6, 7,8});
    auto v = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 2}, std::vector<scalar_t>{1,2, 3,4, 5,6, 7,8});
    cache.append(k, v); // fills exactly to capacity (4), should be fine

    auto k_overflow = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{9, 10});
    auto v_overflow = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{9, 10});
    CHECK_THROWS_AS(cache.append(k_overflow, v_overflow), std::runtime_error);
}

TEST_CASE("SelfAttention: cached prefill+decode matches plain full-sequence forward") {
    const size_t embed_dim = 8, num_heads = 2, seq_len = 5;
    SelfAttention attn(embed_dim, num_heads);

    // deterministic, varied input: row i, col j = i*0.1 + j*0.01
    std::vector<scalar_t> x_data;
    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < embed_dim; j++)
            x_data.push_back(static_cast<scalar_t>(i) * 0.1f + static_cast<scalar_t>(j) * 0.01f);
    auto x_full = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim}, x_data);

    // ground truth: one shot, no cache, no masking special-casing
    auto ground_truth = attn.forward(x_full);

    // cached: prefill first 3 tokens, then decode tokens 3 and 4 one at a time
    KVBlockPool pool(num_heads, embed_dim / num_heads, /*max_blocks=*/4, /*block_size=*/4);
    size_t sequence_id = 0;
    auto x_prefill = extract_rows(x_full, 0, 3, embed_dim);
    auto prefill_out = attn.forward(x_prefill, pool, sequence_id);

    CHECK(prefill_out->shape()[0] == 3);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < embed_dim; j++)
            CHECK(prefill_out->at({i, j}) == doctest::Approx(ground_truth->at({i, j})).epsilon(1e-4));

    auto x_tok3 = extract_rows(x_full, 3, 1, embed_dim);
    auto decode_out3 = attn.forward(x_tok3, pool, sequence_id);
    CHECK(decode_out3->shape()[0] == 1);
    for (size_t j = 0; j < embed_dim; j++)
        CHECK(decode_out3->at({0, j}) == doctest::Approx(ground_truth->at({3, j})).epsilon(1e-4));

    auto x_tok4 = extract_rows(x_full, 4, 1, embed_dim);
    auto decode_out4 = attn.forward(x_tok4, pool, sequence_id);
    CHECK(decode_out4->shape()[0] == 1);
    for (size_t j = 0; j < embed_dim; j++)
        CHECK(decode_out4->at({0, j}) == doctest::Approx(ground_truth->at({4, j})).epsilon(1e-4));
}

TEST_CASE("Sampler: top_k=1 always picks the argmax token, deterministically") {
    // index 3 has by far the highest logit
    auto logits = std::make_shared<Tensor>(std::vector<size_t>{5}, std::vector<scalar_t>{1, 2, 3, 10, 2});
    Sampler sampler(42);

    for (int trial = 0; trial < 20; trial++) {
        size_t chosen = sampler.sample(logits, 1.0f, 1);
        CHECK(chosen == 3);
    }
}

TEST_CASE("Sampler: never picks a token outside the top_k set") {
    // sorted descending: idx3(10), idx2(3), idx1(2), idx4(2), idx0(1)
    // top_k=2 should only ever return idx3 or idx2
    auto logits = std::make_shared<Tensor>(std::vector<size_t>{5}, std::vector<scalar_t>{1, 2, 3, 10, 2});
    Sampler sampler(42);

    bool saw_idx3 = false, saw_idx2 = false;
    for (int trial = 0; trial < 100; trial++) {
        size_t chosen = sampler.sample(logits, 2.0f, 2); // higher temperature so idx2 gets picked sometimes too
        CHECK((chosen == 2 || chosen == 3));
        if (chosen == 3) saw_idx3 = true;
        if (chosen == 2) saw_idx2 = true;
    }
    // over 100 trials with 2 valid candidates, both should show up at least once
    CHECK(saw_idx3);
    CHECK(saw_idx2);
}

TEST_CASE("CharModel::generate - correct length, prompt preserved, valid token ids") {
    const size_t vocab_size = 10, embed_dim = 8, num_heads = 2, num_blocks = 2;
    CharModel model(vocab_size, embed_dim, num_heads, num_blocks);
    std::vector<size_t> prompt = {1, 2, 3};
    Sampler sampler(42);

    auto result = model.generate(prompt, sampler, 5, 1.0f, 3);

    CHECK(result.size() == prompt.size() + 5);
    for (size_t i = 0; i < prompt.size(); i++)
        CHECK(result[i] == prompt[i]); // prompt preserved unchanged at the start
    for (size_t id : result)
        CHECK(id < vocab_size); // every generated id is a valid token
}

TEST_CASE("CharModel::generate - same seed produces identical output") {
    const size_t vocab_size = 10, embed_dim = 8, num_heads = 2, num_blocks = 2;
    CharModel model(vocab_size, embed_dim, num_heads, num_blocks);
    std::vector<size_t> prompt = {1, 2, 3};

    Sampler sampler1(123), sampler2(123);
    auto result1 = model.generate(prompt, sampler1, 5, 1.0f, 3);
    auto result2 = model.generate(prompt, sampler2, 5, 1.0f, 3);

    CHECK(result1 == result2);
}

TEST_CASE("CharModel::generate - top_k=1 matches argmax from plain non-cached forward") {
    // strongest correctness check: proves the cached path (embed/pos/blocks/lm_head,
    // all through the KVCache) produces the same result as the plain forward, not
    // just that SelfAttention alone does
    const size_t vocab_size = 10, embed_dim = 8, num_heads = 2, num_blocks = 2;
    CharModel model(vocab_size, embed_dim, num_heads, num_blocks);
    std::vector<size_t> prompt = {1, 2, 3};

    // ground truth: plain forward, manual argmax on the last row
    std::vector<scalar_t> prompt_data;
    for (auto id : prompt) prompt_data.push_back(static_cast<scalar_t>(id));
    auto prompt_tensor = Tensor::from_vector(prompt_data);
    auto logits = model.forward(prompt_tensor);

    size_t last_row = logits->shape()[0] - 1;
    size_t expected_argmax = 0;
    scalar_t best = logits->at({last_row, 0});
    for (size_t j = 1; j < vocab_size; j++) {
        scalar_t v = logits->at({last_row, j});
        if (v > best) { best = v; expected_argmax = j; }
    }

    Sampler sampler(42);
    auto result = model.generate(prompt, sampler, 1, 1.0f, 1); // top_k=1 -> deterministic argmax

    CHECK(result.size() == prompt.size() + 1);
    CHECK(result.back() == expected_argmax);
}

TEST_CASE("KVBlockPool: two interleaved sequences don't corrupt each other, match independent PagedKVCache") {
    size_t heads = 2, head_dim = 3, block_size = 4;

    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto make_chunk = [&](size_t seq_len) {
        std::vector<scalar_t> k_data(heads * seq_len * head_dim);
        std::vector<scalar_t> v_data(heads * seq_len * head_dim);
        for (auto& x : k_data) x = dist(rng);
        for (auto& x : v_data) x = dist(rng);
        return std::make_pair(
            std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, k_data),
            std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, v_data)
        );
    };

    // seq0: prefill 5, then 2 decode steps (spans multiple blocks, block_size=4)
    // seq1: prefill 3, then 2 decode steps
    auto seq0_chunks = std::vector<std::pair<TensorPtr, TensorPtr>>{make_chunk(5), make_chunk(1), make_chunk(1)};
    auto seq1_chunks = std::vector<std::pair<TensorPtr, TensorPtr>>{make_chunk(3), make_chunk(1), make_chunk(1)};

    // ground truth: each sequence gets its own independent, already-tested PagedKVCache
    PagedKVCache ref0(heads, head_dim, /*max_seq_len=*/16, block_size);
    PagedKVCache ref1(heads, head_dim, /*max_seq_len=*/16, block_size);

    // shared pool: enough blocks for both sequences (seq0 needs 2, seq1 needs 2 -> 4 total, give headroom)
    KVBlockPool pool(heads, head_dim, /*max_blocks=*/8, block_size);
    size_t seq0_id = 0, seq1_id = 1;

    // interleave appends across the two sequences, exercising that they don't stomp on each other
    for (size_t i = 0; i < seq0_chunks.size(); i++) {
        ref0.append(seq0_chunks[i].first, seq0_chunks[i].second);
        pool.append(seq0_id, seq0_chunks[i].first, seq0_chunks[i].second);

        ref1.append(seq1_chunks[i].first, seq1_chunks[i].second);
        pool.append(seq1_id, seq1_chunks[i].first, seq1_chunks[i].second);
    }

    TensorPtr k0_ref = ref0.get_k(), v0_ref = ref0.get_v();
    TensorPtr k0_pool = pool.get_k(seq0_id), v0_pool = pool.get_v(seq0_id);
    TensorPtr k1_ref = ref1.get_k(), v1_ref = ref1.get_v();
    TensorPtr k1_pool = pool.get_k(seq1_id), v1_pool = pool.get_v(seq1_id);

    REQUIRE(k0_pool->shape() == k0_ref->shape());
    REQUIRE(k1_pool->shape() == k1_ref->shape());

    for (size_t h = 0; h < heads; h++) {
        for (size_t i = 0; i < k0_ref->shape()[1]; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                CHECK(k0_pool->at({h, i, j}) == doctest::Approx(k0_ref->at({h, i, j})));
                CHECK(v0_pool->at({h, i, j}) == doctest::Approx(v0_ref->at({h, i, j})));
            }
        }
        for (size_t i = 0; i < k1_ref->shape()[1]; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                CHECK(k1_pool->at({h, i, j}) == doctest::Approx(k1_ref->at({h, i, j})));
                CHECK(v1_pool->at({h, i, j}) == doctest::Approx(v1_ref->at({h, i, j})));
            }
        }
    }
}

TEST_CASE("KVBlockPool: quantized pool matches unquantized pool within int8 error") {
    size_t heads = 2, head_dim = 3, block_size = 4;

    std::mt19937 rng(202);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto make_chunk = [&](size_t seq_len) {
        std::vector<scalar_t> k_data(heads * seq_len * head_dim);
        std::vector<scalar_t> v_data(heads * seq_len * head_dim);
        for (auto& x : k_data) x = dist(rng);
        for (auto& x : v_data) x = dist(rng);
        return std::make_pair(
            std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, k_data),
            std::make_shared<Tensor>(std::vector<size_t>{heads, seq_len, head_dim}, v_data)
        );
    };

    // prefill 5 (spans 2 blocks), then 2 decode steps
    auto chunks = std::vector<std::pair<TensorPtr, TensorPtr>>{make_chunk(5), make_chunk(1), make_chunk(1)};

    KVBlockPool plain_pool(heads, head_dim, /*max_blocks=*/8, block_size);
    KVBlockPool quant_pool(heads, head_dim, /*max_blocks=*/8, block_size, Device::CPU, /*use_quantization=*/true);
    size_t seq_id = 0;

    for (auto& chunk : chunks) {
        plain_pool.append(seq_id, chunk.first, chunk.second);
        quant_pool.append(seq_id, chunk.first, chunk.second);
    }

    TensorPtr k_plain = plain_pool.get_k(seq_id), v_plain = plain_pool.get_v(seq_id);
    TensorPtr k_quant = quant_pool.get_k(seq_id), v_quant = quant_pool.get_v(seq_id);

    REQUIRE(k_quant->shape() == k_plain->shape());

    for (size_t h = 0; h < heads; h++) {
        for (size_t i = 0; i < k_plain->shape()[1]; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                // int8 with per-token scale: worst-case error is scale ~= max_abs/127, values are in [-1, 1]
                CHECK(std::fabs(k_quant->at({h, i, j}) - k_plain->at({h, i, j})) < 0.02f);
                CHECK(std::fabs(v_quant->at({h, i, j}) - v_plain->at({h, i, j})) < 0.02f);
            }
        }
    }
}

TEST_CASE("KVBlockPool: evicts the least-recently-used sequence's block when full") {
    // 1 block total -- sequence 0 fills it, then sequence 1 needs a block and must evict it
    KVBlockPool pool(/*num_heads=*/1, /*head_dim=*/2, /*max_blocks=*/1, /*block_size=*/2);

    auto k0 = std::make_shared<Tensor>(std::vector<size_t>{1, 2, 2}, std::vector<scalar_t>{1,2, 3,4});
    auto v0 = std::make_shared<Tensor>(std::vector<size_t>{1, 2, 2}, std::vector<scalar_t>{1,2, 3,4});
    pool.append(0, k0, v0); // fills the only block with sequence 0's data

    CHECK(pool.get_k(0)->shape()[1] == 2); // sequence 0 has its 2 tokens

    // sequence 1 needs a block, none are free -- should evict sequence 0's block, not throw
    auto k1 = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{5, 6});
    auto v1 = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{5, 6});
    pool.append(1, k1, v1);

    // sequence 1 now owns the (only) block
    CHECK(pool.get_k(1)->shape()[1] == 1);
    CHECK(pool.get_k(1)->at({0, 0, 0}) == 5.0f);
    CHECK(pool.get_k(1)->at({0, 0, 1}) == 6.0f);

    // sequence 0 lost its block to eviction -- it should now have 0 tokens
    CHECK(pool.get_k(0)->shape()[1] == 0);
}

TEST_CASE("KVBlockPool: throws only when there is truly nothing left to evict") {
    // max_blocks=0 -- no blocks exist at all, free or otherwise
    KVBlockPool pool(/*num_heads=*/1, /*head_dim=*/2, /*max_blocks=*/0, /*block_size=*/2);

    auto k = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{1, 2});
    auto v = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 2}, std::vector<scalar_t>{1, 2});
    CHECK_THROWS_AS(pool.append(0, k, v), std::runtime_error);
}
