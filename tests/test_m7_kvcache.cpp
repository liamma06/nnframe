#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "infer/kv_cache.h"
#include "modules/attention.h"
#include "infer/sampler.h"

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
    KVCache cache;
    auto x_prefill = extract_rows(x_full, 0, 3, embed_dim);
    auto prefill_out = attn.forward(x_prefill, cache);

    CHECK(prefill_out->shape()[0] == 3);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < embed_dim; j++)
            CHECK(prefill_out->at({i, j}) == doctest::Approx(ground_truth->at({i, j})).epsilon(1e-4));

    auto x_tok3 = extract_rows(x_full, 3, 1, embed_dim);
    auto decode_out3 = attn.forward(x_tok3, cache);
    CHECK(decode_out3->shape()[0] == 1);
    for (size_t j = 0; j < embed_dim; j++)
        CHECK(decode_out3->at({0, j}) == doctest::Approx(ground_truth->at({3, j})).epsilon(1e-4));

    auto x_tok4 = extract_rows(x_full, 4, 1, embed_dim);
    auto decode_out4 = attn.forward(x_tok4, cache);
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
