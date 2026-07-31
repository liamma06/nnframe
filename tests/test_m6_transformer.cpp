#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "modules/embed.h"
#include "modules/layernorm.h"
#include "modules/attention.h"
#include "modules/transformer_block.h"
#include "modules/pos_embed.h"
#include "modules/char_model.h"
#include <cmath>

// ── Softmax ──────────────────────────────────────────────────────────────────

TEST_CASE("softmax rows sum to 1") {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{2, 4},
        std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f,
                              0.5f, 1.5f, 2.5f, 0.1f}
    );
    auto out = x->softmax(1);
    scalar_t row0 = out->data()[0] + out->data()[1] + out->data()[2] + out->data()[3];
    scalar_t row1 = out->data()[4] + out->data()[5] + out->data()[6] + out->data()[7];
    CHECK(row0 == doctest::Approx(1.0f).epsilon(1e-5f));
    CHECK(row1 == doctest::Approx(1.0f).epsilon(1e-5f));
}

TEST_CASE("softmax output between 0 and 1") {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<scalar_t>{-1.0f, 0.0f, 1.0f}
    );
    auto out = x->softmax(1);
    for (size_t i = 0; i < 3; i++) {
        CHECK(out->data()[i] > 0.0f);
        CHECK(out->data()[i] < 1.0f);
    }
}

TEST_CASE("softmax backward grad flows") {
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<scalar_t>{1.0f, 2.0f, 3.0f}
    );
    x->set_requires_grad(true);
    auto out = x->softmax(1);
    // non-uniform upstream via MSE against a one-hot target
    auto target = std::make_shared<Tensor>(
        std::vector<size_t>{1, 3},
        std::vector<scalar_t>{1.0f, 0.0f, 0.0f}
    );
    auto diff = out->sub(target);
    auto loss = diff->mul(diff)->mean();
    loss->backward();
    bool any_nonzero = false;
    for (size_t i = 0; i < 3; i++)
        if (std::abs(x->grad().data()[i]) > 1e-7f) any_nonzero = true;
    CHECK(any_nonzero);
}

// ── Embedding ─────────────────────────────────────────────────────────────────

TEST_CASE("embedding forward output shape") {
    Embed emb(10, 4); // vocab=10, embed_dim=4
    auto ids = std::make_shared<Tensor>(
        std::vector<size_t>{3},
        std::vector<scalar_t>{0.0f, 2.0f, 5.0f}
    );
    auto out = emb.forward(ids);
    CHECK(out->shape()[0] == 3);
    CHECK(out->shape()[1] == 4);
}

TEST_CASE("embedding forward same id gives same row") {
    Embed emb(10, 4);
    auto ids = std::make_shared<Tensor>(
        std::vector<size_t>{2},
        std::vector<scalar_t>{3.0f, 3.0f} // same id twice
    );
    auto out = emb.forward(ids);
    for (size_t j = 0; j < 4; j++)
        CHECK(out->data()[j] == doctest::Approx(out->data()[4 + j]));
}

TEST_CASE("embedding backward grad accumulates to correct row") {
    Embed emb(5, 3);
    auto ids = std::make_shared<Tensor>(
        std::vector<size_t>{2},
        std::vector<scalar_t>{1.0f, 1.0f} // id=1 appears twice
    );
    auto out = emb.forward(ids);
    out->backward();
    // row 1 of embedding table should have non-zero grad
    auto& params = emb.parameters();
    bool nonzero = false;
    for (size_t j = 0; j < 3; j++)
        if (std::abs(params[0]->grad().data()[1 * 3 + j]) > 1e-7f) nonzero = true;
    CHECK(nonzero);
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

TEST_CASE("layernorm output has near-zero mean per row") {
    LayerNorm ln(4);
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{2, 4},
        std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f,
                              10.0f, 20.0f, 30.0f, 40.0f}
    );
    auto out = ln.forward(x);
    // mean of each row should be ~0 (gamma=1, beta=0 initially)
    scalar_t mean0 = (out->data()[0]+out->data()[1]+out->data()[2]+out->data()[3]) / 4.0f;
    scalar_t mean1 = (out->data()[4]+out->data()[5]+out->data()[6]+out->data()[7]) / 4.0f;
    CHECK(mean0 == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(mean1 == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("layernorm output shape matches input") {
    LayerNorm ln(8);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 1.0f);
    auto out = ln.forward(x);
    CHECK(out->shape()[0] == 4);
    CHECK(out->shape()[1] == 8);
}

// ── SelfAttention ─────────────────────────────────────────────────────────────

TEST_CASE("attention output shape matches input") {
    SelfAttention attn(8, 1);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 0.1f);
    auto out = attn.forward(x);
    CHECK(out->shape()[0] == 4);
    CHECK(out->shape()[1] == 8);
}

TEST_CASE("attention has 4 parameter matrices") {
    SelfAttention attn(8, 1);
    CHECK(attn.parameters().size() == 4);
}

// ── PositionalEncoding ───────────────────────────────────────────────────────

TEST_CASE("positional encoding output shape matches input") {
    PositionalEmbed pe;
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 0.0f);
    auto out = pe.forward(x);
    CHECK(out->shape()[0] == 4);
    CHECK(out->shape()[1] == 8);
}

TEST_CASE("positional encoding differs across positions") {
    PositionalEmbed pe;
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 0.0f);
    auto out = pe.forward(x);
    // row 0 and row 1 should differ
    bool any_different = false;
    for (size_t j = 0; j < 8; j++)
        if (std::abs(out->data()[j] - out->data()[8 + j]) > 1e-6f) any_different = true;
    CHECK(any_different);
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TEST_CASE("transformer block output shape matches input") {
    TransformerBlock block(8, 1);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 0.1f);
    auto out = block.forward(x);
    CHECK(out->shape()[0] == 4);
    CHECK(out->shape()[1] == 8);
}

TEST_CASE("transformer block has correct parameter count") {
    TransformerBlock block(8, 1);
    // attn: 4, ln1: 2, ln2: 2, linear1: 2, linear2: 2
    CHECK(block.parameters().size() == 12);
}

TEST_CASE("transformer block output differs from input") {
    TransformerBlock block(8, 1);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4, 8}, 0.1f);
    auto out = block.forward(x);
    bool any_different = false;
    for (size_t i = 0; i < out->numel(); i++)
        if (std::abs(out->data()[i] - x->data()[i]) > 1e-6f) any_different = true;
    CHECK(any_different);
}

// ── CharModel ─────────────────────────────────────────────────────────────────

TEST_CASE("charmodel output shape is [seq_len, vocab_size]") {
    CharModel model(16, 8, 1, 2); // vocab=16, embed=8, heads=1, blocks=2
    auto ids = std::make_shared<Tensor>(
        std::vector<size_t>{4},
        std::vector<scalar_t>{0.0f, 1.0f, 2.0f, 3.0f}
    );
    auto out = model.forward(ids);
    CHECK(out->shape()[0] == 4);
    CHECK(out->shape()[1] == 16);
}

TEST_CASE("charmodel has parameters") {
    CharModel model(16, 8, 1, 2);
    CHECK(model.parameters().size() > 0);
}
