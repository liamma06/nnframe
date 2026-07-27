#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "layers/gelu.h"
#include "layers/sequential.h"

TEST_CASE("linear forward shape") {
    Linear lin(4, 8);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, 1.0f);
    auto y = lin.forward(x);
    CHECK(y->shape()[0] == 2);
    CHECK(y->shape()[1] == 8);
}

TEST_CASE("relu forward") {
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4}, std::vector<scalar_t>{-2.0f, -1.0f, 0.0f, 3.0f});
    ReLu relu;
    auto y = relu.forward(x);
    CHECK(y->data()[0] == 0.0f);
    CHECK(y->data()[1] == 0.0f);
    CHECK(y->data()[2] == 0.0f);
    CHECK(y->data()[3] == 3.0f);
}

TEST_CASE("relu backward") {
    auto x = std::make_shared<Tensor>(std::vector<size_t>{4}, std::vector<scalar_t>{-2.0f, 1.0f, -0.5f, 3.0f});
    x->set_requires_grad(true);
    ReLu relu;
    auto y = relu.forward(x);
    y->backward();
    // grad passes through where input > 0, blocked elsewhere
    CHECK(x->grad().data()[0] == 0.0f);
    CHECK(x->grad().data()[1] == 1.0f);
    CHECK(x->grad().data()[2] == 0.0f);
    CHECK(x->grad().data()[3] == 1.0f);
}

TEST_CASE("sequential forward shape") {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Layer>>{
        std::make_shared<Linear>(4, 8),
        std::make_shared<ReLu>(),
        std::make_shared<Linear>(8, 2)
    });
    auto x = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, 1.0f);
    auto y = model->forward(x);
    CHECK(y->shape()[0] == 2);
    CHECK(y->shape()[1] == 2);
}

TEST_CASE("sequential parameters count") {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Layer>>{
        std::make_shared<Linear>(4, 8),
        std::make_shared<ReLu>(),
        std::make_shared<Linear>(8, 2)
    });
    // Linear(4,8) has weights+bias, Linear(8,2) has weights+bias = 4 total
    CHECK(model->parameters().size() == 4);
}
