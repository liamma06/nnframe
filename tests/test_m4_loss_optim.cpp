#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "loss/mse.h"
#include "optim/sgd.h"
#include "optim/adamw.h"

TEST_CASE("mse forward - perfect prediction is zero loss") {
    auto pred = std::make_shared<Tensor>(std::vector<size_t>{4}, std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f});
    auto target = std::make_shared<Tensor>(std::vector<size_t>{4}, std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f});
    MSE loss;
    auto out = loss.forward(pred, target);
    CHECK(out->data()[0] == doctest::Approx(0.0f));
}

TEST_CASE("mse forward - known loss value") {
    // pred=[2,2] target=[0,0] -> mean((2^2 + 2^2)/2) = 4.0
    auto pred = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{2.0f, 2.0f});
    auto target = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{0.0f, 0.0f});
    MSE loss;
    auto out = loss.forward(pred, target);
    CHECK(out->data()[0] == doctest::Approx(4.0f));
}

TEST_CASE("mse backward - grad flows to predictions") {
    auto pred = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{3.0f, 1.0f});
    auto target = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 1.0f});
    pred->set_requires_grad(true);

    MSE loss;
    auto out = loss.forward(pred, target);
    out->backward();

    // MSE grad = 2*(pred-target)/n = 2*(2,0)/2 = (2,0)
    CHECK(pred->grad().data()[0] == doctest::Approx(2.0f));
    CHECK(pred->grad().data()[1] == doctest::Approx(0.0f));
}

TEST_CASE("sgd step updates weights") {
    auto w = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 1.0f});
    w->set_requires_grad(true);
    w->init_grad();
    w->grad().mutable_data()[0] = 2.0f;
    w->grad().mutable_data()[1] = 4.0f;

    SGD optim({w}, 0.1f);
    optim.step();

    // w = w - lr * grad = 1 - 0.1*2 = 0.8, 1 - 0.1*4 = 0.6
    CHECK(w->data()[0] == doctest::Approx(0.8f));
    CHECK(w->data()[1] == doctest::Approx(0.6f));
}

TEST_CASE("sgd zero_grad clears gradients") {
    auto w = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 1.0f});
    w->set_requires_grad(true);
    w->init_grad();
    w->grad().mutable_data()[0] = 5.0f;
    w->grad().mutable_data()[1] = 3.0f;

    SGD optim({w}, 0.1f);
    optim.zero_grad();

    CHECK(w->grad().data()[0] == doctest::Approx(0.0f));
    CHECK(w->grad().data()[1] == doctest::Approx(0.0f));
}

TEST_CASE("adamw step moves weight toward target") {
    // single weight starting at 1.0, gradient pushing it down
    auto w = std::make_shared<Tensor>(std::vector<size_t>{1}, std::vector<scalar_t>{1.0f});
    w->set_requires_grad(true);
    w->init_grad();
    w->grad().mutable_data()[0] = 1.0f;

    AdamW optim({w}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f); // no weight decay for clean test
    optim.step();

    // weight should have decreased from 1.0
    CHECK(w->data()[0] < 1.0f);
}

TEST_CASE("adamw step - weight decay shrinks weight") {
    // no gradient, only weight decay should shrink the weight
    auto w = std::make_shared<Tensor>(std::vector<size_t>{1}, std::vector<scalar_t>{1.0f});
    w->set_requires_grad(true);
    w->init_grad();
    w->grad().mutable_data()[0] = 0.0f; // zero grad, only decay

    AdamW optim({w}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);
    optim.step();

    // weight_decay pulls weight toward zero: w -= lr * decay * w
    CHECK(w->data()[0] < 1.0f);
}

TEST_CASE("adamw zero_grad clears gradients") {
    auto w = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 1.0f});
    w->set_requires_grad(true);
    w->init_grad();
    w->grad().mutable_data()[0] = 5.0f;
    w->grad().mutable_data()[1] = 3.0f;

    AdamW optim({w}, 0.01f);
    optim.zero_grad();

    CHECK(w->grad().data()[0] == doctest::Approx(0.0f));
    CHECK(w->grad().data()[1] == doctest::Approx(0.0f));
}
