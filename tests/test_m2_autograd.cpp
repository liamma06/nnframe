#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"

TEST_CASE("add backward - grad flows equally to both inputs") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 2.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{3.0f, 4.0f});
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->add(b);
    c->backward();

    // dc/da = 1, dc/db = 1, upstream = 1 so both grads are [1, 1]
    CHECK(a->grad().at({0}) == 1.0f);
    CHECK(a->grad().at({1}) == 1.0f);
    CHECK(b->grad().at({0}) == 1.0f);
    CHECK(b->grad().at({1}) == 1.0f);
}

TEST_CASE("sub backward - grad is +1 for self, -1 for other") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{5.0f, 6.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 2.0f});
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->sub(b);
    c->backward();

    CHECK(a->grad().at({0}) == 1.0f);
    CHECK(a->grad().at({1}) == 1.0f);
    CHECK(b->grad().at({0}) == -1.0f);
    CHECK(b->grad().at({1}) == -1.0f);
}

TEST_CASE("mul backward - grad is the other tensor's values") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{2.0f, 3.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{4.0f, 5.0f});
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->mul(b);
    c->backward();

    // dc/da = b, dc/db = a
    CHECK(a->grad().at({0}) == 4.0f);
    CHECK(a->grad().at({1}) == 5.0f);
    CHECK(b->grad().at({0}) == 2.0f);
    CHECK(b->grad().at({1}) == 3.0f);
}

TEST_CASE("matmul backward - correct gradients for A and B") {
    // A: [2x2], B: [2x2]
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{1.0f, 0.0f, 0.0f, 1.0f}); // identity
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->matmul(b);
    c->backward();

    // upstream is all 1s, B is identity so dL/dA = upstream @ B^T = all 1s @ I = all 1s summed per row
    // dL/dA[i,k] = sum_j upstream[i,j] * B[k,j] = sum_j 1 * I[k,j] = 1
    CHECK(a->grad().at({0, 0}) == 1.0f);
    CHECK(a->grad().at({0, 1}) == 1.0f);
    CHECK(a->grad().at({1, 0}) == 1.0f);
    CHECK(a->grad().at({1, 1}) == 1.0f);
}

TEST_CASE("chained ops backward - grad flows through multiple ops") {
    // c = a + b, d = c * c => dL/da = 2*c = 2*(a+b)
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{1.0f, 2.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{3.0f, 4.0f});
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->add(b);         // c = [4, 6]
    auto d = c->mul(c);         // d = [16, 36]
    d->backward();

    // dL/da = dL/dd * dd/dc * dc/da = 1 * 2*c * 1 = 2*c = [8, 12]
    CHECK(a->grad().at({0}) == 8.0f);
    CHECK(a->grad().at({1}) == 12.0f);
    CHECK(b->grad().at({0}) == 8.0f);
    CHECK(b->grad().at({1}) == 12.0f);
}
