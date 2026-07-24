#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"

TEST_CASE("construct from fill value") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, 1.0f);
    CHECK(t->rank() == 2);
    CHECK(t->numel() == 6);
    CHECK(t->shape()[0] == 2);
    CHECK(t->shape()[1] == 3);
    CHECK(t->at({0, 0}) == 1.0f);
    CHECK(t->at({1, 2}) == 1.0f);
}

TEST_CASE("construct from data vector") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    CHECK(t->at({0, 0}) == 1.0f);
    CHECK(t->at({0, 2}) == 3.0f);
    CHECK(t->at({1, 0}) == 4.0f);
    CHECK(t->at({1, 2}) == 6.0f);
}

TEST_CASE("row-major strides") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 4}, 0.0f);
    CHECK(t->strides()[0] == 12);
    CHECK(t->strides()[1] == 4);
    CHECK(t->strides()[2] == 1);
}

TEST_CASE("at write and read back") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, 0.0f);
    t->at({1, 2}) = 99.0f;
    CHECK(t->at({1, 2}) == 99.0f);
}

TEST_CASE("reshape returns view with same buffer") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto r = t->reshape({3, 2});
    CHECK(r->shape()[0] == 3);
    CHECK(r->shape()[1] == 2);
    CHECK(r->strides()[0] == 2);
    CHECK(r->strides()[1] == 1);
    // same buffer: mutate via reshape, read via original
    r->at({0, 0}) = 99.0f;
    CHECK(t->at({0, 0}) == 99.0f);
}

TEST_CASE("element-wise add same shape") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{1, 2, 3, 4});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{5, 6, 7, 8});
    auto c = a->add(b);
    CHECK(c->at({0, 0}) == 6.0f);
    CHECK(c->at({0, 1}) == 8.0f);
    CHECK(c->at({1, 0}) == 10.0f);
    CHECK(c->at({1, 1}) == 12.0f);
}

TEST_CASE("element-wise sub and mul") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{10, 20});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2}, std::vector<scalar_t>{3, 4});
    auto s = a->sub(b);
    auto m = a->mul(b);
    CHECK(s->at({0}) == 7.0f);
    CHECK(s->at({1}) == 16.0f);
    CHECK(m->at({0}) == 30.0f);
    CHECK(m->at({1}) == 80.0f);
}

TEST_CASE("broadcasting add {1,4} + {3,4}") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{1, 4}, std::vector<scalar_t>{1, 2, 3, 4});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{3, 4}, std::vector<scalar_t>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});
    auto c = a->add(b);
    CHECK(c->shape()[0] == 3);
    CHECK(c->shape()[1] == 4);
    CHECK(c->at({0, 0}) == 2.0f);
    CHECK(c->at({1, 0}) == 3.0f);
    CHECK(c->at({2, 3}) == 7.0f);
}

TEST_CASE("matmul 2x3 * 3x2") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, std::vector<scalar_t>{7, 8, 9, 10, 11, 12});
    auto c = a->matmul(b);
    CHECK(c->shape()[0] == 2);
    CHECK(c->shape()[1] == 2);
    CHECK(c->at({0, 0}) == 58.0f);
    CHECK(c->at({0, 1}) == 64.0f);
    CHECK(c->at({1, 0}) == 139.0f);
    CHECK(c->at({1, 1}) == 154.0f);
}

TEST_CASE("matmul on transposed tensor") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{7, 8, 9, 10, 11, 12});
    auto bt = b->transpose();
    auto c = a->matmul(bt);
    CHECK(c->shape()[0] == 2);
    CHECK(c->shape()[1] == 2);
    CHECK(c->at({0, 0}) == 50.0f);
    CHECK(c->at({0, 1}) == 68.0f);
    CHECK(c->at({1, 0}) == 122.0f);
    CHECK(c->at({1, 1}) == 167.0f);
}

TEST_CASE("allclose") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{3}, std::vector<scalar_t>{1.0f, 2.0f, 3.0f});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{3}, std::vector<scalar_t>{1.0f, 2.0000001f, 3.0f});
    CHECK(a->allclose(*b));
    auto c = std::make_shared<Tensor>(std::vector<size_t>{3}, std::vector<scalar_t>{1.0f, 2.1f, 3.0f});
    CHECK(!a->allclose(*c));
}
