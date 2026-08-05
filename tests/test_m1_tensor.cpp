#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"

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

TEST_CASE("reshape splits last dim into heads (rank-3)") {
    // [seq=2, embed_dim=4] -> [seq=2, num_heads=2, head_dim=2], same layout as multi-head split
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, std::vector<scalar_t>{1, 2, 3, 4, 10, 20, 30, 40});
    auto r = t->reshape({2, 2, 2});
    CHECK(r->shape()[0] == 2);
    CHECK(r->shape()[1] == 2);
    CHECK(r->shape()[2] == 2);
    // token 0, head 0 and head 1
    CHECK(r->at({0, 0, 0}) == 1.0f);
    CHECK(r->at({0, 0, 1}) == 2.0f);
    CHECK(r->at({0, 1, 0}) == 3.0f);
    CHECK(r->at({0, 1, 1}) == 4.0f);
    // token 1, head 0 and head 1
    CHECK(r->at({1, 0, 0}) == 10.0f);
    CHECK(r->at({1, 0, 1}) == 20.0f);
    CHECK(r->at({1, 1, 0}) == 30.0f);
    CHECK(r->at({1, 1, 1}) == 40.0f);
}

TEST_CASE("permute 2D swap matches manual transpose") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto p = t->permute({1, 0});
    CHECK(p->shape()[0] == 3);
    CHECK(p->shape()[1] == 2);
    // p[i,j] == t[j,i]
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 2; j++)
            CHECK(p->at({i, j}) == t->at({j, i}));
}

TEST_CASE("permute rank-3 puts head dim first, matches manual construction") {
    // [seq=2, num_heads=2, head_dim=2] -> [num_heads=2, seq=2, head_dim=2]
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{1, 2, 3, 4, 10, 20, 30, 40});
    auto p = t->permute({1, 0, 2});
    CHECK(p->shape()[0] == 2);
    CHECK(p->shape()[1] == 2);
    CHECK(p->shape()[2] == 2);
    // head 0, all tokens
    CHECK(p->at({0, 0, 0}) == 1.0f);
    CHECK(p->at({0, 0, 1}) == 2.0f);
    CHECK(p->at({0, 1, 0}) == 10.0f);
    CHECK(p->at({0, 1, 1}) == 20.0f);
    // head 1, all tokens
    CHECK(p->at({1, 0, 0}) == 3.0f);
    CHECK(p->at({1, 0, 1}) == 4.0f);
    CHECK(p->at({1, 1, 0}) == 30.0f);
    CHECK(p->at({1, 1, 1}) == 40.0f);
}

TEST_CASE("permute 3-way rotation matches formula out_idx[j] = self_idx[axes[j]]") {
    // not a simple 2-axis swap - this is the case the naive "scatter" bug got wrong
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6, 7, 8});
    auto p = t->permute({2, 0, 1});
    CHECK(p->shape()[0] == 2);
    CHECK(p->shape()[1] == 2);
    CHECK(p->shape()[2] == 2);
    // p->at({q,r,s}) should equal t->at({r,s,q})  (self_idx[axes[j]] = out_idx[j] rearranged)
    for (size_t q = 0; q < 2; q++)
        for (size_t r = 0; r < 2; r++)
            for (size_t s = 0; s < 2; s++)
                CHECK(p->at({q, r, s}) == t->at({r, s, q}));
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

TEST_CASE("matmul rank-3 loops leading (head) dimension independently") {
    // L=2, M=2, K=3, N=2 - deliberately non-square so a dimension mixup would be caught
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 3}, std::vector<scalar_t>{
        1, 2, 3,  4, 5, 6,      // l=0
        7, 8, 9,  10, 11, 12    // l=1
    });
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 2}, std::vector<scalar_t>{
        1, 0,  0, 1,  1, 1,     // l=0
        2, 0,  0, 2,  1, 1      // l=1
    });
    auto c = a->matmul(b);

    CHECK(c->shape()[0] == 2);
    CHECK(c->shape()[1] == 2);
    CHECK(c->shape()[2] == 2);

    // l=0 slice: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]]
    CHECK(c->at({0, 0, 0}) == 4.0f);
    CHECK(c->at({0, 0, 1}) == 5.0f);
    CHECK(c->at({0, 1, 0}) == 10.0f);
    CHECK(c->at({0, 1, 1}) == 11.0f);

    // l=1 slice: [[7,8,9],[10,11,12]] @ [[2,0],[0,2],[1,1]]
    CHECK(c->at({1, 0, 0}) == 23.0f);
    CHECK(c->at({1, 0, 1}) == 25.0f);
    CHECK(c->at({1, 1, 0}) == 32.0f);
    CHECK(c->at({1, 1, 1}) == 34.0f);
}

TEST_CASE("softmax rank-3 normalizes each (head, query) row independently") {
    // shape [num_heads=2, seq=2, seq=2]
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{
        1, 3,  2, 2,    // head0: row0=[1,3], row1=[2,2]
        0, 0,  5, 1     // head1: row0=[0,0], row1=[5,1]
    });
    auto s = t->softmax(2);

    CHECK(s->shape()[0] == 2);
    CHECK(s->shape()[1] == 2);
    CHECK(s->shape()[2] == 2);

    // head0, row0: [1,3] -> subtract max 3 -> exp[-2,0] -> normalize
    CHECK(s->at({0, 0, 0}) == doctest::Approx(0.11920292f).epsilon(0.0001));
    CHECK(s->at({0, 0, 1}) == doctest::Approx(0.88079708f).epsilon(0.0001));
    // head0, row1: [2,2] -> equal -> [0.5,0.5]
    CHECK(s->at({0, 1, 0}) == doctest::Approx(0.5f));
    CHECK(s->at({0, 1, 1}) == doctest::Approx(0.5f));
    // head1, row0: [0,0] -> equal -> [0.5,0.5]
    CHECK(s->at({1, 0, 0}) == doctest::Approx(0.5f));
    CHECK(s->at({1, 0, 1}) == doctest::Approx(0.5f));
    // head1, row1: [5,1] -> subtract max 5 -> exp[0,-4] -> normalize
    CHECK(s->at({1, 1, 0}) == doctest::Approx(0.98201379f).epsilon(0.0001));
    CHECK(s->at({1, 1, 1}) == doctest::Approx(0.01798621f).epsilon(0.0001));

    // every row sums to 1, regardless of head or query
    for (size_t h = 0; h < 2; h++)
        for (size_t q = 0; q < 2; q++)
            CHECK(s->at({h, q, 0}) + s->at({h, q, 1}) == doctest::Approx(1.0f));
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
