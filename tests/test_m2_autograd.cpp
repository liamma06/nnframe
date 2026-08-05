#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"

TEST_CASE("softmax rank-3 backward - gradient computed independently per (head, query) row") {
    // t's rows all have equal pairs, so softmax(t) is exactly [0.5, 0.5] everywhere -
    // that pins down the softmax output precisely, so any error shows up purely from
    // the backward indexing (h,q,k), not from imprecise exp() arithmetic.
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{
        1, 1,  2, 2,      // head0
        3, 3,  4, 4       // head1
    });
    t->set_requires_grad(true);

    auto s = t->softmax(2);
    // deliberately distinct per-row magnitudes (10s, 30s, 200s, single-digit) so an
    // (h,q) index mixup in the backward loop would produce a clearly wrong value
    auto scale = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{
        10, 20,   30, 60,
        100, 300,  7, 9
    });
    auto c = s->mul(scale);
    c->backward();

    // s=[0.5,0.5] everywhere => dt[h,q,k] = 0.5 * (scale[h,q,k] - avg(scale[h,q,:]))
    CHECK(t->grad().at({0, 0, 0}) == doctest::Approx(-2.5f));
    CHECK(t->grad().at({0, 0, 1}) == doctest::Approx(2.5f));
    CHECK(t->grad().at({0, 1, 0}) == doctest::Approx(-7.5f));
    CHECK(t->grad().at({0, 1, 1}) == doctest::Approx(7.5f));
    CHECK(t->grad().at({1, 0, 0}) == doctest::Approx(-50.0f));
    CHECK(t->grad().at({1, 0, 1}) == doctest::Approx(50.0f));
    CHECK(t->grad().at({1, 1, 0}) == doctest::Approx(-0.5f));
    CHECK(t->grad().at({1, 1, 1}) == doctest::Approx(0.5f));
}

TEST_CASE("matmul rank-3 backward - gradients computed independently per head slice") {
    // same L=2,M=2,K=3,N=2 (non-square) shapes as the rank-3 forward test
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 3}, std::vector<scalar_t>{
        1, 2, 3,  4, 5, 6,
        7, 8, 9,  10, 11, 12
    });
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 2}, std::vector<scalar_t>{
        1, 0,  0, 1,  1, 1,
        2, 0,  0, 2,  1, 1
    });
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    auto c = a->matmul(b);
    c->backward(); // upstream is all-ones, shape [2,2,2]

    // da[l,m,k] = sum_n upstream[l,m,n] * b[l,k,n] = sum_n b[l,k,n] (upstream all ones)
    // l=0: b rows are [1,0]->1, [0,1]->1, [1,1]->2 - same for both m rows
    CHECK(a->grad().at({0, 0, 0}) == 1.0f);
    CHECK(a->grad().at({0, 0, 1}) == 1.0f);
    CHECK(a->grad().at({0, 0, 2}) == 2.0f);
    CHECK(a->grad().at({0, 1, 0}) == 1.0f);
    CHECK(a->grad().at({0, 1, 1}) == 1.0f);
    CHECK(a->grad().at({0, 1, 2}) == 2.0f);
    // l=1: b rows are [2,0]->2, [0,2]->2, [1,1]->2
    CHECK(a->grad().at({1, 0, 0}) == 2.0f);
    CHECK(a->grad().at({1, 0, 1}) == 2.0f);
    CHECK(a->grad().at({1, 0, 2}) == 2.0f);
    CHECK(a->grad().at({1, 1, 0}) == 2.0f);
    CHECK(a->grad().at({1, 1, 1}) == 2.0f);
    CHECK(a->grad().at({1, 1, 2}) == 2.0f);

    // db[l,k,n] = sum_m a[l,m,k] * upstream[l,m,n] = sum_m a[l,m,k] (upstream all ones)
    // l=0: a columns are k=0:[1,4]->5, k=1:[2,5]->7, k=2:[3,6]->9 - same for both n cols
    CHECK(b->grad().at({0, 0, 0}) == 5.0f);
    CHECK(b->grad().at({0, 0, 1}) == 5.0f);
    CHECK(b->grad().at({0, 1, 0}) == 7.0f);
    CHECK(b->grad().at({0, 1, 1}) == 7.0f);
    CHECK(b->grad().at({0, 2, 0}) == 9.0f);
    CHECK(b->grad().at({0, 2, 1}) == 9.0f);
    // l=1: a columns are k=0:[7,10]->17, k=1:[8,11]->19, k=2:[9,12]->21
    CHECK(b->grad().at({1, 0, 0}) == 17.0f);
    CHECK(b->grad().at({1, 0, 1}) == 17.0f);
    CHECK(b->grad().at({1, 1, 0}) == 19.0f);
    CHECK(b->grad().at({1, 1, 1}) == 19.0f);
    CHECK(b->grad().at({1, 2, 0}) == 21.0f);
    CHECK(b->grad().at({1, 2, 1}) == 21.0f);
}

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

TEST_CASE("reshape backward - requires_grad false produces no tracking") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{4}, std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f});
    auto b = a->reshape({2, 2});
    CHECK(b->requires_grad() == false);
}

TEST_CASE("reshape backward - grad flows back in flat order, unchanged by shape") {
    // a: [6], reshape -> [2,3], multiply by distinct per-element scale, then backward.
    // reshape never reorders elements, so da[flat i] should equal scale[flat i] exactly.
    auto a = std::make_shared<Tensor>(std::vector<size_t>{6}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    a->set_requires_grad(true);

    auto b = a->reshape({2, 3});
    auto scale = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{10, 20, 30, 40, 50, 60});
    auto c = b->mul(scale);
    c->backward();

    CHECK(a->grad().at({0}) == 10.0f);
    CHECK(a->grad().at({1}) == 20.0f);
    CHECK(a->grad().at({2}) == 30.0f);
    CHECK(a->grad().at({3}) == 40.0f);
    CHECK(a->grad().at({4}) == 50.0f);
    CHECK(a->grad().at({5}) == 60.0f);
}

TEST_CASE("reshape backward - rank-3 head-split shape still routes gradient correctly") {
    // [seq=2, embed_dim=4] -> [seq=2, num_heads=2, head_dim=2], the exact shape used for multi-head attention
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, std::vector<scalar_t>{1, 2, 3, 4, 10, 20, 30, 40});
    a->set_requires_grad(true);

    auto b = a->reshape({2, 2, 2});
    auto scale = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{1, 1, 1, 1, 2, 2, 2, 2});
    auto c = b->mul(scale);
    c->backward();

    // token 0 rows scaled by 1, token 1 rows scaled by 2 - grad should mirror that in the original [2,4] shape
    CHECK(a->grad().at({0, 0}) == 1.0f);
    CHECK(a->grad().at({0, 1}) == 1.0f);
    CHECK(a->grad().at({0, 2}) == 1.0f);
    CHECK(a->grad().at({0, 3}) == 1.0f);
    CHECK(a->grad().at({1, 0}) == 2.0f);
    CHECK(a->grad().at({1, 1}) == 2.0f);
    CHECK(a->grad().at({1, 2}) == 2.0f);
    CHECK(a->grad().at({1, 3}) == 2.0f);
}

TEST_CASE("permute backward - requires_grad false produces no tracking") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto p = a->permute({1, 0});
    CHECK(p->requires_grad() == false);
}

TEST_CASE("permute backward - 2D swap routes gradient to transposed position") {
    // a: [2,3], p = permute({1,0}) -> [3,2], p[i,j] = a[j,i]
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    a->set_requires_grad(true);

    auto p = a->permute({1, 0});
    auto scale = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, std::vector<scalar_t>{10, 20, 30, 40, 50, 60});
    auto c = p->mul(scale);
    c->backward();

    // c[i,j] = a[j,i] * scale[i,j]  =>  da[j,i] = scale[i,j]
    CHECK(a->grad().at({0, 0}) == 10.0f); // scale[0,0]
    CHECK(a->grad().at({1, 0}) == 20.0f); // scale[0,1]
    CHECK(a->grad().at({0, 1}) == 30.0f); // scale[1,0]
    CHECK(a->grad().at({1, 1}) == 40.0f); // scale[1,1]
    CHECK(a->grad().at({0, 2}) == 50.0f); // scale[2,0]
    CHECK(a->grad().at({1, 2}) == 60.0f); // scale[2,1]
}

TEST_CASE("permute backward - 3-way rotation (regression test for scatter-vs-gather bug)") {
    // axes={2,0,1} is NOT its own inverse, unlike a plain 2-axis swap - this is the
    // case where writing out_idx[axes[j]] = self_idx[j] (scatter) gives the WRONG
    // answer, while out_idx[j] = self_idx[axes[j]] (gather, matching permute's
    // forward definition) gives the right one. This test fails under the buggy version.
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6, 7, 8});
    a->set_requires_grad(true);

    auto p = a->permute({2, 0, 1}); // p[q,r,s] = a[r,s,q]
    auto scale = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 2}, std::vector<scalar_t>{100, 200, 300, 400, 500, 600, 700, 800});
    auto c = p->mul(scale);
    c->backward();

    // c[q,r,s] = a[r,s,q] * scale[q,r,s]  =>  da[r,s,q] = scale[q,r,s], i.e. da[i,j,k] = scale[k,i,j]
    CHECK(a->grad().at({0, 0, 0}) == 100.0f); // scale[0,0,0]
    CHECK(a->grad().at({0, 1, 0}) == 200.0f); // scale[0,0,1]
    CHECK(a->grad().at({1, 0, 0}) == 300.0f); // scale[0,1,0]
    CHECK(a->grad().at({1, 1, 0}) == 400.0f); // scale[0,1,1]
    CHECK(a->grad().at({0, 0, 1}) == 500.0f); // scale[1,0,0]
    CHECK(a->grad().at({0, 1, 1}) == 600.0f); // scale[1,0,1]
    CHECK(a->grad().at({1, 0, 1}) == 700.0f); // scale[1,1,0]
    CHECK(a->grad().at({1, 1, 1}) == 800.0f); // scale[1,1,1]
}
