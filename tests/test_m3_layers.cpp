#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"
#include "linear.h"

TEST_CASE("linear forward shape") {
    Linear lin(4, 8);
    auto x = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, 1.0f);
    auto y = lin.forward(x);
    CHECK(y->shape()[0] == 2);
    CHECK(y->shape()[1] == 8);
}
