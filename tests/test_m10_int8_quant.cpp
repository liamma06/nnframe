#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "infer/int8_quant.h"
#include <cmath>
#include <vector>

TEST_CASE("quantize -> dequantize round trip is close but not exact") {
    // int8 with a per-block scale has ~1/127 relative error near the block's max value --
    // check it's close, not bit-exact.
    std::vector<float> values = {1.0f, 3.14159265f, -2.71828f, 0.5f, -100.0f, 42.0f};
    std::vector<int8_t> packed(values.size());
    std::vector<float> back(values.size());

    float scale = quantize(values.data(), packed.data(), values.size());
    dequantize(packed.data(), back.data(), values.size(), scale);

    for (size_t i = 0; i < values.size(); i++) {
        CHECK(std::fabs(back[i] - values[i]) <= scale + 1e-6f);
    }
}

TEST_CASE("quantize handles all-zero input without dividing by zero") {
    std::vector<float> values = {0.0f, 0.0f, 0.0f};
    std::vector<int8_t> packed(values.size());
    std::vector<float> back(values.size());

    float scale = quantize(values.data(), packed.data(), values.size());
    dequantize(packed.data(), back.data(), values.size(), scale);

    for (size_t i = 0; i < values.size(); i++) {
        CHECK(back[i] == 0.0f);
    }
}

TEST_CASE("quantize maps the max-magnitude value to the full int8 range") {
    std::vector<float> values = {10.0f, -10.0f, 5.0f};
    std::vector<int8_t> packed(values.size());

    quantize(values.data(), packed.data(), values.size());

    // the largest-magnitude input should land at or near +/-127
    CHECK(std::abs(static_cast<int>(packed[1])) >= 126);
}
