#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/bf16.h"
#include <cmath>
#include <cstring>

TEST_CASE("float_to_bf16 -> bf16_to_float round trip is close but not exact") {
    // bf16 keeps 7 mantissa bits vs fp32's 23, so round trip loses precision --
    // check it's close (within bf16's ~2^-8 relative error), not bit-exact.
    float values[] = {1.0f, 3.14159265f, -2.71828f, 0.0001f, 1234.5678f, -0.5f};

    for (float v : values) {
        uint16_t packed = float_to_bf16(v);
        float back = bf16_to_float(packed);

        float rel_error = std::fabs(back - v) / std::max(std::fabs(v), 1e-12f);
        CHECK(rel_error < 0.01f); // bf16's worst-case relative error is ~1/256 (~0.4%)
    }
}

TEST_CASE("float_to_bf16 is exact for values with no mantissa bits below bit 16") {
    // values that are already "bf16-clean" (zero in the bottom 16 mantissa bits)
    // should round-trip exactly, since nothing gets rounded away.
    float v = 1.0f; // exponent-only, mantissa all zero -- exactly representable in bf16
    uint16_t packed = float_to_bf16(v);
    float back = bf16_to_float(packed);
    CHECK(back == v);

    float v2 = 2.0f;
    CHECK(bf16_to_float(float_to_bf16(v2)) == v2);

    float v3 = 0.0f;
    CHECK(bf16_to_float(float_to_bf16(v3)) == v3);
}

TEST_CASE("float_to_bf16 rounds ties to even") {
    // construct a float whose bottom 16 bits are exactly 0x8000 (an exact tie),
    // once with the kept bit currently even, once currently odd, and check both
    // land on the even neighbor instead of always rounding the same direction.

    // even case: top 16 bits end in 0 (...0), bottom 16 bits = 0x8000 exactly
    uint32_t even_bits = 0x3F800000u | 0x00008000u; // 1.0's exponent/sign, mantissa top bit set, tie in the dropped half
    float even_val;
    std::memcpy(&even_val, &even_bits, sizeof(even_val));
    uint16_t even_result = float_to_bf16(even_val);
    CHECK((even_result & 1) == 0); // should land even either way

    // odd case: flip the kept bit to 1 before the tie
    uint32_t odd_bits = even_bits | 0x00010000u; // set bit16 -> kept LSB starts odd
    float odd_val;
    std::memcpy(&odd_val, &odd_bits, sizeof(odd_val));
    uint16_t odd_result = float_to_bf16(odd_val);
    CHECK((odd_result & 1) == 0); // should round up to the even neighbor
}
