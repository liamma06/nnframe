#pragma once
#include <cstddef>
#include <cstdint>

float quantize(const float* in, int8_t* out, size_t n);
void dequantize(const int8_t* in, float* out, size_t n, float scale);