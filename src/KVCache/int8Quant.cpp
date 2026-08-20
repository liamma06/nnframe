#include "infer/int8_quant.h"
#include <cmath>

float quantize(const float* in, int8_t* out, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(in[i]) > max_abs) {
            max_abs = std::abs(in[i]);
        }
    }
    
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        for (size_t i = 0; i < n; ++i) out[i] = 0;
        return 1.0f;
    }

    for (size_t i = 0; i < n; ++i) {
        float rounded = std::round(in[i] / scale);

        //clamps
        if (rounded > 127.0f) rounded = 127.0f;
        if (rounded < -127.0f) rounded = -127.0f;
        
        out[i] = static_cast<int8_t>(rounded);
    }

    return scale;
}

void dequantize(const int8_t* in, float* out, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(in[i]) * scale;
    }
}