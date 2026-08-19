#pragma once
#include <cstdint>
#include <cstring>

inline uint16_t float_to_bf16(float value){
    /*
        gets raw bits -> memcpy 
        take top16 bits

        I still don't fully understnad this bit stuff 
    */
    uint32_t raw_bits_32;
    std::memcpy(&raw_bits_32, &value, sizeof(value));

    uint32_t lsb = (raw_bits_32 >> 16) & 1; //shift right 16 bits and peak
    uint32_t rounding_bias = 0x7fff + lsb;   
    uint16_t result = static_cast<uint16_t>((raw_bits_32 + rounding_bias) >> 16);

    return result;
}

inline float bf16_to_float(uint16_t bits){
    // reverse  
    uint32_t raw_bits_32 = static_cast<uint32_t>(bits) << 16;

    float value;
    std::memcpy(&value, &raw_bits_32, sizeof(value));
    return value;
}
