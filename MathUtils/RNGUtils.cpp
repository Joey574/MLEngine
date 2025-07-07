#include "MathUtils.hpp"

uint32_t MathUtils::xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float MathUtils::fastRandFloat(uint32_t state) {
    return (xorshift32(state) / 4294967295.0f) * 2.0f - 1.0f;
}
