#pragma once
#include "Layer.hpp"


template <bool dropout, bool skipconn> 
void Layer::BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if (type == LayerType::input) { return; }

    ComputeDT(truth, n);

    if constexpr (dropout) {
        ApplyDropoutBP(n);
    }

    if constexpr (skipconn) {
        ComputeSkipDN(input, n);
    } else {
        ComputeDN(input, n);
    }
}
