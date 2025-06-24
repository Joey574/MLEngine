#pragma once
#include "Layer.hpp"

template <Layer::LayerType ltype, bool dropout, bool skipconn> 
void Layer::BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if constexpr (ltype == LayerType::input) { return; }

    if constexpr (ltype == LayerType::output) {
        ComputeDTOutput(truth, n);
    } else {
        ComputeDT(truth, n);
    }

    if constexpr (dropout) {
        ApplyDropoutBP(n);
    }

    if constexpr (skipconn) {
        ComputeSkipDN(input, n);
    } else {
        ComputeDN(input, n);
    }
}
