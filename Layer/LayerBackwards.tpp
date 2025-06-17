#pragma once
#include "Layer.hpp"


template <bool dropout> 
void Layer::BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if (type == LayerType::input) { return; }

    ComputeDT(truth, n);

    if constexpr (dropout) {
        ApplyDropoutBP(n);
    }

    ComputeDN(input, n);
}
