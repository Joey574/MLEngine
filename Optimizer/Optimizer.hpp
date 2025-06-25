#pragma once

struct Optimizer {
public:
    using UpdateFunc = void (*)(float* __restrict, float* __restrict, const float* __restrict, const float* __restrict, float, size_t n);

    UpdateFunc update;

private:

    void SGD();
    void MomentumSGD();
    void RMSProp();
    void Adam();
};
