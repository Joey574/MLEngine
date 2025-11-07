#include "SGD.hpp"

void SGD::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

void SGD::Build() {
    assert(defined && !built);
    built = true;
}

void SGD::Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasSize, elements, learningRate);
}

void SGD::Compute(float* __restrict parameters, float* __restrict derivatives, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);
    const float factor = learningRate / (float)elements;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        parameters[i] -= derivatives[i]*factor;
    }
}
