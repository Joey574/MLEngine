#include "MomentumSGD.hpp"

void MomentumSGD::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

void MomentumSGD::Build() {
    assert(defined && !built);
    built = true;
}

void MomentumSGD::Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightVelocity, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasVelocity, biasSize, elements, learningRate);
}

void MomentumSGD::Compute(float* __restrict parameters, float* __restrict derivatives, float* __restrict velocity, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);
    const float factor = learningRate / (float)elements;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity[i] = (velocity[i]*momentum)+(derivatives[i]*factor);
        parameters[i] -= velocity[i];
    }
}
