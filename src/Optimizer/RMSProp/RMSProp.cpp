#include "RMSProp.hpp"

void RMSProp::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

void RMSProp::Build() {
    assert(defined && !built);
    built = true;
}

void RMSProp::Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightSquares, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasSquares, biasSize, elements, learningRate);
}

void RMSProp::Compute(float* __restrict parameters, float* __restrict derivatives, float* __restrict squares, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);
    
    const float factor = learningRate / (float)elements;
    const float decayRate = 1.0f-decay;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        squares[i] = (decay*squares[i])+(decayRate*squares[i]*squares[i]);
        parameters[i] -= (factor /(std::sqrt(squares[i]+epsilon)))*squares[i];
    }
}
