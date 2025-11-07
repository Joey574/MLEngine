#include "SGD.hpp"

/// @brief Defines internal sgd information based on config (literally nothing)
/// @param config Optimizer config information
void SGD::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

/// @brief Builds internal sgd informaiton based on config (literally nothing)
/// @param weightSize Number of weights in the layer
/// @param biasSize Number of biases in the layer
void SGD::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);
    built = true;
}

/// @brief 
/// @param weights 
/// @param biases 
/// @param weightDerivatives 
/// @param biasDerivatives 
/// @param weightSize 
/// @param biasSize 
/// @param elements 
/// @param learningRate 
void SGD::Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasSize, elements, learningRate);
}

/// @brief Internal function that applies optimization rule
/// @param parameters Pointer to parameters to update
/// @param derivatives Pointer to parameter derivatives
/// @param numParameters Number of parameters
/// @param elements Number of training samples used
/// @param learningRate Learning rate value to use
void SGD::Compute(float* __restrict parameters, const float* __restrict derivatives, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);
    const float factor = learningRate / (float)elements;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        parameters[i] -= derivatives[i]*factor;
    }
}
