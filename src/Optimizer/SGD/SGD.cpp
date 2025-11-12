#include "SGD.hpp"

/// @brief Defines internal sgd information based on config (literally nothing)
/// @param config Optimizer config information
void SGD::Define(const YAML::Node& config) {
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
/// @param elements 
/// @param learningRate 
void SGD::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, elements, learningRate);
    Compute(biases, biasDerivatives, elements, learningRate);
}

/// @brief Internal function that applies optimization rule
/// @param parameters Pointer to parameters to update
/// @param derivatives Pointer to parameter derivatives
/// @param elements Number of training samples used
/// @param learningRate Learning rate value to use
void SGD::Compute(Tensor<float>& parameters, const Tensor<float>& derivatives, size_t elements, float learningRate) {
    assert(parameters.Size() == derivatives.Size());
    assert(defined && built);

    const float factor = learningRate / (float)elements;
    const size_t numParameters = parameters.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        parameters.Data()[i] -= derivatives.Data()[i]*factor;
    }
}
