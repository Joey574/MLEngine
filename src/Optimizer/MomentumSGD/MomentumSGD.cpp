#include "MomentumSGD.hpp"

void MomentumSGD::Define(YAML::Node& config) {
    assert(!(defined || built));

    momentum = config[Y_OPT_MOMENTUM].as<float>(Y_MOMENTUM_DEFAULT);
    defined = true;
}

void MomentumSGD::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = Tensor<float>(weightSize);
    biasVelocity = Tensor<float>(biasSize);
    built = true;
}

void MomentumSGD::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightVelocity, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasVelocity, biasSize, elements, learningRate);
}

void MomentumSGD::Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);
    const float factor = learningRate / (float)elements;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity.Data()[i] = (velocity.Data()[i]*momentum)+(derivatives.Data()[i]*factor);
        parameters.Data()[i] -= velocity.Data()[i];
    }
}
