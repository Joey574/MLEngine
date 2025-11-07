#include "MomentumSGD.hpp"

void MomentumSGD::Define(YAML::Node& config) {
    assert(!(defined || built));

    momentum = config[Y_OPT_MOMENTUM].as<float>(Y_MOMENTUM_DEFAULT);
    defined = true;
}

void MomentumSGD::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = (float*)MathUtils::Allocate(weightSize*sizeof(float));
    biasVelocity = (float*)MathUtils::Allocate(biasSize*sizeof(float));
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
