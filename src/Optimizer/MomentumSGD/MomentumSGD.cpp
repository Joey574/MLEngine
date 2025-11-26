#include "MomentumSGD.hpp"

void MomentumSGD::Define(const YAML::Node& config) {
    assert(!(defined || built));

    momentum = config[Y_OPT_MOMENTUM].as<float>(Y_MOMENTUM_DEFAULT);
    defined  = true;
}

void MomentumSGD::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = Tensor<float>(weightSize);
    biasVelocity   = Tensor<float>(biasSize);

    weightVelocity.Zero();
    biasVelocity.Zero();
    built = true;
}

void MomentumSGD::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightVelocity, elements, learningRate);
    Compute(biases, biasDerivatives, biasVelocity, elements, learningRate);
}

void MomentumSGD::Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, size_t elements, float learningRate) {
    assert(parameters.Size() == derivatives.Size());
    assert(defined && built);

    const float factor         = learningRate / (float)elements;
    const size_t numParameters = parameters.Size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity.Data()[i] = (velocity.Data()[i] * momentum) + (derivatives.Data()[i] * factor);
        parameters.Data()[i] -= velocity.Data()[i];
    }
}

int MomentumSGD::Save(std::ofstream& f) const {
    assert(defined && built);
    assert(!weightVelocity.IsEmpty() && !biasVelocity.IsEmpty());

    f.write((char*)weightVelocity.Data(), weightVelocity.Size() * sizeof(float));
    f.write((char*)biasVelocity.Data(), biasVelocity.Size() * sizeof(float));
    return 0;
}
int MomentumSGD::Load(std::ifstream& f) {
    assert(defined && built);

    f.read((char*)weightVelocity.Data(), weightVelocity.Size() * sizeof(float));
    f.read((char*)biasVelocity.Data(), biasVelocity.Size() * sizeof(float));
    built = true;
    return 0;
}
