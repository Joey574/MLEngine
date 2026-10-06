#include "RMSProp.hpp"

void RMSProp::Define(const YAML::Node& config) {
    assert(!(defined || built));

    decay   = config[Y_OPT_DECAY].as<float>(Y_DECAY_DEFAULT);
    epsilon = config[Y_OPT_EPSL].as<float>(Y_EPSL_DEFAULT);
    defined = true;
}

void RMSProp::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightSquares = Tensor<float>(weightSize);
    biasSquares   = Tensor<float>(biasSize);

    weightSquares.Zero();
    biasSquares.Zero();
    built = true;
}

void RMSProp::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightSquares, elements, learningRate);
    Compute(biases, biasDerivatives, biasSquares, elements, learningRate);
}

void RMSProp::Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& squares, size_t elements, float learningRate) {
    assert(parameters.Size() == derivatives.Size());
    assert(defined && built);

    const float factor    = learningRate / (float)elements;
    const float decayRate = 1.0f - decay;

    const size_t numParameters = parameters.Size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        squares.Data()[i] = (decay * squares.Data()[i]) + (decayRate * squares.Data()[i] * squares.Data()[i]);
        parameters.Data()[i] -= (factor / (std::sqrt(squares.Data()[i] + epsilon))) * squares.Data()[i];
    }
}

int RMSProp::Save(std::ofstream& f) const {
    assert(defined && built);
    assert(!weightSquares.IsEmpty() && !biasSquares.IsEmpty());

    f.write((char*)weightSquares.Data(), weightSquares.Size() * sizeof(float));
    f.write((char*)biasSquares.Data(), biasSquares.Size() * sizeof(float));
    return 0;
}
int RMSProp::Load(std::ifstream& f) {
    assert(defined && built);

    f.read((char*)weightSquares.Data(), weightSquares.Size() * sizeof(float));
    f.read((char*)biasSquares.Data(), biasSquares.Size() * sizeof(float));
    built = true;
    return 0;
}