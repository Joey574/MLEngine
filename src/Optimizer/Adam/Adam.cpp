#include "Adam.hpp"

void Adam::Define(const YAML::Node& config) {
    assert(!(defined || built));

    iteration = 0;
    b1 = config[Y_OPT_B1].as<float>(Y_B1_DEFAULT);
    b2 = config[Y_OPT_B2].as<float>(Y_B2_DEFAULT);
    epsilon = config[Y_OPT_EPSL].as<float>(Y_EPSL_DEFAULT);
    defined = true;
}

void Adam::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = Tensor<float>(weightSize);
    biasVelocity = Tensor<float>(biasSize);
    weightSquares = Tensor<float>(weightSize);
    biasSquares = Tensor<float>(biasSize);
    built = true;
}

void Adam::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightVelocity, weightSquares, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasVelocity, biasSquares, biasSize, elements, learningRate);
    iteration++;
}

void Adam::Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);

    const float factor = learningRate / (float)elements;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1Denominator = (1.0f-std::pow(b1, (float)iteration));
    const float b2Denominator = (1.0f-std::pow(b2, (float)iteration));

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity.Data()[i] = (velocity.Data()[i]*b1)+b1Rate*derivatives.Data()[i];
        squares.Data()[i] = (squares.Data()[i]*b2)+b2Rate*derivatives.Data()[i]*derivatives.Data()[i];

        const float mh = velocity.Data()[i]/b1Denominator;
        const float vh = squares.Data()[i]/b2Denominator;

        parameters.Data()[i] -= factor*(mh/std::sqrt(vh+epsilon));
    }
}
