#include "Adam.hpp"

void Adam::Define(YAML::Node& config) {
    assert(!(defined || built));

    iteration = 0;
    b1 = config[Y_OPT_B1].as<float>(Y_B1_DEFAULT);
    b2 = config[Y_OPT_B2].as<float>(Y_B2_DEFAULT);
    epsilon = config[Y_OPT_EPSL].as<float>(Y_EPSL_DEFAULT);
    defined = true;
}

void Adam::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = (float*)MathUtils::Allocate(weightSize*sizeof(float));
    biasVelocity = (float*)MathUtils::Allocate(biasSize*sizeof(float));
    weightSquares = (float*)MathUtils::Allocate(weightSize*sizeof(float));
    biasSquares = (float*)MathUtils::Allocate(biasSize*sizeof(float));
    built = true;
}

void Adam::Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate) {
    assert(defined && built);

    Compute(weights, weightDerivatives, weightVelocity, weightSquares, weightSize, elements, learningRate);
    Compute(biases, biasDerivatives, biasVelocity, biasSquares, biasSize, elements, learningRate);
    iteration++;
}

void Adam::Compute(float* __restrict parameters, float* __restrict derivatives, float* __restrict velocity, float* __restrict squares, size_t numParameters, size_t elements, float learningRate) {
    assert(defined && built);

    const float factor = learningRate / (float)elements;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1Denominator = (1.0f-std::pow(b1, (float)iteration));
    const float b2Denominator = (1.0f-std::pow(b2, (float)iteration));

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity[i] = (velocity[i]*b1)+b1Rate*derivatives[i];
        squares[i] = (squares[i]*b2)+b2Rate*derivatives[i]*derivatives[i];

        const float mh = velocity[i]/b1Denominator;
        const float vh = squares[i]/b2Denominator;

        parameters[i] -= factor*(mh/std::sqrt(vh+epsilon));
    }
}
