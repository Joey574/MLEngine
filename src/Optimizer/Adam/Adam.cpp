#include "Adam.hpp"

void Adam::Define(const YAML::Node& config) {
    assert(!(defined || built));

    iteration = 1;
    b1        = config[Y_OPT_B1].as<float>(Y_B1_DEFAULT);
    b2        = config[Y_OPT_B2].as<float>(Y_B2_DEFAULT);
    epsilon   = config[Y_OPT_EPSL].as<float>(Y_EPSL_DEFAULT);
    defined   = true;
}

void Adam::Build(size_t weightSize, size_t biasSize) {
    assert(defined && !built);

    weightVelocity = Tensor<float>(weightSize);
    biasVelocity   = Tensor<float>(biasSize);
    weightSquares  = Tensor<float>(weightSize);
    biasSquares    = Tensor<float>(biasSize);

    weightVelocity.Zero();
    biasVelocity.Zero();
    weightSquares.Zero();
    biasSquares.Zero();
    built = true;
}

void Adam::Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t elements, float learningRate) {
    assert(defined && built);
    assert(iteration > 0);
    assert(epsilon > 0.0f);

    if (!weights.IsEmpty())
        Compute(weights, weightDerivatives, weightVelocity, weightSquares, elements, learningRate);
    if (!biases.IsEmpty())
        Compute(biases, biasDerivatives, biasVelocity, biasSquares, elements, learningRate);
    iteration++;
}

void Adam::Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t elements, float learningRate) {
    assert(parameters.Data() != nullptr && derivatives.Data() != nullptr);
    assert(!parameters.HasNan() && !derivatives.HasNan());
    assert(parameters.Size() == derivatives.Size());
    assert(defined && built);
    assert(iteration > 0);
    assert(epsilon > 0.0f);

    // std::cout << "P: " << parameters.Mean() << "\nD: " << derivatives.Mean() << "\nV: " << velocity.Mean() << "\nS: " << squares.Mean() << "\nE: " << elements << "\nLr: " <<
    // learningRate << "\n\n";

    float* __restrict pData = parameters.Data();
    float* __restrict dData = derivatives.Data();
    float* __restrict vData = velocity.Data();
    float* __restrict sData = squares.Data();

    const float factor = learningRate / (float)elements;
    const float b1Rate = 1.0f - b1;
    const float b2Rate = 1.0f - b2;

    const float b1InvDenominator = 1.0f / (1.0f - powf(b1, (float)iteration));
    const float b2InvDenominator = 1.0f / (1.0f - powf(b2, (float)iteration));

    const size_t numParameters = parameters.Size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        vData[i] = (vData[i] * b1) + (b1Rate * dData[i]);
        sData[i] = (sData[i] * b2) + (b2Rate * dData[i] * dData[i]);

        const float mh = vData[i] * b1InvDenominator;
        const float vh = sData[i] * b2InvDenominator;

        pData[i] -= factor * (mh / sqrtf(vh + epsilon));
    }

    assert(!parameters.HasNan());
}

int Adam::Save(std::ofstream& f) const {
    assert(defined && built);
    assert(!weightVelocity.IsEmpty() && !biasVelocity.IsEmpty());
    assert(!weightSquares.IsEmpty() && !biasSquares.IsEmpty());

    if (!weightVelocity.IsEmpty())
        f.write((char*)weightVelocity.Data(), weightVelocity.Size() * sizeof(float));
    if (!biasVelocity.IsEmpty())
        f.write((char*)biasVelocity.Data(), biasVelocity.Size() * sizeof(float));
    if (!weightSquares.IsEmpty())
        f.write((char*)weightSquares.Data(), weightSquares.Size() * sizeof(float));
    if (!biasSquares.IsEmpty())
        f.write((char*)biasSquares.Data(), biasSquares.Size() * sizeof(float));
    f.write((char*)&iteration, sizeof(size_t));
    return 0;
}
int Adam::Load(std::ifstream& f) {
    assert(defined && built);

    if (!weightVelocity.IsEmpty())
        f.read((char*)weightVelocity.Data(), weightVelocity.Size() * sizeof(float));
    if (!biasVelocity.IsEmpty())
        f.read((char*)biasVelocity.Data(), biasVelocity.Size() * sizeof(float));
    if (!weightSquares.IsEmpty())
        f.read((char*)weightSquares.Data(), weightSquares.Size() * sizeof(float));
    if (!biasSquares.IsEmpty())
        f.read((char*)biasSquares.Data(), biasSquares.Size() * sizeof(float));
    f.read((char*)&iteration, sizeof(size_t));
    return 0;
}
