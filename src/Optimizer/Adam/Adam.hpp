#pragma once
#include "../../MathUtils/MathUtils.hpp"

struct Adam {
    public:
    void Update(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate);

    void Define(YAML::Node& config);
    void Build(size_t weightSize, size_t biasSize);

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    size_t iteration;
    float b1;
    float b2;
    float epsilon;

    float* weightVelocity;
    float* biasVelocity;
    float* weightSquares;
    float* biasSquares;
    
    void Compute(float* __restrict parameters, float* __restrict derivatives, float* __restrict velocity, float* __restrict squares, size_t numParameters, size_t elements, float learningRate);
};
