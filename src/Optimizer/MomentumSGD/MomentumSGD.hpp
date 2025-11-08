#pragma once

struct MomentumSGD {
    public:
    void Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate);

    void Define(YAML::Node& config);
    void Build(size_t weightSize, size_t biasSize);

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    float momentum;
    Tensor<float> weightVelocity;
    Tensor<float> biasVelocity;

    void Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, size_t numParameters, size_t elements, float learningRate);
};
