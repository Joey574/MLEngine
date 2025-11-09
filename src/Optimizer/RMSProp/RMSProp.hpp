#pragma once

struct RMSProp {
    public:
    void Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t weightSize, size_t biasSize, size_t elements, float learningRate);

    void Define(const YAML::Node& config);
    void Build(size_t weightSize, size_t biasSize);

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    float decay;
    float epsilon;
    Tensor<float> weightSquares;
    Tensor<float> biasSquares;

    void Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& squares, size_t numParameters, size_t elements, float learningRate);
};
