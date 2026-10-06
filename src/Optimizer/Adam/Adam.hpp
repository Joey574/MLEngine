#pragma once

struct Adam {
  public:
    void Update(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives, size_t elements, float learningRate);

    void Define(const YAML::Node& config);
    void Build(size_t weightSize, size_t biasSize);

    int Save(std::ofstream& f) const;
    int Load(std::ifstream& f);

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

  private:
    bool defined = false;
    bool built   = false;

    size_t iteration;
    float b1;
    float b2;
    float epsilon;

    Tensor<float> weightVelocity;
    Tensor<float> biasVelocity;
    Tensor<float> weightSquares;
    Tensor<float> biasSquares;

    void Compute(Tensor<float>& parameters, Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t elements, float learningRate);
};
