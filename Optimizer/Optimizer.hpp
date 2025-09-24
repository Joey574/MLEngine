#pragma once

struct SGD {
public:
    void Define(YAML::Node& config);
    void Update(size_t batchSize) const;

    static size_t SizeOfType(size_t weights, size_t biases);
};

struct MomentumSGD {
public:
    void Define(YAML::Node& config);
    void Update(size_t batchSize) const;

    static size_t SizeOfType(size_t weights, size_t biases);

private:
    float momentumCoef;

    Tensor<float> momentumWeights;
    Tensor<float> momentumBiases;
};

struct RMSProp {
public:
    void Define(YAML::Node& config);
    void Update(size_t batchSize) const;

    static size_t SizeOfType(size_t weights, size_t biases);

private:
    float decay;
    float epsl;

    Tensor<float> meanSquareWeights;
    Tensor<float> meanSquareBiases;
};

struct Adam {
public:
    void Define(YAML::Node& config);
    void Update(size_t batchSize) const;

    static size_t SizeOfType(size_t weights, size_t biases);

private:
    float biasCorrection1;
    float biasCorrection2;
    float epsl;

    size_t timeStep;

    Tensor<float> firstMomentWeights;
    Tensor<float> firstMomentBiases;

    Tensor<float> secondMomentWeights;
    Tensor<float> secondMomentBiases;
};

struct Optimizer {
public:
    using OptimizerVariant = std::variant<SGD, MomentumSGD, RMSProp, Adam>;

    enum Type {
        None, SGD, MomentumSGD, RMSProp, Adam
    };

    Optimizer(Type type = Type::None) {
        this->type = type;
    }

    void Define(YAML::Node& config);
    void Update(size_t batchSize) const;

    void Save(std::ofstream& file) const;
    void Load(std::ifstream& file) const;

    Type OptimizerType() const { return type; }

    static size_t SizeOfType(Type type, size_t weights, size_t biases);

    static Type ParseType(const std::string& name);
    static std::string ParseName(Type type);
    
private:
    Type type;
    OptimizerVariant variant;

    float learningRate;

    Tensor<float>* weights;
    Tensor<float>* biases;

    Tensor<float>* derivativeWeights;
    Tensor<float>* derivativeBiases;
};
