#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Optimizer/Optimizer.hpp"

struct Layer {
    public:
    enum class Type {
        None, Input, Hidden, Output
    };
    enum class WeightInitialization {
        None, He, Normalize, Xavier
    };

    inline Type GetType() const { return type; }
    static inline Type ParseType(const std::string& name) {
        std::string lower(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "input") {
            return Type::Input;
        } else if (lower == "hidden") {
            return Type::Hidden;
        } else if (lower == "output") {
            return Type::Output;
        } else {
            return Type::None;
        }
    }
    static inline std::string ParseName(const Type type) {
        switch (type) {
            case Type::None:
                return "None";
            case Type::Input:
                return "Input";
            case Type::Hidden:
                return "Hidden";
            case Type::Output:
                return "Output";
            default:
                return "";
        }
    }

    void Define(YAML::Node& layerConfig, YAML::Node& optimizerConfig, size_t in, size_t out);
    void Build();

    void Forward(const float* __restrict input, size_t elements);
    void InputForward(const float* __restrict input, size_t elements);
    void HiddenForward(const float* __restrict input, size_t elements);

    void Backward(const float* __restrict truth, const float* __restrict input, const float* __restrict nextWeights, size_t elements);
    void InputBackward();
    void HiddenBackward(const float* __restrict truth, const float* __restrict input, const float* __restrict nextWeights, size_t elements);
    void OutputBackward(const float* __restrict truth, const float* __restrict input, size_t elements);
    void ComputeBackward(const float* __restrict input, size_t elements);

    void Update(size_t elements);

    template <bool train> inline float* Output() {
        if constexpr (train) {
            return trainingActivations;
        } else {
            return testingActivations;
        }
    }
    inline float* Weights() { return weights; }

    inline bool IsDefined() { return defined; }
    inline bool IsBuilt() { return built; }

    private:
    bool defined = false;
    bool built = false;

    Type type;

    Activation activation;
    LossMetric lossMetric;
    Optimizer optimizer;

    size_t nodes;
    size_t iNodes;
    size_t oNodes;
    

    float* weights;
    float* biases;
    float* weightDerivatives;
    float* biasDerivatives;
    float* totalDerivatives;

    size_t weightSize;
    size_t biasSize;


    float* trainingTotals;
    float* trainingActivations;
    float* testingTotals;
    float* testingActivations;
};
