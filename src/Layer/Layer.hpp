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

    void Define(const YAML::Node& layerConfig, const YAML::Node& optimizerConfig, const TrainingConfig& trainingConfig, size_t in, size_t out);
    void Build();

    template <bool training> void Forward(const Tensor<float>& input, size_t elements);
    template <bool training> void InputForward(const Tensor<float>& input, size_t elements);
    template <bool training> void HiddenForward(const Tensor<float>& input, size_t elements);

    void Backward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements);
    void InputBackward();
    void HiddenBackward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements);
    void OutputBackward(const Tensor<float>& truth, const Tensor<float>& input, size_t elements);
    void ComputeBackward(const Tensor<float>& input, size_t elements);

    void Update(size_t elements);

    template <bool train> inline Tensor<float>& Output() {
        if constexpr (train) {
            return trainingActivations;
        } else {
            return testingActivations;
        }
    }
    inline Tensor<float>& Weights() { return weights; }

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
    
    Tensor<float> weights;
    Tensor<float> biases;

    Tensor<float> totalDerivatives;
    Tensor<float> weightDerivatives;
    Tensor<float> biasDerivatives;

    Tensor<float> trainingTotals;
    Tensor<float> trainingActivations;
    Tensor<float> testingTotals;
    Tensor<float> testingActivations;
};
