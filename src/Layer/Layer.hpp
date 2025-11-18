#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Optimizer/Optimizer.hpp"

struct Layer {
    public:
    enum class Type {
        None, Input, Hidden, Output
    };
    enum class WeightType {
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
    static inline WeightType ParseWeightType(const std::string& name) {
        std::string lower(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "he") {
            return WeightType::He;
        } else if (lower == "normalize") {
            return WeightType::Normalize;
        } else if (lower == "xavier") {
            return WeightType::Xavier;
        } else {
            return WeightType::None;
        }
    }
    static inline std::string ParseWeightName(const WeightType type) {
        switch (type) {
            case WeightType::None:
                return "None";
            case WeightType::He:
                return "He";
            case WeightType::Normalize:
                return "Normalize";
            case WeightType::Xavier:
                return "Xavier";
            default:
                return "";
        }
    }

    void Define(const YAML::Node& layerConfig, const YAML::Node& optimizerConfig, const TrainingConfig& trainingConfig, size_t in, size_t out);
    void Build();

    int Save(std::ofstream& f) const;
    int Load(std::ifstream& f);
    inline int SaveOptimizers(std::ofstream& f) const { return optimizer.Save(f); }
    inline int LoadOptimizers(std::ifstream& f) { return optimizer.Load(f); }

    template <bool TRAINING> void Forward(const Tensor<float>& input);
    template <bool TRAINING> void InputForward(const Tensor<float>& input);
    template <bool TRAINING> void HiddenForward(const Tensor<float>& input);

    void Backward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements);
    void InputBackward();
    void HiddenBackward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements);
    void OutputBackward(const Tensor<float>& truth, const Tensor<float>& input, size_t elements);
    void ComputeBackward(const Tensor<float>& input, size_t elements);

    void Update(size_t elements);
    
    inline float Score(const Tensor<float>& truth) const {
        return (*lossMetric.metric)(testingActivations, truth);
    }

    template <bool TRAIN> inline Tensor<float>& Output() {
        if constexpr (TRAIN) {
            return trainingActivations;
        } else {
            return testingActivations;
        }
    }
    inline Tensor<float>& Weights() { return weights; }

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    Type type;
    WeightType weightType;

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

    void InitializeParameters();
};
