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

    void Forward();
    void Backward();

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
    size_t i_nodes;
    size_t o_nodes;
};
