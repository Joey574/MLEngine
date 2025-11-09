#pragma once
#include "SGD/SGD.hpp"
#include "MomentumSGD/MomentumSGD.hpp"
#include "RMSProp/RMSProp.hpp"
#include "Adam/Adam.hpp"

/// @brief Acts as a wrapper to various Optimizer implementations
struct Optimizer {
    public:
    enum class Type {
        None, SGD, MomentumSGD, RMSProp, Adam
    };

    void Define(const YAML::Node& config, size_t weightSize, size_t biasSize);
    void Build(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives);

    void Update(size_t elements);

    static inline Type ParseType(const std::string& name) {
        auto lower = std::string(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "sgd") {
            return Type::SGD;
        } else if (lower == "momentumsgd") {
            return Type::MomentumSGD;
        } else if (lower == "rmsprop") {
            return Type::RMSProp;
        } else if (lower == "adam") {
            return Type::Adam;
        } else {
            return Type::None;
        }
    }
    static inline std::string ParseName(const Type type) {
        switch (type) {
            case Type::None:
                return "None";
            case Type::SGD:
                return "SGD";
            case Type::MomentumSGD:
                return "MomentumSGD";
            case Type::RMSProp:
                return "RMSProp";
            case Type::Adam:
                return "Adam";
            default:
                return "";
        }
    }

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }
    inline Type GetType() const { return type; }
    
    private:
    using Data = std::variant<SGD, MomentumSGD, RMSProp, Adam>;

    Type type;
    Data data;

    // data needed by all optimizer implementations
    bool defined = false;
    bool built = false;

    float learningRate;
    size_t weightSize;
    size_t biasSize;

    Tensor<float>* weights;
    Tensor<float>* biases;
    Tensor<float>* weightDerivatives;
    Tensor<float>* biasDerivatives;
};
