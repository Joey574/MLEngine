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

    void Define(YAML::Node& config);
    void Build();

    void Compute();

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

    inline bool IsDefined() const { return std::visit([](auto& data){ return data.IsDefined(); } , data); }
    inline bool IsBuilt() const { return std::visit([](auto& data){ return data.IsBuilt(); } , data); }
    inline Type GetType() const { return type; }
    
    private:
    using Data = std::variant<SGD, MomentumSGD, RMSProp, Adam>;

    Type type;
    Data data;
};
