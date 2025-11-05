#pragma once

struct Optimizer {
public:
    enum class Type {
        None, SGD, MomentumSGD, RMSProp, Adam
    };

    void Define(YAML::Node& config);
    void Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }
private:
    bool defined = false;
    bool built = false;

};
