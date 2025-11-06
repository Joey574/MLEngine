#pragma once


struct MomentumSGD {
    public:
    void Compute();

    private:

};

struct RMSProp {
    public:
    void Compute();

    private:

};

struct Adam {
    public:
    void Compute();
    
    private:

};


struct Optimizer {
    using Data = std::variant<MomentumSGD, RMSProp, Adam>;

    public:
    enum class Type {
        None, SGD, MomentumSGD, RMSProp, Adam
    };

    void Define(YAML::Node& config);
    void Build();

    void Compute();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }
    inline Type GetType() const { return type; }
    
    private:
    Type type;
    bool defined = false;
    bool built = false;

    Data data;
};
