#pragma once

struct MomentumSGD {
    public:
    void Compute();

    void Define();
    void Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    float momentum;
    float* weightVelocity;
    float* biasVelocity;
};
