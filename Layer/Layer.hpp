#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"

struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output
    };

    Layer(float* w, float* b, size_t in, size_t n, Activation actv, LossMetric lm) : 
    m_w(w), m_b(b), inodes(in), nodes(n), size(in*n), activation(actv), lossmetric(lm) {}

    std::string name;
    LayerType type;

    size_t nodes;
    size_t inodes;
    size_t size;

    LossMetric lossmetric;
    Activation activation;

    void forward(
        bool training,
        const float* __restrict x,
        float* __restrict z,
        float* __restrict a,
        size_t n
    );

    void backward(
        const float* __restrict y,
        const float* __restrict pa,
        const float* __restrict z,
        const float* __restrict a,
        const float* __restrict nw,
        float* __restrict dt,
        float* __restrict dw,
        float* __restrict db,
        size_t nenodes,
        size_t n
    );
    
    nlohmann::json metadata() const;

private:
    const float* m_w;
    const float* m_b;
};
