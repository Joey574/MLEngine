#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output
    };

    Layer(float* w, float* b, size_t in, size_t n, Activation actv, LossMetric lm) : 
    m_w(w), m_b(b), m_z(nullptr), m_a(nullptr), m_dt(nullptr), m_dw(nullptr), m_db(nullptr), inodes(in), nodes(n), activation(actv), lossmetric(lm) {}

    std::string name;
    LayerType type;

    size_t nodes;
    size_t inodes;

    void AssignOutputs(float* z, float* a, float* dt, float* dw, float* db) {
        m_z = z; m_a = a; m_dt = dt; m_dw = dw; m_db = db;
    }

    void forward(bool training, const float* __restrict const x, size_t n);
    void backward(const float* __restrict y, const float* __restrict pa, size_t n);
    float score(const float* __restrict pred, const float* __restrict y, size_t r, size_t c);

    nlohmann::json metadata() const;

private:
    static float Sum256(__m256 _x);

    LossMetric lossmetric;
    Activation activation;

    const float* m_w;
    const float* m_b;

    float* m_z;
    float* m_a;

    float* m_dt;
    float* m_dw;
    float* m_db;
};
