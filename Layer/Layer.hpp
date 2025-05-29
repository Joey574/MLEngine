#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

struct Layer {
    enum class LayerType {
        none, input, hidden, output
    };

    Layer(float* __restrict w, float* __restrict b, float* __restrict z, float* __restrict a, size_t in, size_t n) : w(w), b(b), z(z), a(a), inodes(in), nodes(n) {}


    std::string name;
    LayerType type;

    size_t nodes;
    size_t inodes;

    LossMetric lossmetric;
    Activation activation;

    const float* const w;
    const float* const b;
    float* const z;
    float* const a;

    void forward(const float* __restrict x);
    void backward();
};

struct Layer {
    enum class LayerType {
        none, input, hidden, output
    };

  
    float* dt;
    float* dw;
    float* db;

    void forward(
        bool training,
        const float* __restrict x,
        float* __restrict z,
        float* __restrict a,
        size_t n
    ) const;

    void backward(
        float* __restrict w,
        const float* __restrict z,
        const float* __restrict a,
        const float* __restrict x,
        const float* __restrict y, 
        size_t n
    ) const;
};
