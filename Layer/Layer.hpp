#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

struct Layer {
    enum class LayerType {
        none, input, hidden, output
    };

    Layer(size_t in, size_t n) : inodes(in), nodes(n) {}

    std::string name;
    LayerType type;

    size_t nodes;
    size_t inodes;

    
};

struct Layer {
    enum class LayerType {
        none, input, hidden, output
    };

    std::string name;
    LayerType type;

    NeuralNetwork::Metric metric;
    NeuralNetwork::Loss loss;

    // previous layer's nodes
    size_t pnodes;
    // current layer's nodes
    size_t nodes;

    NeuralNetwork::ActivationFunction actvtype;
    void (*activation)(const float*, float*, size_t);
    void (*derivative)(const float*, float*, size_t);

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
