#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Optimizer/Optimizer.hpp"

struct Layer {
public:
    enum Type {
        None, Input, Hidden, Output
    };
    enum WeightInitialization {
        None, He, Normalize, Xavier
    };


    inline Type LayerType() { return layerType; }

private:
    Type layerType;

    Activation activation;
    LossMetric lossMetric;
    Optimizer optimizer;

    Tensor<float> weights;
    Tensor<float> biases;
};
