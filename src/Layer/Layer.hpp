#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Optimizer/Optimizer.hpp"

struct Layer {
public:
    enum class Type {
        None, Input, Hidden, Output
    };
    enum class WeightInitialization {
        None, He, Normalize, Xavier
    };


    inline Type LayerType() { return layerType; }

private:
    Type layerType;

    Activation activation;
    LossMetric lossMetric;
    Optimizer optimizer;
};
