#include "Layer.hpp"

void Layer::Define(const YAML::Node& layerConfig, const YAML::Node& optimizerConfig, const TrainingConfig& trainingConfig, size_t in, size_t out) {
    assert(!(defined || built));
    assert(!(optimizer.IsDefined() || optimizer.IsBuilt()));
    assert(layerConfig[Y_LAYERTYPE] && layerConfig[Y_NODES]);

    type = ParseType(layerConfig[Y_LAYERTYPE].as<std::string>());

    // set nodes, input nodes, and output nodes sizes
    nodes = layerConfig[Y_NODES].as<size_t>();
    iNodes = in;
    oNodes = out;

    // set activation function, default to linear
    activation.AssignPointers(layerConfig[Y_ACTIVATION].as<std::string>(Y_ACTV_DEFAULT));

    // set loss / metric functions, default to none
    lossMetric.AssignPointers(
        layerConfig[Y_LOSS].as<std::string>(Y_LOSS_DEFAULT),
        layerConfig[Y_METRIC].as<std::string>(Y_METRIC_DEFAULT)
    );

    // allocate tensors
    switch (type) {
        case Type::Input:
            weights = Tensor<float>(0);
            biases = Tensor<float>(0);
            weightDerivatives = Tensor<float>(0);
            biasDerivatives = Tensor<float>(0);
            break;
        case Type::Hidden: case Type::Output:
            weights = Tensor<float>(iNodes, nodes);
            weightDerivatives = Tensor<float>(iNodes, nodes);
            biases = Tensor<float>(nodes);
            biasDerivatives = Tensor<float>(nodes);
            break;
    }

    // allocate training and testing tensors
    testingTotals = Tensor<float>(nodes, trainingConfig.testSize);
    trainingTotals = Tensor<float>(nodes, trainingConfig.batchSize);
    testingActivations = Tensor<float>(nodes, trainingConfig.testSize);
    trainingActivations = Tensor<float>(nodes, trainingConfig.batchSize);

    optimizer.Define(optimizerConfig, weights.Size(), biases.Size());
    defined = true;
}

void Layer::Build() {
    assert(defined && !built);
    assert(optimizer.IsDefined() && !optimizer.IsBuilt());

    optimizer.Build(weights, biases, weightDerivatives, biasDerivatives);
    built = true;
}
