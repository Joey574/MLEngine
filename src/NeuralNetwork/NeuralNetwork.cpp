#include "NeuralNetwork.hpp"

int NeuralNetwork::Define(YAML::Node& config, Dataset& dataset, const TrainingConfig& trainingConfig) {
    assert(!(defined || built));
    assert(dataset.IsDefined());
    this->config = &config;
    this->dataset = &dataset;
    this->trainingConfig = &trainingConfig;

    auto layerConfigs = config[Y_LAYERS];
    auto optimizerConfig = config[Y_OPT_OPTIMIZER];

    layers = std::vector<Layer>(layerConfigs.size());

    for (size_t i = 0; i < layers.size(); i++) {
        auto layerConf = layerConfigs[i];
        size_t iNodes = i == 0 ? 0 : layerConfigs[i-1][Y_NODES].as<size_t>();
        size_t nNodes = i == layers.size()-1 ? 0 : layerConfigs[i+1][Y_NODES].as<size_t>();

        layers[i].Define(layerConf, optimizerConfig, trainingConfig, iNodes, nNodes);
    }

    defined = true;
    return 0;
}

int NeuralNetwork::Build() {
    assert(defined && !built);
    assert(dataset->IsBuilt());
    std::cout << "[i] Building neural network\n";

    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].Build();
    }

    built = true;
    return 0;
}
