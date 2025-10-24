#include "NeuralNetwork.hpp"

int NeuralNetwork::Define(YAML::Node& config, Dataset& dataset) {
    assert(!(defined || built));
    assert(dataset.IsDefined());
    this->config = &config;
    this->dataset = &dataset;

    defined = true;
    return 0;
}

int NeuralNetwork::Build() {
    assert(defined && !built);
    assert(dataset->IsBuilt());

    built = true;
    return 0;
}
