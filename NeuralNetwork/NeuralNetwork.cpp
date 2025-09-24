#include "NeuralNetwork.hpp"

void NeuralNetwork::Define(YAML::Node& config, Dataset& dataset) {
    this->config = &config;
    this->dataset = &dataset;
}
