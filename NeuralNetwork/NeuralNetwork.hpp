#pragma once
#include "../Layer/Layer.hpp"
#include "../Dataset/Dataset.hpp"

struct NeuralNetwork {
public:

    void Define(YAML::Node& config, Dataset& dataset);

    void Load(std::ifstream& file);
    void Save(std::ofstream& file) const;
    void LoadOptimizers(std::ifstream& file);
    void SaveOptimizers(std::ofstream& file) const;

    void Forward();
    void Backward();
    float Score();

private:
    Dataset* dataset;
    std::vector<Layer> layers;

    YAML::Node* config;
};
