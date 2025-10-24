#pragma once
#include "../Layer/Layer.hpp"
#include "../Dataset/Dataset.hpp"

struct NeuralNetwork {
public:

    int Define(YAML::Node& config, Dataset& dataset);
    int Build();

    void Load(std::ifstream& file);
    void Save(std::ofstream& file) const;
    void LoadOptimizers(std::ifstream& file);
    void SaveOptimizers(std::ofstream& file) const;

    void Forward();
    void Backward();
    float Score();

    bool IsDefined() const { return defined; }
    bool IsBuilt() const { return built; }

private:

    bool defined = false;
    bool built = false;

    Dataset* dataset;
    std::vector<Layer> layers;

    YAML::Node* config;
};
