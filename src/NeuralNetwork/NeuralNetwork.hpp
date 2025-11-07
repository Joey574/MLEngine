#pragma once
#include "../Layer/Layer.hpp"
#include "../Dataset/Dataset.hpp"

struct NeuralNetwork {
    public:

    int Define(YAML::Node& config, Dataset& dataset);
    int Build();

    int Load(std::ifstream& file);
    void Save(std::ofstream& file) const;
    void LoadOptimizers(std::ifstream& file);
    void SaveOptimizers(std::ofstream& file) const;

    void Forward(size_t elements);
    void Backward(size_t elements);
    Score Validate();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    YAML::Node* config;
    Dataset* dataset;

    std::vector<Layer> layers;
};
