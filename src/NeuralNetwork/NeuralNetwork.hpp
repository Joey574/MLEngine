#pragma once
#include "../Layer/Layer.hpp"
#include "../Dataset/Dataset.hpp"

struct NeuralNetwork {
    public:

    int Define(YAML::Node& config, Dataset& dataset, const TrainingConfig& trainingConfig);
    int Build();

    int Load(std::ifstream& file);
    void Save(std::ofstream& file) const;
    void LoadOptimizers(std::ifstream& file);
    void SaveOptimizers(std::ofstream& file) const;

    void Forward(size_t startElement, size_t numElements);
    void Backward(size_t startElement, size_t numElements);
    Score Validate();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;

    const TrainingConfig* trainingConfig;
    YAML::Node* config;
    Dataset* dataset;

    std::vector<Layer> layers;
};
