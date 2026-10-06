#pragma once
#include "../Dataset/Dataset.hpp"
#include "../Layer/Layer.hpp"

struct NeuralNetwork {
  public:
    int Define(YAML::Node& config, Dataset& dataset, const TrainingConfig& trainingConfig);
    int Build();

    int Save(std::ofstream& f) const;
    int Load(std::ifstream& f);
    int SaveOptimizers(std::ofstream& f) const;
    int LoadOptimizers(std::ifstream& f);

    void Forward(size_t startElement, size_t numElements);
    void Backward(size_t startElement, size_t numElements);
    Score Validate();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

  private:
    bool defined = false;
    bool built   = false;

    const TrainingConfig* trainingConfig;
    YAML::Node* config;
    Dataset* dataset;

    std::vector<Layer> layers;
};
