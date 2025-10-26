#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../Dataset/Dataset.hpp"

struct Supervisor {
public:

    Supervisor() {
        model = new NeuralNetwork();
        dataset = new Dataset();
    }
    ~Supervisor() {
        delete model;
        delete dataset;
    }

    int Define(YAML::Node& config);
    int Build();

    int Load(const std::string& path, const std::string& name);

    nlohmann::json Train(nlohmann::json& history);

    bool IsDefined() const { return defined; }
    bool IsBuilt() const { return built; }

private:

    bool defined = false;
    bool built = false;

    NeuralNetwork* model;
    Dataset* dataset;

    YAML::Node* config;

    void Save(const std::string& path, const std::string& name);
};
