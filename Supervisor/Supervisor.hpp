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

    void Define(YAML::Node& config);

    void Load(const std::string& path, const std::string& name);

    nlohmann::json Train(nlohmann::json& history);

private:

    NeuralNetwork* model;
    Dataset* dataset;

    YAML::Node* config;

    void Save(const std::string& path, const std::string& name);
};
