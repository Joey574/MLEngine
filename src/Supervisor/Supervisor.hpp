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

    int Define(YAML::Node& config, std::string& path, std::string& name);
    int Build();

    int Load();

    nlohmann::json Train(nlohmann::json& history);

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

private:

    bool defined = false;
    bool built = false;

    std::string path;
    std::string name;

    Score bestScore;
    NeuralNetwork* model;
    Dataset* dataset;

    YAML::Node* config;

    void Save() const;
};
