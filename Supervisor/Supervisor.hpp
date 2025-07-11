#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../DataLoader/DataLoader.hpp"

/* @brief

*/
struct Supervisor {
public:

    enum class EnsembleTechnique {
        none, sum, average
    };

    void Train();

private:
    YAML::Node config;
    nlohmann::json history;

    std::vector<NeuralNetwork*> m_networks;
    DataLoader m_dataset;

    // network wrapper functions
    void InitializeNetworks();
    void StartNetworks();
    void EndNetworks();
    void AdvanceNetworks();
    void TestNetworks();
};
