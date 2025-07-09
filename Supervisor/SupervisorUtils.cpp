#include "Supervisor.hpp"

void Supervisor::InitializeNetworks() {
    for (NeuralNetwork* nn : m_networks) {
        //nn->Initialize();
    }
}
void Supervisor::StartNetworks() {
    for (NeuralNetwork* nn : m_networks) {
        //nn->Start();
    }
}
void Supervisor::EndNetworks() {
    for (NeuralNetwork* nn : m_networks) {
        //nn->End();
    }
}
void Supervisor::AdvanceNetworks() {
    for (NeuralNetwork* nn : m_networks) {
        //nn->Fit(m_dataset);
    }
}
void Supervisor::TestNetworks() {
    std::vector<const float* __restrict> predictions;

    // get all network's predictions
    for (NeuralNetwork* nn : m_networks) {
        predictions.push_back(nn->Predict(m_dataset));
    }

    // score networks
}
