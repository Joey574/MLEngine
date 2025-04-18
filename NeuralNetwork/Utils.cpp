#include "NeuralNetwork.hpp"

std::string NeuralNetwork::compactDimensions() const {
    std::string compact = "";
    
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        compact += std::to_string(m_layers[i].nodes).append("-");
    }

    compact += std::to_string(m_layers.back().nodes);
    return compact;
}
std::string NeuralNetwork::compactActvations() const {
    std::string compact = "";

    for (size_t i = 1; i < m_layers.size() - 1; i++) {
        compact += activationString(m_layers[i].actv).append("-");
    }

    compact += activationString(m_layers.back().actv);
    return compact;
}

nlohmann::json NeuralNetwork::metadata() const {
    /*
        loss
        metric
        weight_init
        dimensions
        activations
    */

    nlohmann::json metadata;
    metadata["loss"] = lossMetricString(m_loss.type);
    metadata["metric"] = lossMetricString(m_metric.type);
    metadata["weights"] = weightString(m_weight_init);
    metadata["dimensions"] = compactDimensions();
    metadata["activations"] = compactActvations();
    metadata["parameters"] = m_network_size;

    return metadata;
}