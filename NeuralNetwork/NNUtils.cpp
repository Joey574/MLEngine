#include "NeuralNetwork.hpp"

std::string NeuralNetwork::CompactDimensions() const {
    std::string compact = "";
    
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        compact += std::to_string(m_layers[i].nodes).append("-");
    }

    compact += std::to_string(m_layers.back().nodes);
    return compact;
}
std::string NeuralNetwork::CompactActvations() const {
    std::string compact = "";

    for (size_t i = 1; i < m_layers.size() - 1; i++) {
        compact += ActivationString(m_layers[i].actv).append("-");
    }

    compact += ActivationString(m_layers.back().actv);
    return compact;
}

nlohmann::json NeuralNetwork::Metadata() const {
    /*
        loss
        metric
        weight_init
        dimensions
        activations
    */

    nlohmann::json metadata;
    metadata["loss"] = LossMetricString(m_loss.type);
    metadata["metric"] = LossMetricString(m_metric.type);
    metadata["weights"] = WeightString(m_weight_init);
    metadata["dimensions"] = CompactDimensions();
    metadata["activations"] = CompactActvations();
    metadata["parameters"] = m_network_size;

    return metadata;
}

int NeuralNetwork::Save(int fd) const {
    ssize_t n = write(fd, m_network, m_network_size*sizeof(float));
    if (n != m_network_size*sizeof(float)) {
        return 1;
    }
    return 0;
}
int NeuralNetwork::Load(int fd, WeightInitialization trueweight) {
    m_weight_init = trueweight;

    ssize_t n = read(fd, m_network, m_network_size*sizeof(float));
    if (n != m_network_size*sizeof(float)) {
        return 1;
    }
    return 0;
}
