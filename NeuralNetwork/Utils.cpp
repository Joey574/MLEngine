#include "NeuralNetwork.hpp"

void NeuralNetwork::initialize_network() {
    m_network_size = 0;
    m_weights_size = 0;
    m_biases_size = 0;

    // set network sizing based on layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        m_weights_size += m_layers[i].nodes * m_layers[i+1].nodes;
        m_biases_size += m_layers[i].nodes;
    }
    m_network_size = m_weights_size + m_biases_size;

    m_network = (float*)aligned_alloc(64, m_network_size*sizeof(float));
    m_biases = m_network + m_weights_size;
}
void NeuralNetwork::initialize_batch_data(size_t num_elements) {
    m_batch_activation_size = 0;

    for (size_t i = 0; i < m_layers.size(); i++) {
        m_batch_activation_size += m_layers[i].nodes * num_elements;
    }

    m_batch_data_size = (3 * m_batch_activation_size) + m_network_size;

    m_batch_data = (float*)aligned_alloc(64, m_batch_data_size*sizeof(float));
    m_activation = &m_batch_data[m_batch_activation_size];

    // set derivative pointers
    m_d_total = &m_activation[m_batch_activation_size];
    m_d_weights = &m_d_total[m_batch_activation_size];
	m_d_biases = &m_d_weights[m_weights_size];
}
void NeuralNetwork::initialize_test_data(size_t num_elements) {
    m_test_activation_size = 0;

    for (size_t i = 0; i < m_layers.size(); i++) {
        m_test_activation_size += m_layers[i].nodes * num_elements;
    }

    m_test_data_size = m_test_activation_size * 2;

    m_test_data = (float*)aligned_alloc(64, m_test_data_size*sizeof(float));
    m_test_activation = &m_test_data[m_test_activation_size];
}

void NeuralNetwork::initialize_weights(WeightInitialization init) {
    switch (init) {
        case WeightInitialization::he:
        break;

        case WeightInitialization::normalize:
        break;

        case WeightInitialization::xavier:
        break;
    }
}

std::string NeuralNetwork::compact_dimensions() const {
    std::string compact = "";
    
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        compact += std::to_string(m_layers[i].nodes).append("-");
    }

    compact += std::to_string(m_layers.back().nodes);
    return compact;
}

nlohmann::json NeuralNetwork::metadata() const {
    /*
        [int32]: loss
        [int32]: metric
        [int32]: weight_init
        [variable]: dims_string
    */

    nlohmann::json metadata;
    metadata["loss"] = m_loss.type;
    metadata["metric"] = m_metric.type;
    metadata["weights"] = m_weight_init;
    metadata["dimensions"] = compact_dimensions();
    metadata["activations"] = "TODO";

    return metadata;
}