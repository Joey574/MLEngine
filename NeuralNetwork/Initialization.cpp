#include "NeuralNetwork.hpp"

void NeuralNetwork::initialize(std::vector<size_t> dimensions, std::vector<ActivationFunctions> activations, LossMetric loss, LossMetric metric, WeightInitialization weightInit) {
    m_weight_init = weightInit;
    m_loss.type = loss;
    m_metric.type = metric;

    if (dimensions.size() != activations.size()+1) {
        std::cerr << "activations must be one less in size than dimensions\n";
        return;
    }

    // set dimensions
    m_layers = std::vector<Layer>(dimensions.size());
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].nodes = dimensions[i];
    }

    // set activations
    m_layers[0].actv = ActivationFunctions::none;
    for (size_t i = 1; i < m_layers.size(); i++) {
        m_layers[i].actv = activations[i-1];
    }

    // main initialization of all the internal goodies
    initializeNetwork();

    // set the weights
    initializeWeights();
}

void NeuralNetwork::initializeNetwork() {
    m_network_size = 0;
    m_weights_size = 0;
    m_biases_size = 0;

    // set network sizing based on layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        m_weights_size += m_layers[i].nodes * m_layers[i+1].nodes;
        m_biases_size += m_layers[i+1].nodes;
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


void NeuralNetwork::initializeWeights() {
    float lowerRand;
    float upperRand;
    size_t idx = 0;

    std::random_device rd;
    std::default_random_engine gen(rd());


    switch (m_weight_init) {
        case WeightInitialization::he:
            lowerRand = 0.0f;

            for (size_t i = 0; i < m_layers.size() - 1; i++) {
                upperRand = std::sqrt(2.0f / m_layers[i+1].nodes);

                std::normal_distribution<float> dist(lowerRand, upperRand);
                for(size_t j = 0; j < m_layers[i].nodes * m_layers[i+1].nodes; j++, idx++) {
                    m_network[idx] = dist(gen);
                }
            }

            break;
        case WeightInitialization::normalize:
            lowerRand = -0.5f;
            upperRand = 0.5f;

            for (size_t i = 0; i < m_layers.size() - 1; i++) {
                std::uniform_real_distribution<float> dist(lowerRand, upperRand);

                for (size_t j = 0; j < m_layers[i].nodes * m_layers[i+1].nodes; j++, idx++) {
                    m_network[idx] = dist(gen) * std::sqrt(1.0f / m_layers[i+1].nodes);
                }
            }

            break;
        case WeightInitialization::xavier:

            for (size_t i = 0; i < m_layers.size() - 1; i++) {
                lowerRand = (-1.0f / std::sqrt(m_layers[i+1].nodes));
                upperRand = 1.0f / std::sqrt(m_layers[i+1].nodes);

                std::uniform_real_distribution<float> dist(lowerRand, upperRand);
                for (size_t j = 0; j < m_layers[i].nodes * m_layers[i+1].nodes; j++, idx++) {
                    m_network[idx] = dist(gen);
                }
            }

            break;

        default:
            std::cerr << "no weight initialization set\n";
    }
}
