#include "NeuralNetwork.hpp"

void NeuralNetwork::Initialize(const std::string& path, const std::string& name, const std::vector<size_t>& dimensions, const std::vector<ActivationFunctions>& activations, LossMetric loss, LossMetric metric, WeightInitialization weightInit) {
    std::random_device rd;
    m_weight_init = weightInit;
    m_path = path;
    m_name = name;
    m_seed = rd();

    if (dimensions.size() != activations.size()+1) {
        std::cerr << "activations must be one less in size than dimensions\n";
        return;
    }

    // set dimensions
    m_layers = std::vector<Layer>(dimensions.size());
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].nodes = dimensions[i];
    }

    AssignActvFunctions(activations);
    AssignLossFunctions(loss, metric);

    // main initialization of all the internal goodies
    InitializeNetwork();

    // set the weights
    InitializeWeights();
}

void NeuralNetwork::InitializeNetwork() {
    m_network_size = 0;
    m_weights_size = 0;
    m_biases_size = 0;

    // set network sizing based on layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        m_weights_size += m_layers[i].nodes * m_layers[i+1].nodes;
        m_biases_size += m_layers[i+1].nodes;
    }
    m_network_size = m_weights_size + m_biases_size;

    m_network = (float*)aligned_alloc(32, m_network_size*sizeof(float));
    m_biases = m_network + m_weights_size;
}
void NeuralNetwork::InitializeWeights() {
    float lowerRand;
    float upperRand;
    size_t idx = 0;
    
    std::default_random_engine gen(m_seed);

    // zero out biases
    memset(m_biases, 0, m_biases_size*sizeof(float));

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
            // no weight initialization has been set, zero the network
            memset(m_network, 0, m_weights_size*sizeof(float));
    }
}
void NeuralNetwork::InitializeBatchData(size_t num_elements) {
    m_batch_activation_size = 0;

    for (size_t i = 1; i < m_layers.size(); i++) {
        m_batch_activation_size += m_layers[i].nodes * num_elements;
    }

    m_batch_data_size = (3 * m_batch_activation_size) + m_network_size;

    m_batch_data = (float*)aligned_alloc(32, m_batch_data_size*sizeof(float));
    m_activation = &m_batch_data[m_batch_activation_size];

    // set derivative pointers
    m_d_total = &m_activation[m_batch_activation_size];
    m_d_weights = &m_d_total[m_batch_activation_size];
	m_d_biases = &m_d_weights[m_weights_size];
}
void NeuralNetwork::InitializeTestData(size_t num_elements) {
    m_test_activation_size = 0;

    for (size_t i = 1; i < m_layers.size(); i++) {
        m_test_activation_size += m_layers[i].nodes * num_elements;
    }

    m_test_data_size = m_test_activation_size * 2;

    m_test_data = (float*)aligned_alloc(32, m_test_data_size*sizeof(float));
    m_test_activation = &m_test_data[m_test_activation_size];
}

void NeuralNetwork::InitializeLoss(LossMetric loss) {
    m_loss.type = loss;

    switch(loss) {
        case LossMetric::mae:
            m_loss.loss = &MaeLoss;
            break;
        case LossMetric::mse:
            m_loss.loss = &MseLoss;
            break;
        case LossMetric::onehot:
            m_loss.loss = &OneHotLoss;
            break;
        default:
            m_loss.type = LossMetric::none;
            m_loss.loss = nullptr;
            break;
    }
}
void NeuralNetwork::InitializeMetric(LossMetric metric) {
    m_metric.type = metric;

    switch (metric) {
        case LossMetric::mae:
            m_metric.metric = &NeuralNetwork::MaeScore;
            break;
        case LossMetric::mse:
            m_metric.metric = &NeuralNetwork::MseScore;
            break;
        case LossMetric::accuracy:
            m_metric.metric = &NeuralNetwork::AccuracyScore;
            break;
        default:
            m_metric.type = LossMetric::none;
            m_metric.metric = nullptr;
            break;
    }
}