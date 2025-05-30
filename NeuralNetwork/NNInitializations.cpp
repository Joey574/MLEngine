#include "NeuralNetwork.hpp"

void NeuralNetwork::Initialize(const std::string& path, const std::string& name, const std::vector<size_t>& dims, const std::vector<Activation::Type>& actvs, LossMetric::Type loss, LossMetric::Type metric, WeightInitialization weightInit) {
    std::random_device rd;
    m_path = path;
    m_name = name;
    m_seed = rd();

    // grab initial metadata
    std::ifstream f(path+"state.meta");
    try {
        m_meta = nlohmann::json::parse(f);
    } catch (nlohmann::json::parse_error& e) {}
    f.close();

    if (dims.size() != actvs.size()+1) {
        std::cerr << "activations must be one less in size than dimensions\n";
        return;
    }

    // initialize network memory
    InitializeNetwork(dims);    

    // initialize weights
    InitializeWeights(weightInit);

    // initialize layer size
    m_layers.reserve(dims.size());

    size_t widx = 0;
    size_t bidx = 0;

    // construct layers
    for (size_t i = 0; i < dims.size(); i++) {
        float* w = &m_network[widx];
        float* b = &m_biases[bidx];

        size_t in = i == 0 ? 0 : dims[i-1];
        size_t n = dims[i];

        Activation actv = i == 0 ? Activation() : Activation(actvs[i-1]);
        LossMetric lm = i < dims.size()-1 ? LossMetric() : LossMetric(loss, metric);

        m_layers.emplace_back(w, b, in, n, actv, lm);

        widx += in*n;
        widx += i == 0 ? 0 : n;
    }
}

void NeuralNetwork::InitializeNetwork(const std::vector<size_t>& dims) {
    m_weights_size = 0;
    m_biases_size = 0;

    // set network sizing based on layers
    for (size_t i = 1; i < dims.size(); i++) {
        m_weights_size += dims[i-1] * dims[i];
        m_biases_size += dims[i];
    }

    m_network_size = m_weights_size + m_biases_size;
    m_network = (float*)aligned_alloc(32, m_network_size*sizeof(float));
}
void NeuralNetwork::InitializeWeights(WeightInitialization type) {
    float lowerRand;
    float upperRand;
    size_t idx = 0;
    
    std::default_random_engine gen(m_seed);

    // zero out biases
    memset(m_biases, 0, m_biases_size*sizeof(float));

    switch (type) {
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
void NeuralNetwork::InitializeBatchData(size_t n) {
    m_batch_actv_size = 0;

    for (size_t i = 1; i < m_layers.size(); i++) {
        m_batch_actv_size += m_layers[i].nodes * n;
    }

    m_batch_data_size = (3 * m_batch_actv_size) + m_network_size;

    m_batch_data = (float*)aligned_alloc(32, m_batch_data_size*sizeof(float));
    m_batch_actv = &m_batch_data[m_batch_actv_size];

    // set derivative pointers
    m_d_total = &m_batch_actv[m_batch_actv_size];
    m_d_weights = &m_d_total[m_batch_actv_size];
	m_d_biases = &m_d_weights[m_weights_size];
}
void NeuralNetwork::InitializeTestData(size_t n) {
    m_test_actv_size = 0;

    for (size_t i = 1; i < m_layers.size(); i++) {
        m_test_actv_size += m_layers[i].nodes * n;
    }

    m_test_data_size = m_test_actv_size * 2;

    m_test_data = (float*)aligned_alloc(32, m_test_data_size*sizeof(float));
    m_test_actv = &m_test_data[m_test_actv_size];
}
