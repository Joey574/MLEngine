#include "NeuralNetwork.hpp"

void NeuralNetwork::Initialize(const std::string& path, const std::string& name, YAML::Node& config, bool setweights) {
    std::random_device rd;
    this->config = config;
    m_weightinit = ParseWeight(config[Y_WEIGHT].as<std::string>());
    m_path = path;
    m_name = name;
    m_seed = rd();

    m_layers.reserve(config[Y_LAYERS].size());
    m_network_bytes = 0;

    YAML::Node optimizerConf = config[Y_OPT_OPTIMIZER];

    // define layers
    YAML::Node layers = config[Y_LAYERS];
    for (size_t i = 0; i < layers.size(); i++) {
        size_t in = i == 0 ? 0 : layers[i-1][Y_NODES].as<size_t>();
        size_t nn = i == layers.size()-1 ? 0 : layers[i+1][Y_NODES].as<size_t>();

        Layer layer;
        layer.Define(m_layers, i, layers[i], optimizerConf, in, nn);
        m_layers.push_back(layer);
    }

    // initialize layers
    for (size_t i = 0; i < layers.size(); i++) {
        m_layers[i].Initialize();
        m_network_bytes += m_layers[i].layer_bytes;
    }

    // initialize network memory
    m_network = (float*)aligned_alloc(32, m_network_bytes*sizeof(float));

    if (setweights) {
        InitializeWeights(ParseWeight(config[Y_WEIGHT].as<std::string>()));
    }
}

void NeuralNetwork::InitializeWeights(Layer::WeightInitialization type) {
    size_t dataidx = 0;
    memset(m_network, 0, m_network_bytes);
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].InitializeWeights(&m_network[dataidx], type, m_seed+i);
        dataidx += m_layers[i].params;
    }
}
void NeuralNetwork::InitializeLayerData(size_t bn, size_t tn) {
    m_batch_data_bytes = 0;
    m_test_data_bytes = 0;

    for (Layer& layer : m_layers) {
        layer.InitializeSizes(bn, tn);

        m_batch_data_bytes += layer.layer_batch_bytes;
        m_test_data_bytes += layer.layer_test_bytes;
    }

    m_batch_data = (float*)aligned_alloc(32, m_batch_data_bytes);
    m_test_data = (float*)aligned_alloc(32, m_test_data_bytes);

    memset(m_batch_data, 0, m_batch_data_bytes);
    memset(m_test_data, 0, m_test_data_bytes);
}
void NeuralNetwork::InitializeLayerPointers(size_t bn, size_t tn) {
    size_t dataidx = 0;
    size_t batchidx = 0;
    size_t testidx = 0;

    char* net = (char*)m_network;
    char* batch = (char*)m_batch_data;
    char* test = (char*)m_test_data;
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        char* data = &net[dataidx];
        char* batchdata = &batch[batchidx];
        char* testdata = &test[testidx];

        m_layers[i].InitializePointers(data, batchdata, testdata, bn, tn);

        dataidx += m_layers[i].layer_bytes;
        batchidx += m_layers[i].layer_batch_bytes;
        testidx += m_layers[i].layer_test_bytes;
    }

    for (size_t i = 0; i < m_layers.size(); i++) {
        float* nextweights = i == m_layers.size()-1 ? nullptr : m_layers[i+1].Weights();

        m_layers[i].InitializeSpecialPointers(nextweights);
    }
}