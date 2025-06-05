#include "NeuralNetwork.hpp"

void NeuralNetwork::Initialize(const std::string& path, const std::string& name, const std::vector<size_t>& dims, const std::vector<Activation::Type>& actvs, LossMetric::Type loss, LossMetric::Type metric, Layer::WeightInitialization weightInit) {
    std::random_device rd;
    m_weightinit = weightInit;
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

    // initialize layer size
    m_layers.reserve(dims.size());

    m_network_size = 0;

    // construct layers
    for (size_t i = 0; i < dims.size(); i++) {

        Layer::LayerType type = 
            i == 0 ? Layer::LayerType::input : 
            i == dims.size()-1 ? Layer::LayerType::output : 
            Layer::LayerType::hidden;

        size_t in = i == 0 ? 0 : dims[i-1];
        size_t n = dims[i];
        size_t nn = i == dims.size()-1 ? 0 : dims[i+1];

        Activation actv = i == 0 ? Activation() : Activation(actvs[i-1]);
        LossMetric lm = i < dims.size()-1 ? LossMetric() : LossMetric(loss, metric);

        Layer layer;
        layer.Initialize(type, in, n, nn, actv, lm, 0.2f);        
        m_layers.push_back(layer);

        m_network_size += layer.params;
    }

    // initialize network memory
    m_network = (float*)aligned_alloc(32, m_network_size*sizeof(float));

    // initialize weights
    InitializeWeights(weightInit, dims);
}
void NeuralNetwork::InitializeWeights(Layer::WeightInitialization type, const std::vector<std::size_t>& layers) {
    size_t dataidx = 0;

    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].InitializeWeights(&m_network[dataidx], type, m_seed+i);
        dataidx += m_layers[i].params;
    }
}
void NeuralNetwork::InitializeLayerData(size_t bn, size_t tn) {
    size_t batch_bytes = 0;
    size_t test_bytes = 0;

    for (Layer& layer : m_layers) {
        layer.InitializeSizes(bn, tn);

        batch_bytes += layer.layer_batch_bytes;
        test_bytes += layer.layer_test_bytes;
    }

    m_batch_data_size = batch_bytes/sizeof(float);
    m_test_data_size = test_bytes/sizeof(float);

    m_batch_data = (float*)aligned_alloc(32, batch_bytes);
    m_test_data = (float*)aligned_alloc(32, test_bytes);
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