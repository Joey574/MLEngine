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

        Layer l;
        l.Initialize(type, in, n, nn, actv, lm);        
        m_layers.push_back(l);

        m_network_size += l.layer_size;
    }

    // initialize network memory
    m_network = (float*)aligned_alloc(32, m_network_size*sizeof(float));

    // initialize weights
    InitializeWeights(weightInit, dims);
}
void NeuralNetwork::InitializeWeights(Layer::WeightInitialization type, const std::vector<std::size_t>& layers) {
    size_t dataidx = 0;

    for (Layer& layer : m_layers) {
        layer.InitializeWeights(&m_network[dataidx], type, m_seed++);
        dataidx += layer.layer_size;
    }
}
void NeuralNetwork::InitializeLayerData(size_t bn, size_t tn) {
    m_batch_data_size = 0;
    m_test_data_size = 0;

    for (Layer& layer : m_layers) {
        layer.InitializeSizes(bn, tn);

        m_batch_data_size += layer.layer_batch_size;
        m_test_data_size += layer.layer_test_size;
    }

    m_batch_data = (float*)aligned_alloc(32, m_batch_data_size*sizeof(float));
    m_test_data = (float*)aligned_alloc(32, m_test_data_size*sizeof(float));
}
void NeuralNetwork::InitializeLayerPointers(size_t bn, size_t tn) {
    size_t dataidx = 0;
    size_t batchidx = 0;
    size_t testidx = 0;
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        float* data = &m_network[dataidx];
        float* batchdata = &m_batch_data[batchidx];
        float* testdata = &m_test_data[testidx];

        m_layers[i].InitializePointers(data, batchdata, testdata, bn, tn);

        dataidx += m_layers[i].layer_size;
        batchidx += m_layers[i].layer_batch_size;
        testidx += m_layers[i].layer_test_size;
    }

    for (size_t i = 0; i < m_layers.size(); i++) {
        float* nextweights = i == m_layers.size()-1 ? nullptr : m_layers[i+1].Weights();

        m_layers[i].InitializeSpecialPointers(nextweights);
    }
}