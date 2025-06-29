#include "Layer.hpp"

void Layer::Define(std::vector<Layer>& layers, size_t idx, YAML::Node config, YAML::Node optimizerConfig, size_t in, size_t nn) {
    this->inodes = in;
    this->nenodes = nn;

    m_layers = &layers;
    m_layer_idx = idx;

    type = ParseType(config[Y_LAYERTYPE].as<std::string>());
    nodes = config[Y_NODES].as<size_t>();

    if (config[Y_ACTIVATION]) {
        activation.AssignPointers(Activation::ParseType(config[Y_ACTIVATION].as<std::string>()));
    }

    if (config[Y_LOSS] && config[Y_METRIC]) {
        lossmetric.AssignPointers(
            LossMetric::ParseType(config[Y_LOSS].as<std::string>()),
            LossMetric::ParseType(config[Y_METRIC].as<std::string>())
        );
    }

    if (config[Y_DROPOUT]) {
        m_d_rate = config[Y_DROPOUT].as<float>();
        m_d_dropout = m_d_rate > 0.0f;
        m_d_dropoutdist = std::bernoulli_distribution(1.0f-m_d_rate);
    }

    m_optimizer.Define(optimizerConfig);

    if (config[Y_SKIPCONN]) {
        m_s_skipconn = true;
        m_s_idx = config[Y_SKIPCONN].as<size_t>();
        
        m_s_base = inodes;
        m_s_skip = (*m_layers)[m_s_idx].nodes;

        inodes = m_s_base + m_s_skip;
    }
}
void Layer::Initialize() {
    // initialize member data
    gen = std::mt19937(SEED+m_layer_idx);

    AssignLayerSize();
    AssignFunctionPointers();
}

/// @brief Initializes batchsize and testsize
void Layer::InitializeSizes(size_t bn, size_t tn) {
    layer_batch_bytes = 0;
    layer_test_bytes = 0;

    switch (type) {
        case LayerType::input:
            m_a_bytes = MathUtils::RoundTo(32, nodes*bn*sizeof(float));
            m_ta_bytes = MathUtils::RoundTo(32, nodes*tn*sizeof(float));

            layer_batch_bytes = m_a_bytes;
            layer_test_bytes = m_ta_bytes;            
            break;
        case LayerType::output: case LayerType::hidden: 
            SetBasicBatchTestBytes(bn, tn);
            break;
    }
}

/// @brief Sets the internal pointers for network data, batch data, and test data, bn and tn MUST match previously passed
void Layer::InitializePointers(char* data, char* batchdata, char* testdata, size_t bn, size_t tn) {
    // initialize all to nullptr
    m_w = nullptr;
    m_b = nullptr;
    m_z = nullptr;
    m_a = nullptr;
    m_dt = nullptr;
    m_dw = nullptr;
    m_db = nullptr;
    m_d_dpmask = nullptr;
    m_tz = nullptr;
    m_ta = nullptr;

    m_net = data;
    m_batch = batchdata;
    m_test = testdata;

    // assign data pointers
    size_t offset = 0;
    switch (type) {
        case LayerType::input:
            break;
        case LayerType::hidden: case LayerType::output:
            m_w = (float*)(data+offset);
            offset += MathUtils::RoundTo(32, wsize*sizeof(float));

            m_b = (float*)(data+offset);
            offset += MathUtils::RoundTo(32, bsize*sizeof(float));
            break;
    }

    // assign batch data pointers
    offset = 0;
    switch (type) {
        case LayerType::input:
            m_a = (float*)batchdata;
            break;
        case LayerType::hidden: case LayerType::output:
            AssignBasicBatchPtrs(batchdata, bn);
            break;
    }

    // assign test data pointers
    offset = 0;
    switch (type) {
        case LayerType::input:
            m_ta = (float*)testdata;
            break;
        case LayerType::hidden: case LayerType::output:
            size_t output_size = nodes*tn*sizeof(float);

            m_tz = (float*)(testdata+offset);
            offset += MathUtils::RoundTo(32, output_size);

            m_ta = (float*)(testdata+offset);
            offset += MathUtils::RoundTo(32, output_size);
            break;
    }
}

/// @brief sets the next weight parameter that's used in backprop
void Layer::InitializeSpecialPointers(float* nextweight) {
    // assign special pointers used in backprop
    m_nw = nextweight;
}

/// @brief initializes the layers weights based on init type
void Layer::InitializeWeights(float* data, WeightInitialization init) {
    if (wsize == 0 && bsize == 0) { return; }

    float lowerRand;
    float upperRand;
    size_t idx = 0;

    // zero out biases
    memset(&data[wsize], 0, bsize*sizeof(float));

    if (init == WeightInitialization::he) {
        
        lowerRand = 0.0f;
        upperRand = std::sqrt(2.0f/nodes);

        std::normal_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen);
        }
    } else if (init == WeightInitialization::normalize) {
        
        lowerRand = -0.5f;
        upperRand = 0.5f;

        std::uniform_real_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen) * std::sqrt(1.0f/nodes);
        }
    } else if (init == WeightInitialization::xavier) {
        
        lowerRand = (-1.0f/std::sqrt(nodes));
        upperRand = 1.0f/std::sqrt(nodes);

        std::uniform_real_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen);
        }
    } else {
        // no weight initialization has been set, zero the weights
        memset(data, 0, wsize*sizeof(float));
    }
}
