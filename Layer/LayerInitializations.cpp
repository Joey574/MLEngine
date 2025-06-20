#include "Layer.hpp"

void Layer::Define(std::vector<Layer>& layers, size_t idx, YAML::Node config, size_t in, size_t nn) {
    this->inodes = in;
    this->nenodes = nn;

    m_layers = &layers;
    m_layer_idx = idx;

    type = ParseType(config[Y_LAYERTYPE].as<std::string>());
    nodes = config[Y_NODES].as<size_t>();

    if (config[Y_ACTIVATION]) {
        activation.AssignPointers(Activation::ParseSingleType(config[Y_ACTIVATION].as<std::string>()));
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

    if (config[Y_REGULARIZATION]) {
        std::string reg = config[Y_REGULARIZATION].as<std::string>();

        if (reg == "l1") {
            m_l1 = true;
            m_l1_lambda = config[Y_L1_LAMBDA].as<float>(0.0001f);
        } else if (reg == "l2") {
            m_l2 = true;
            m_l2_lambda = config[Y_L2_LAMBDA].as<float>(0.0001f);
        }
    }

    if (config[Y_MOMENTUM]) {
        m_m_momentum = true;
        m_m_coefficient = config[Y_MOMENTUM].as<float>();
    }

    if (config[Y_SKIPCONN]) {
        m_s_skipconn = true;
        m_s_idx = config[Y_SKIPCONN].as<size_t>();
        inodes += (*m_layers)[m_s_idx].nodes;
    }
}
void Layer::Initialize() {
    // initialize member data
    std::random_device rd;
    gen = std::mt19937(rd());

    layer_bytes = 0;
    wsize = 0;
    bsize = 0;
    params = 0;

    AssignLayerSize();
    AssignFunctionPointers();
}

/// @brief Initializes batchsize and testsize
void Layer::InitializeSizes(size_t bn, size_t tn) {
    layer_batch_bytes = 0;
    layer_test_bytes = 0;

    switch (type) {
        case LayerType::input:
            // space for activation
            layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));

            // space for test activation
            layer_test_bytes += RoundTo(32, nodes*tn*sizeof(float));
            
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
    m_m_vw = nullptr;
    m_m_vb = nullptr;

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
            offset += RoundTo(32, wsize*sizeof(float));

            m_b = (float*)(data+offset);
            offset += RoundTo(32, bsize*sizeof(float));
            break;
    }

    // assign batch data pointers
    offset = 0;
    switch (type) {
        case LayerType::input:
            m_a = m_z = (float*)batchdata;
            break;
        case LayerType::hidden: case LayerType::output:
            AssignBasicBatchPtrs(batchdata, bn);
            break;
    }

    // assign test data pointers
    offset = 0;
    switch (type) {
        case LayerType::input:
            m_ta = m_tz = (float*)testdata;
            break;
        case LayerType::hidden: case LayerType::output:
            size_t output_size = nodes*tn*sizeof(float);

            if (m_layer_idx != m_layers->size()-1 && (*m_layers)[m_layer_idx+1].m_s_skipconn) {
                size_t skip_idx = (*m_layers)[m_layer_idx+1].m_s_idx;
                size_t layer_out = (*m_layers)[skip_idx].nodes;

                output_size += layer_out*tn*sizeof(float);
            }

            m_tz = (float*)(testdata+offset);
            offset += RoundTo(32, output_size);

            m_ta = (float*)(testdata+offset);
            offset += RoundTo(32, output_size);
            break;
    }
}

/// @brief sets the next weight parameter that's used in backprop
void Layer::InitializeSpecialPointers(float* nextweight) {
    // assign special pointers used in backprop
    m_nw = nextweight;
}

/// @brief initializes the layers weights based on init type
void Layer::InitializeWeights(float* data, WeightInitialization init, uint64_t seed) {
    if (type == LayerType::input) { return; }

    switch (init) {
        case WeightInitialization::he:
            SetWeights<WeightInitialization::he>(data, seed);
            break;
        case WeightInitialization::normalize:
            SetWeights<WeightInitialization::normalize>(data, seed);
            break;
        case WeightInitialization::xavier:
            SetWeights<WeightInitialization::xavier>(data, seed);
            break;
        default:
            SetWeights<WeightInitialization::none>(data, seed);
    }
}
