#include "Layer.hpp"

/// @brief Initializes basic layer data, sets layer size, does not touch batchsize or testsize
void Layer::Initialize(LayerType type, size_t in, size_t n, size_t nn, Activation actv, LossMetric lm) {
    this->type = type;
    
    inodes = in;
    nodes = n;
    nenodes = nn;

    activation = actv;
    lossmetric = lm;

    // set network size
    switch (type) {
        case LayerType::input:
            layer_size = 0;
            wsize = 0;
            break;
        default:
            layer_size = in*n + n;
            wsize = in*n;
    }
}

/// @brief Initializes batchsize and testsize
void Layer::InitializeSizes(size_t bn, size_t tn) {
    switch (type) {
        case LayerType::input:
            layer_batch_size = 0;
            layer_test_size = 0;
            break;
        default:
            layer_batch_size = (3*nodes*bn)+wsize+nodes;
            layer_test_size = 2*nodes*tn;
    }
}

/// @brief Sets the internal pointers for network data, batch data, and test data, bn and tn MUST match previously passed
void Layer::InitializePointers(float* data, float* batchdata, float* testdata, size_t bn, size_t tn) {

    // assign data pointers
    switch (type) {
        case LayerType::input:
            m_w = nullptr;
            m_b = nullptr;
            break;
        default:
            m_w = data;
            m_b = &data[wsize];
    }

    // assign batch data pointers
    switch (type) {
        case LayerType::input:
            m_z = nullptr;
            m_a = nullptr;
            m_dt = nullptr;
            m_dw = nullptr;
            m_db = nullptr;
            break;
        default:
            m_z = batchdata;
            m_a = &batchdata[nodes*bn];
            m_dt = &m_a[nodes*bn];
            m_dw = &m_dt[nodes*bn];
            m_db = &m_dw[wsize];
    }

    // assign test data pointers
    switch (type) {
        case LayerType::input:
            m_tz = nullptr;
            m_ta = nullptr;
            break;
        default:
            m_tz = testdata;
            m_ta = &testdata[nodes*tn];
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

    float lowerRand;
    float upperRand;
    size_t idx = 0;
    
    std::default_random_engine gen(seed);

    // zero out biases
    memset(&data[wsize], 0, nodes*sizeof(float));

    switch (init) {
        case WeightInitialization::he:
        {
            lowerRand = 0.0f;
            upperRand = std::sqrt(2.0f/nodes);

            std::normal_distribution<float> dist(lowerRand, upperRand);
            for (size_t i = 0; i < wsize; i++) {
                data[i] = dist(gen);
            }
        }
            break;
        case WeightInitialization::normalize:
        {
            lowerRand = -0.5f;
            upperRand = 0.5f;

            std::uniform_real_distribution<float> dist(lowerRand, upperRand);
            for (size_t i = 0; i < wsize; i++) {
                data[i] = dist(gen) * std::sqrt(1.0f/nodes);
            }
        }
            break;
        case WeightInitialization::xavier:
        {
            lowerRand = (-1.0f/std::sqrt(nodes));
            upperRand = 1.0f/std::sqrt(nodes);

            std::uniform_real_distribution<float> dist(lowerRand, upperRand);

            for (size_t i = 0; i < wsize; i++) {
                data[i] = dist(gen);
            }
        }
            break;
        default:
            // no weight initialization has been set, zero the weights
            memset(data, 0, wsize*sizeof(float));
    }
}
