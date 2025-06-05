#include "Layer.hpp"

/// @brief Initializes basic layer data, sets layer size, does not touch batchsize or testsize
void Layer::Initialize(LayerType type, size_t in, size_t n, size_t nn, Activation actv, LossMetric lm, float dropout) {
    this->type = type;
    this-> m_dropout = dropout;
    
    inodes = in;
    nodes = n;
    nenodes = nn;

    activation = actv;
    lossmetric = lm;

    // set network size
    switch (type) {
        case LayerType::input:
            layer_bytes = 0;
            wsize = 0;
            bsize = 0;
            params = 0;
            break;
        default:
            layer_bytes = (in*n+n)*sizeof(float);
            wsize = in*n;
            bsize = n;
            params = wsize+bsize;
    }

    if (dropout > 0.0f) {
        executeForward = &Layer::DropoutForward;
        executeBackward = &Layer::DropoutBackward;
    } else {
        executeForward = &Layer::BasicForward;
        executeBackward = &Layer::BasicBackward;        
    }
}

/// @brief Initializes batchsize and testsize
void Layer::InitializeSizes(size_t bn, size_t tn) {
    switch (type) {
        case LayerType::input:
            layer_batch_bytes = 0;
            layer_test_bytes = 0;
            break;
        case LayerType::hidden:
            layer_batch_bytes = ((3*nodes*bn)+wsize+nodes)*sizeof(float);
            layer_test_bytes = (2*nodes*tn)*sizeof(float);

            if (m_dropout > 0.0f) {
                // packed to uint8_t, in future should be bit packed
                layer_batch_bytes += nodes*bn;
            }
            break;
        case LayerType::output:
            layer_batch_bytes = ((3*nodes*bn)+wsize+nodes)*sizeof(float);
            layer_test_bytes = (2*nodes*tn)*sizeof(float);
            break;
    }
}

/// @brief Sets the internal pointers for network data, batch data, and test data, bn and tn MUST match previously passed
void Layer::InitializePointers(char* data, char* batchdata, char* testdata, size_t bn, size_t tn) {

    // assign data pointers
    switch (type) {
        case LayerType::input:
            m_w = nullptr;
            m_b = nullptr;
            break;
        default:
            m_w = (float*)data;
            m_b = &m_w[wsize];
    }

    // assign batch data pointers
    switch (type) {
        case LayerType::input:
            m_z = nullptr;
            m_a = nullptr;
            m_dt = nullptr;
            m_dw = nullptr;
            m_db = nullptr;
            m_dpmask = nullptr;
            break;
        case LayerType::hidden:
            m_z = (float*)batchdata;
            m_a = &m_z[nodes*bn];
            m_dt = &m_a[nodes*bn];
            m_dw = &m_dt[nodes*bn];
            m_db = &m_dw[wsize];

            if (m_dropout > 0.0f) {
                m_dpmask = (uint8_t*)&m_db[nodes];
            } else {
                m_dpmask = nullptr;
            }
            break;
        case LayerType::output:
            m_z = (float*)batchdata;
            m_a = &m_z[nodes*bn];
            m_dt = &m_a[nodes*bn];
            m_dw = &m_dt[nodes*bn];
            m_db = &m_dw[wsize];
            m_dpmask = nullptr;
            break;
    }

    // assign test data pointers
    switch (type) {
        case LayerType::input:
            m_tz = nullptr;
            m_ta = nullptr;
            break;
        default:
            m_tz = (float*)testdata;
            m_ta = &m_tz[nodes*tn];
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
