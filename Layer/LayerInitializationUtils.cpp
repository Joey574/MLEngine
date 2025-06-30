#include "Layer.hpp"

void Layer::AssignLayerSize() {
    layer_bytes = 0;
    wsize = 0;
    bsize = 0;
    params = 0;

    // set network size
    switch (type) {
        case LayerType::input:
            break;
        case LayerType::hidden: case LayerType::output:
            wsize = inodes*nodes;
            bsize = nodes;
            params = wsize+bsize;

            // size for weights and biases
            m_w_bytes = MathUtils::RoundTo(32, wsize*sizeof(float));
            m_b_bytes = MathUtils::RoundTo(32, bsize*sizeof(float));

            layer_bytes += m_w_bytes + m_b_bytes;
            break;
        case LayerType::conv2D: case LayerType::conv3D:
            break;
    }
}

void Layer::AssignBasicBatchPtrs(char* batchdata, size_t bn) {
    size_t offset = 0;

    m_z = (float*)(batchdata+offset);
    offset += m_z_bytes;

    m_a = (float*)(batchdata+offset);
    offset += m_a_bytes;

    m_dt = (float*)(batchdata+offset);
    offset += m_dt_bytes;

    m_dw = (float*)(batchdata+offset);
    offset += m_dw_bytes;

    m_db = (float*)(batchdata+offset);
    offset += m_db_bytes;

    if (m_d_dropout) {
        m_d_dpmask = (uint8_t*)(batchdata+offset);
        offset += m_d_dpmask_bytes;
    }

    // size in optimizer params
    m_optimizer.Initialize(m_dw, m_db, (batchdata+offset), wsize, bsize);
    offset += m_o_bytes;
}

void Layer::SetBasicBatchTestBytes(size_t bn, size_t tn) {
 
    m_z_bytes = MathUtils::RoundTo(32, nodes*bn*sizeof(float));
    m_a_bytes = MathUtils::RoundTo(32, nodes*bn*sizeof(float));

    m_tz_bytes = MathUtils::RoundTo(32, nodes*tn*sizeof(float));
    m_ta_bytes = MathUtils::RoundTo(32, nodes*tn*sizeof(float));

    m_dt_bytes = MathUtils::RoundTo(32, nodes*bn*sizeof(float));
    m_dw_bytes = m_w_bytes;
    m_db_bytes = m_b_bytes;

    if (m_d_dropout) {
        // bit packed
        m_d_dpmask_bytes = MathUtils::RoundTo(32, (nodes*bn+7)/8);
    }

    // size for optimizer
    m_o_bytes = m_optimizer.Size(wsize, bsize);

    // size in all the things
    layer_batch_bytes = m_z_bytes + m_a_bytes + m_dt_bytes + m_dw_bytes + m_db_bytes + 
        m_d_dpmask_bytes + m_o_bytes;

    // size for total and activation
    layer_test_bytes = m_tz_bytes + m_ta_bytes;

    // ensure total bytes are all aligned to 32
    assert(layer_batch_bytes%32==0);
    assert(layer_test_bytes%32==0);
}

void Layer::AssignFunctionPointers() {

    // forwards
    if (type == LayerType::input) {
        executeForwardTrain = &Layer::InputForward<true>;
        executeForwardInfer = &Layer::InputForward<false>;
    } else {
        if (m_d_dropout) {
            if (m_s_skipconn) {
                executeForwardTrain = &Layer::BasicForward<true, true, true>;
                executeForwardInfer = &Layer::BasicForward<false, true, true>;
            } else {
                executeForwardTrain = &Layer::BasicForward<true, true, false>;
                executeForwardInfer = &Layer::BasicForward<false, true, false>;
            }
        } else {
            if (m_s_skipconn) {
                executeForwardTrain = &Layer::BasicForward<true, false, true>;
                executeForwardInfer = &Layer::BasicForward<false, false, true>;
            } else {
                executeForwardTrain = &Layer::BasicForward<true, false, false>;
                executeForwardInfer = &Layer::BasicForward<false, false, false>;
            }
        }
    }

    // backwards
    if (type == LayerType::input) {
        if (m_d_dropout) {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::input, true, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::input, true, false>;
            }
        } else {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::input, false, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::input, false, false>;
            }
        }
    } else if (type == LayerType::hidden) {
        if (m_d_dropout) {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::hidden, true, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::hidden, true, false>;
            }
        } else {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::hidden, false, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::hidden, false, false>;
            }
        }
    } else if (type == LayerType::output) {
        if (m_d_dropout) {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::output, true, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::output, true, false>;
            }
        } else {
            if (m_s_skipconn) {
                executeBackward = &Layer::BasicBackward<LayerType::output, false, true>;
            } else {
                executeBackward = &Layer::BasicBackward<LayerType::output, false, false>;
            }
        }
    }
}