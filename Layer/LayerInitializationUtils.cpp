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
            m_w_bytes = RoundTo(32, wsize*sizeof(float));
            m_b_bytes = RoundTo(32, bsize*sizeof(float));

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

    if (m_m_momentum) {
        m_m_vw = (float*)(batchdata+offset);
        offset += m_m_vw_bytes;

        m_m_vb = (float*)(batchdata+offset);
        offset += m_m_vb_bytes;
    }
}

void Layer::SetBasicBatchTestBytes(size_t bn, size_t tn) {
 
    m_z_bytes = RoundTo(32, nodes*bn*sizeof(float));
    m_a_bytes = RoundTo(32, nodes*bn*sizeof(float));

    m_tz_bytes = RoundTo(32, nodes*tn*sizeof(float));
    m_ta_bytes = RoundTo(32, nodes*tn*sizeof(float));

    m_dt_bytes = RoundTo(32, nodes*bn*sizeof(float));
    m_dw_bytes = m_w_bytes;
    m_db_bytes = m_b_bytes;

    if (m_d_dropout) {
        // bit packed
        m_d_dpmask_bytes = RoundTo(32, (nodes+(bn-1))*bn/8);
    }

    if (m_m_momentum) {
        m_m_vw_bytes = m_w_bytes;
        m_m_vb_bytes = m_b_bytes;
    }

    // size in all the things
    layer_batch_bytes = m_z_bytes + m_a_bytes + m_dt_bytes + m_dw_bytes + m_db_bytes + 
        m_d_dpmask_bytes + m_m_vw_bytes + m_m_vb_bytes;

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
    if (m_d_dropout) {
        if (m_s_skipconn) {
            executeBackward = &Layer::BasicBackward<true, true>;
        } else {
            executeBackward = &Layer::BasicBackward<true, false>;
        }
    } else {
        if (m_s_skipconn) {
            executeBackward = &Layer::BasicBackward<false, true>;
        } else {
            executeBackward = &Layer::BasicBackward<false, false>;
        }
    }

    // updates
    if (m_m_momentum) {
        if (m_l1) {
            updateLayer = &Layer::MomentumUpdate<true, false>;
        } else if (m_l2) {
            updateLayer = &Layer::MomentumUpdate<false, true>;
        } else {
            updateLayer = &Layer::MomentumUpdate<false, false>;
        }
    } else {
        if (m_l1) {
            updateLayer = &Layer::BasicUpdate<true, false>;
        } else if (m_l2) {
            updateLayer = &Layer::BasicUpdate<false, true>;
        } else {
            updateLayer = &Layer::BasicUpdate<false, false>;
        }
    }
}