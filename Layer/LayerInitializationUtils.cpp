#include "Layer.hpp"

void Layer::AssignLayerSize() {
    // set network size
    switch (type) {
        case LayerType::input:
            break;
        case LayerType::hidden: case LayerType::output:
            wsize = inodes*nodes;
            bsize = nodes;
            params = wsize+bsize;

            // size for weights and biases
            layer_bytes += RoundTo(32, wsize*sizeof(float));
            layer_bytes += RoundTo(32, bsize*sizeof(float));
            break;
        case LayerType::conv2D: case LayerType::conv3D:
            break;
    }
}

void Layer::AssignBasicBatchPtrs(char* batchdata, size_t bn) {
    size_t offset = 0;
    size_t output_size = nodes*bn*sizeof(float);

    if (m_layer_idx != (*m_layers).size()-1 && (*m_layers)[m_layer_idx+1].m_s_skipconn) {
        size_t skip_idx = (*m_layers)[m_layer_idx+1].m_s_idx;
        size_t layer_out = (*m_layers)[skip_idx].nodes;

        output_size += layer_out*bn*sizeof(float);
    }

    m_z = (float*)(batchdata+offset);
    offset += RoundTo(32, output_size);

    m_a = (float*)(batchdata+offset);
    offset += RoundTo(32, output_size);

    m_dt = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

    m_dw = (float*)(batchdata+offset);
    offset += RoundTo(32, wsize*sizeof(float));

    m_db = (float*)(batchdata+offset);
    offset += RoundTo(32, bsize*sizeof(float));

    if (m_d_dropout) {
        m_d_dpmask = (uint8_t*)(batchdata+offset);
        offset += RoundTo(32, nodes*bn);
    }

    if (m_m_momentum) {
        m_m_vw = (float*)(batchdata+offset);
        offset += RoundTo(32, wsize*sizeof(float));

        m_m_vb = (float*)(batchdata+offset);
        offset += RoundTo(32, bsize*sizeof(float));
    }
}

void Layer::SetBasicBatchTestBytes(size_t bn, size_t tn) {
    size_t batch_output_bytes = 0;
    size_t test_output_bytes = 0;

    // space for total, activation, and dt
    batch_output_bytes += nodes*bn*sizeof(float);
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));

    // space for weights and biases
    test_output_bytes += nodes*tn*sizeof(float);

    // space for total and activation test stuff
    layer_test_bytes += RoundTo(32, nodes*tn*sizeof(float));
    layer_test_bytes += RoundTo(32, nodes*tn*sizeof(float));

    if (m_d_dropout) {
        // bit packed
        layer_batch_bytes += RoundTo(32, (nodes+(bn-1))*bn/8);
    }

    if (m_m_momentum) {
        layer_batch_bytes += RoundTo(32, wsize*sizeof(float));
        layer_batch_bytes += RoundTo(32, bsize*sizeof(float));
    }

    if (m_layer_idx != (*m_layers).size()-1 && (*m_layers)[m_layer_idx+1].m_s_skipconn) {
        size_t skip_idx = (*m_layers)[m_layer_idx+1].m_s_idx;
        size_t layer_out = (*m_layers)[skip_idx].nodes;

        // size relevant skipconn output into self output buffer
        batch_output_bytes += layer_out*bn*sizeof(float);
        test_output_bytes += layer_out*tn*sizeof(float);
    }

    layer_batch_bytes += 2*RoundTo(32, batch_output_bytes);
    layer_test_bytes += 2*RoundTo(32, test_output_bytes);

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
        executeBackward = &Layer::BasicBackward<true>;
    } else {
        executeBackward = &Layer::BasicBackward<false>;
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