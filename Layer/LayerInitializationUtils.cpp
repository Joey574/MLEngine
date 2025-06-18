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
        case LayerType::convolutional:
            break;
    }
}

void Layer::AssignHiddenBatchPtrs(char* batchdata, size_t bn) {
    size_t offset = 0;

    m_z = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

    m_a = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

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
void Layer::AssignOutputBatchPtrs(char* batchdata, size_t bn) {
    size_t offset = 0;   

    m_z = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

    m_a = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

    m_dt = (float*)(batchdata+offset);
    offset += RoundTo(32, nodes*bn*sizeof(float));

    m_dw = (float*)(batchdata+offset);
    offset += RoundTo(32, wsize*sizeof(float));

    m_db = (float*)(batchdata+offset);
    offset += RoundTo(32, bsize*sizeof(float));

    if (m_m_momentum) {
        m_m_vw = (float*)(batchdata+offset);
        offset += RoundTo(32, wsize*sizeof(float));

        m_m_vb = (float*)(batchdata+offset);
        offset += RoundTo(32, bsize*sizeof(float));
    }
}

void Layer::SetHiddenBatchTestBytes(size_t bn, size_t tn) {
    // space for total, activation, and dt
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));

    // space for weights and biases
    layer_batch_bytes += RoundTo(32, wsize*sizeof(float));
    layer_batch_bytes += RoundTo(32, bsize*sizeof(float));

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

    // ensure total bytes are all aligned to 32
    assert(layer_batch_bytes%32==0);
    assert(layer_test_bytes%32==0);
}
void Layer::SetOutputBatchTestBytes(size_t bn, size_t tn) {
    // space for total, activation, and dt
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));
    layer_batch_bytes += RoundTo(32, nodes*bn*sizeof(float));

    // space for weights and biases
    layer_batch_bytes += RoundTo(32, wsize*sizeof(float));
    layer_batch_bytes += RoundTo(32, bsize*sizeof(float));

    // space for total and activation test stuff
    layer_test_bytes += RoundTo(32, nodes*tn*sizeof(float));
    layer_test_bytes += RoundTo(32, nodes*tn*sizeof(float));

    // space for velocity data
    if (m_m_momentum) {
        layer_batch_bytes += RoundTo(32, wsize*sizeof(float));
        layer_batch_bytes += RoundTo(32, bsize*sizeof(float));
    }

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
            executeForwardTrain = &Layer::BasicForward<true, true>;
            executeForwardInfer = &Layer::BasicForward<false, true>;
        } else {
            executeForwardTrain = &Layer::BasicForward<true, false>;
            executeForwardInfer = &Layer::BasicForward<false, false>;
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