#include "Layer.hpp"

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
    } else {
        m_d_dpmask = nullptr;
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

    // space for velocity data
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
