#include "Optimizer.hpp"

template <Optimizer::Regularization reg>
void Optimizer::SGD(float* __restrict w, float* __restrict b, size_t wsize, size_t bsize, size_t n) {
    if constexpr (reg == Regularization::l1) {
        SGDL1Compute(w, m_s_dw, wsize, m_lr, n, m_reg_lambda);
        SGDL1Compute(b, m_s_db, bsize, m_lr, n, m_reg_lambda);
    } else if constexpr (reg == Regularization::l2) {
        SGDL2Compute(w, m_s_dw, wsize, m_lr, n, m_reg_lambda);
        SGDL2Compute(b, m_s_db, bsize, m_lr, n, m_reg_lambda);
    } else {
        SGDCompute(w, m_s_dw, wsize, m_lr, n);
        SGDCompute(b, m_s_db, bsize, m_lr, n);
    }
}

template <Optimizer::Regularization reg>
void Optimizer::MomentumSGD(float* __restrict w, float* __restrict b, size_t wsize, size_t bsize, size_t n) {
    if constexpr (reg == Regularization::l1) {
        MomentumSGDL1Compute(w, m_m_vw, m_s_dw, wsize, m_lr, n, m_reg_lambda, m_m_coef);
        MomentumSGDL1Compute(b, m_m_vb, m_s_db, bsize, m_lr, n, m_reg_lambda, m_m_coef);
    } else if constexpr (reg == Regularization::l2) {
        MomentumSGDL2Compute(w, m_m_vw, m_s_dw, wsize, m_lr, n, m_reg_lambda, m_m_coef);
        MomentumSGDL2Compute(b, m_m_vb, m_s_db, bsize, m_lr, n, m_reg_lambda, m_m_coef);
    } else {
        MomentumSGDCompute(w, m_m_vw, m_s_dw, wsize, m_lr, n, m_m_coef);
        MomentumSGDCompute(b, m_m_vb, m_s_db, bsize, m_lr, n, m_m_coef);
    }
}

void Optimizer::RMSProp(float* __restrict w, float* __restrict b, size_t wsize, size_t bsize, size_t n) {
    RMSPropCompute(w, m_r_gw, m_s_dw, wsize, m_lr, n, m_r_decay, m_r_epsl);
    RMSPropCompute(b, m_r_gb, m_s_db, bsize, m_lr, n, m_r_decay, m_r_epsl);
}

void Optimizer::Adam(float* __restrict w, float* __restrict b, size_t wsize, size_t bsize, size_t n) {
    // TODO figure out how to save m_a_t for model reloading
    
    AdamCompute(w, m_a_wm, m_a_wv, m_s_dw, wsize, m_lr, n, m_a_b1, m_a_b2, m_a_epsl, m_a_t);
    AdamCompute(b, m_a_bm, m_a_bv, m_s_db, bsize, m_lr, n, m_a_b1, m_a_b2, m_a_epsl, m_a_t);

    m_a_t++;
}
