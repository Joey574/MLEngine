#include "Optimizer.hpp"

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
