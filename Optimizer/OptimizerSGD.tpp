#include "Optimizer.hpp"

template <Optimizer::Regularization reg>
void Optimizer::SGD(float* __restrict w, float* __restrict b, size_t wsize, size_t bsize, size_t n) {
    if constexpr (reg == Regularization::l1) {
        SGDL1Compute(w, m_s_dw, wsize, m_lr, m_reg_lambda);
        SGDL1Compute(b, m_s_db, bsize, m_lr, m_reg_lambda);
    } else if constexpr (reg == Regularization::l2) {
        SGDL2Compute(w, m_s_dw, wsize, m_lr, m_reg_lambda);
        SGDL2Compute(b, m_s_db, bsize, m_lr, m_reg_lambda);
    } else {
        SGDCompute(w, m_s_dw, wsize, m_lr);
        SGDCompute(b, m_s_db, bsize, m_lr);
    }
}
