#pragma once

struct Optimizer {
public:
    using UpdateFunc = void (*)(float*, float*, const float*, const float*, float, size_t, size_t);

    enum class Regularization {
        none, l1, l2
    };

    enum class Update {
        none, sgd, momentum_sgd, rms_prop, adam
    };

    UpdateFunc update;


private:

    // basic data
    Update m_update;
    float m_lr;

    // sgd data
    float* m_s_dw;
    float* m_s_db;

    // momentum data
    float m_m_coef;
    float* m_m_vw;
    float* m_m_vb;

    // rmsprop data

    // adam data

    // regularization techniques
    Regularization m_reg;
    float m_reg_lambda;

    template <Regularization reg> void SGD(float* w, float* b, size_t wsize, size_t bsize, size_t n);
    template <Regularization reg> void MomentumSGD(float* w, float* b, size_t wsize, size_t bsize, size_t n);
    void RMSProp();
    void Adam();

    static void SGDCompute(float* p, const float* d, size_t n, float lr);
    static void SGDL1Compute(float* p, const float* d, size_t n, float lr, float lambda);
    static void SGDL2Compute(float* p, const float* d, size_t n, float lr, float lambda);
};
