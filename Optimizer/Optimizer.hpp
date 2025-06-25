#pragma once

struct Optimizer {
public:
    using UpdateFunc = void (*)(float*, float*, const float*, const float*, float, size_t, size_t);
    using SGDComputeFunc = void (*)(float*, const float*, size_t, float);
    using SGDRegComputeFunc = void (*)(float*, const float*, size_t, float, float);

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
    static SGDComputeFunc SGDCompute;
    static SGDRegComputeFunc SGDL1Compute;
    static SGDRegComputeFunc SGDL2Compute;


    // momentum data
    float m_m_coef;
    float* m_m_vw;
    float* m_m_vb;

    // rmsprop data

    // adam data

    // regularization techniques
    bool m_reg_l1;
    bool m_reg_l2;
    float m_reg_lambda;

    template <Regularization reg> void SGD(float* w, float* b, size_t wsize, size_t bsize, size_t n);
    template <Regularization reg> void MomentumSGD(float* w, float* b, size_t wsize, size_t bsize, size_t n);
    void RMSProp();
    void Adam();

    static void SGDComputeScalar(float* p, const float* d, size_t n, float lr);
    static void SGDComputeAVX2(float* p, const float* d, size_t n, float lr);
    static void SGDComputeAVX512(float* p, const float* d, size_t n, float lr);

    static void SGDL1ComputeScalar(float* p, const float* d, size_t n, float lr, float lambda);
    static void SGDL1ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda);
    static void SGDL1ComputeAVX512(float* p, const float* d, size_t n, float lr, float lambda);

    static void SGDL2ComputeScalar(float* p, const float* d, size_t n, float lr, float lambda);
    static void SGDL2ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda);
    static void SGDL2ComputeAVX512(float* p, const float* d, size_t n, float lr, float lambda);
};
