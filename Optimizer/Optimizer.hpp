#pragma once

struct Optimizer {
public:
    using UpdateFn = void (Optimizer::*)(float*, float*, size_t, size_t, size_t);

    enum class Regularization {
        none, l1, l2
    };

    enum class Update {
        none, sgd, momentumsgd, rmsprop, adam
    };

    UpdateFn update;
    
    void Define(YAML::Node config);
    void Initialize(float* dw, float* db, char* data, size_t wsize, size_t bsize);
    size_t Size(size_t wsize, size_t bsize);

    static std::string ParseRegName(Regularization reg);
    static std::string ParseUpdName(Update upd);
    static Regularization ParseRegType(const std::string& reg);
    static Update ParseUpdType(const std::string& upd);

private:

    void AssignPtr();
    static size_t RoundTo(size_t alignment, size_t n);

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

    static void SGDCompute(float* p, const float* d, size_t size, float lr, size_t n);
    static void SGDL1Compute(float* p, const float* d, size_t size, float lr, size_t n, float lambda);
    static void SGDL2Compute(float* p, const float* d, size_t size, float lr, size_t n, float lambda);

    static void MomentumSGDCompute(float* p, float* v, const float* d, size_t size, float lr, size_t n, float coef);
    static void MomentumSGDL1Compute(float* p, float* v, const float* d, size_t size, float lr, size_t n, float lambda, float coef);
    static void MomentumSGDL2Compute(float* p, float* v, const float* d, size_t size, float lr, size_t n, float lambda, float coef);
};
