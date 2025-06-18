#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"

struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output, convolutional
    };
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    Layer() { memset(this, 0, sizeof(Layer)); }

    static std::string ParseName(LayerType type);
    static LayerType ParseType(const std::string& type);

    void Initialize(LayerType type, size_t in, size_t n, size_t nn, Activation actv, LossMetric lm, float dropout, bool momentum);
    void Initialize(std::vector<Layer>& layers, size_t idx, YAML::Node config, size_t in, size_t nn);
    void InitializeSizes(size_t bn, size_t tn);
    void InitializePointers(char* data, char* batchdata, char* testdata, size_t bn, size_t tn);
    void InitializeSpecialPointers(float* nextweight);

    void InitializeWeights(float* data, WeightInitialization init, uint64_t seed);

    template<bool training>float* Output() { return training ? m_a : m_ta; }
    float* Truth() { return m_dt; }
    float* Weights() { return m_w; }

    template <bool training> void forward(
        float* __restrict x,
        size_t n
    );

    void backward(
        const float* __restrict truth,
        const float* __restrict input,
        size_t n
    );
    
    void update(
        float lr,
        size_t n
    );

    
    LayerType type;

    size_t nodes;
    size_t inodes;
    size_t nenodes;

    LossMetric lossmetric;
    Activation activation;

    size_t layer_bytes;
    size_t layer_batch_bytes;
    size_t layer_test_bytes;

    size_t params;

private:

    void (Layer::*executeForwardTrain)(float* , size_t);
    void (Layer::*executeForwardInfer)(float*, size_t);
    void (Layer::*executeBackward)(const float*, const float*, size_t);
    void (Layer::*updateLayer)(float, size_t);

    // template methods
    template <WeightInitialization> void SetWeights(float* data, uint64_t seed);

    // forward prop methods
    template <bool training, bool dropout> void BasicForward (float* __restrict input, size_t n);
    template <bool training> void ConvolutionalForward(float* __restrict input, size_t n);

    // backprop methods
    template <bool dropout> void BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n);

    // update methods
    template <bool l1, bool l2> void BasicUpdate(float lr, size_t n);
    template <bool l1, bool l2> void MomentumUpdate(float lr, size_t n);


    // forward prop utils
    template<bool training> void InputForward(float* __restrict input, size_t n);
    void ApplyDropoutFP(size_t n);

    // backprop utils
    void ComputeDT(const float* __restrict truth, size_t n);
    void ComputeDN(const float* __restrict input, size_t n);
    void ApplyDropoutBP(size_t n);

    // upadte utils
    void ApplyBasicUpdate(const float* __restrict d, float* __restrict p, const __m256 _factor);
    void ApplyL1Update(const float* __restrict d, float* __restrict p, const __m256 _factor, const __m256 _coef);
    void ApplyL2Update(const float* __restrict d, float* __restrict p, const __m256 _factor, const __m256 _coef);
    float ApplyBasicUpdate(const float d, const float p, const float factor);
    float ApplyL1Update(const float d, const float p, const float factor, const float coef);
    float ApplyL2Update(const float d, const float p, const float factor, const float coef);


    // private initialization utils
    void AssignLayerSize();
    void AssignHiddenBatchPtrs(char* batchdata, size_t bn);
    void AssignOutputBatchPtrs(char* batchdata, size_t bn);
    void AssignFunctionPointers();

    void SetHiddenBatchTestBytes(size_t bn, size_t tn);
    void SetOutputBatchTestBytes(size_t bn, size_t tn);
    size_t RoundTo(size_t alignment, size_t n);


    // network data
    float* m_w;
    float* m_b;

    // batch data
    float* m_z;
    float* m_a;
    float* m_nw;
    float* m_dt;
    float* m_dw;
    float* m_db;

    // test data
    float* m_tz;
    float* m_ta;

    // weight and bias size
    size_t wsize;
    size_t bsize;

    // dropout data
    bool m_d_dropout;
    float m_d_rate;
    uint8_t* m_d_dpmask;
    std::bernoulli_distribution m_d_dropoutdist;

    // convolutional data
    size_t m_c_filters;
    size_t m_c_stride;
    size_t m_c_size;

    // momentum data
    bool m_m_momentum;
    float m_m_coefficient = 0.9f;
    float* m_m_vw;
    float* m_m_vb;

    // l1/l2 data
    bool m_l1;
    float m_l1_lambda;
    bool m_l2;
    float m_l2_lambda;

    // general rng
    std::mt19937 gen;

    // layer metadata
    nlohmann::json m_meta;
};

#include "LayerTemplates.impl.hpp"
