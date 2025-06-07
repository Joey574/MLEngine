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

    void Initialize(LayerType type, size_t in, size_t n, size_t nn, Activation actv, LossMetric lm, float dropout);
    void InitializeSizes(size_t bn, size_t tn);
    void InitializePointers(char* data, char* batchdata, char* testdata, size_t bn, size_t tn);
    void InitializeSpecialPointers(float* nextweight);

    void InitializeWeights(float* data, WeightInitialization init, uint64_t seed);

    float* Output(bool training) { return training ? m_a : m_ta; }
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

    
    nlohmann::json metadata();

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

    void (Layer::*executeForwardTrain)(float* __restrict, size_t n);
    void (Layer::*executeForwardInfer)(float* __restrict, size_t n);
    void (Layer::*executeBackward)(const float* __restrict, const float* __restrict, size_t n);

    // template methods
    template <WeightInitialization> void SetWeights(float* data, uint64_t seed);

    // forward prop methods
    template<bool> void BasicForward(float* __restrict input, size_t n);
    template<bool> void DropoutForward(float* __restrict input, size_t n);
    template <bool> void ConvolutionalForward(float* __restrict input, size_t n);

    // backprop methods
    void BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n);
    void DropoutBackward(const float* __restrict truth, const float* __restrict input, size_t n);

    // backprop utils
    void ComputeDT(const float* __restrict truth, size_t n);
    void ComputeDN(const float* __restrict input, size_t n);

    /// @brief only works with powers of 2
    inline size_t RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
    }

    // private initialization utils
    void AssignHiddenBatchPtrs(char* batchdata, size_t bn);
    void AssignOutputBatchPtrs(char* batchdata, size_t bn);

    void SetHiddenBatchTestBytes(size_t bn, size_t tn);
    void SetOutputBatchTestBytes(size_t bn, size_t tn);


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
    float m_d_dropout;
    uint8_t* m_d_dpmask;
    std::bernoulli_distribution m_d_dropoutdist;

    // convolutional data
    size_t m_c_filters;
    size_t m_c_stride;
    size_t m_c_size;
    
    // general rng
    std::mt19937 gen;

    // layer metadata
    nlohmann::json m_meta;
};

#include "LayerForwards.impl.hpp"
#include "LayerTemplates.impl.hpp"