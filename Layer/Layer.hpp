#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"

struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output
    };
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    void Initialize(LayerType type, size_t in, size_t n, size_t nn, Activation actv, LossMetric lm, float dropout);
    void InitializeSizes(size_t bn, size_t tn);
    void InitializePointers(float* data, float* batchdata, float* testdata, size_t bn, size_t tn);
    void InitializeSpecialPointers(float* nextweight);

    void InitializeWeights(float* data, WeightInitialization init, uint64_t seed);

    float* Output(bool training) { return training ? m_a : m_ta; }
    float* Truth() { return m_dt; }
    float* Weights() { return m_w; }

    void forward(
        bool training,
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

    
    nlohmann::json metadata() const;

    LayerType type;

    size_t nodes;
    size_t inodes;
    size_t nenodes;

    LossMetric lossmetric;
    Activation activation;

    size_t layer_size;
    size_t layer_batch_size;
    size_t layer_test_size;

private:

    void (Layer::*executeForward)(bool, float* __restrict, size_t n);
    void (Layer::*executeBackward)(const float* __restrict, const float* __restrict, size_t n);

    // forward prop methods
    void BasicForward(bool training, float* __restrict input, size_t n);
    void DropoutForward(bool training, float* __restrict input, size_t n);

    // backprop methods
    void BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n);
    void DropoutBackward(const float* __restrict truth, const float* __restrict input, size_t n);

    // backprop utils
    void ComputeDT(const float* __restrict truth, size_t n);
    void ComputeDN(const float* __restrict input, size_t n);

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

    size_t wsize;

    float m_dropout;
    float* m_dpmask;
};
