#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../MathUtils/MathUtils.hpp"
#include "../Optimizer/Optimizer.hpp"
#include "../Tensor/Tensor.hpp"

/* @brief

*/
struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output, conv2D, conv3D
    };
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    Layer() { memset(this, 0, sizeof(Layer)); }

    static std::string ParseName(LayerType type);
    static LayerType ParseType(const std::string& type);

    void Define(std::vector<Layer>& layers, size_t idx, YAML::Node config, YAML::Node optimizerConfig, size_t in, size_t nn);
    void Initialize();
    void InitializeSizes(size_t bn, size_t tn);
    void InitializePointers(char* data, char* batchdata, char* testdata, size_t bn, size_t tn);
    void InitializeSpecialPointers(float* nextweight);

    void InitializeWeights(float* data, WeightInitialization init);

    template<bool training>float* Output() { return training ? m_a : m_ta; }
    float* Truth() { return m_dt; }
    float* Weights() { return m_w; }

    template <bool training> void forward(
        float* x,
        size_t n
    );

    void backward(
        const float* truth,
        const float* input,
        size_t n
    );
    
    void update(
        size_t n
    );

    std::string VisualizeNet();
    std::string VisualizeBatch();
    std::string VisualizeTest();
    static std::string StartEndTotal(size_t offset, size_t start, size_t end);

    // meta data
    std::vector<Layer>* m_layers;
    size_t m_layer_idx;
    
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

    // optimizer data
    Optimizer m_optimizer;
    size_t m_o_bytes;

protected:

    // bools for various options
    bool m_d_dropout;
    bool m_s_skipconn;

    // dropout data
    float m_d_rate;
    uint8_t* m_d_dpmask;
    std::bernoulli_distribution m_d_dropoutdist;
    size_t m_d_dpmask_bytes;

    // convolutional data
    size_t m_c_filters;
    size_t m_c_stride;
    size_t m_c_size;

    // skipconn data
    size_t m_s_idx;
    size_t m_s_base;
    size_t m_s_skip;

private:

    void (Layer::*executeForwardTrain)(float* , size_t);
    void (Layer::*executeForwardInfer)(float*, size_t);
    void (Layer::*executeBackward)(const float*, const float*, size_t);
    void (Layer::*updateLayer)(float, size_t);


    // forward prop methods
    template <bool training, bool dropout, bool skipconn> void BasicForward (float* input, size_t n);
    template <bool training> void Convolutional2DForward(float* input, size_t n);

    // backprop methods
    template <LayerType ltype, bool dropout, bool skipconn> void BasicBackward(const float* truth, const float* input, size_t n);

    // forward prop utils
    template<bool training> void InputForward(float* input, size_t n);
    void ApplyDropoutFP(size_t n);

    // backprop utils
    void ComputeDT(const float* truth, size_t n);
    void ComputeDTOutput(const float* truth, size_t n);
    void ComputeDN(const float* input, size_t n);
    void ComputeSkipDN(const float* input, size_t n);
    void ApplyDropoutBP(size_t n);


    // private initialization utils
    void AssignLayerSize();
    void AssignBasicBatchPtrs(char* batchdata, size_t bn);
    void AssignFunctionPointers();

    void SetBasicBatchTestBytes(size_t bn, size_t tn);
    static std::string CleanSize(size_t bytes);

    // network data
    char* m_net;
    float* m_w;
    float* m_b;
    size_t m_w_bytes;
    size_t m_b_bytes;

    // batch data
    char* m_batch;
    float* m_z;
    float* m_a;
    float* m_nw;
    float* m_dt;
    float* m_dw;
    float* m_db;
    size_t m_z_bytes;
    size_t m_a_bytes;
    size_t m_dt_bytes;
    size_t m_dw_bytes;
    size_t m_db_bytes;

    // test data
    char* m_test;
    float* m_tz;
    float* m_ta;
    size_t m_tz_bytes;
    size_t m_ta_bytes;

    // weight and bias size
    size_t wsize;
    size_t bsize;

    // general rng
    std::mt19937 gen;
};
