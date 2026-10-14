#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../MathUtils/MathUtils.hpp"
#include "../Optimizer/Optimizer.hpp"

/* @brief
Meat and potatoes of all the machine learning implementations, handles input and output for neural network layers
and manages the optimizer for parameter update
*/
struct Layer {
public:

    enum class LayerType {
        none, input, hidden, output, conv2D, conv3D
    };
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    Layer() :
        nodes(0),
        inodes(0),
        nenodes(0),
        layer_bytes(0),
        layer_batch_bytes(0),
        layer_test_bytes(0),
        m_w(nullptr),
        m_b(nullptr),
        m_z(nullptr),
        m_a(nullptr),
        m_d_dropout(false),
        m_s_skipconn(false)
    {}

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

    template <bool training> void forward(float* x, size_t n);
    void backward(const float* truth, const float* input, size_t n);
    void update(size_t n);

    std::string VisualizeNet();
    std::string VisualizeBatch();
    std::string VisualizeTest();
    static std::string StartEndTotal(size_t offset, size_t start, size_t end);

    // meta data
    std::vector<Layer>* m_layers;
    size_t m_layer_idx;

    LayerType type;

    size_t nodes = 0;
    size_t inodes = 0;
    size_t nenodes = 0;

    LossMetric lossmetric;
    Activation activation;

    size_t layer_bytes = 0;
    size_t layer_batch_bytes = 0;
    size_t layer_test_bytes = 0;

    size_t params = 0;

    // optimizer data
    Optimizer m_optimizer;
    size_t m_o_bytes = 0;

protected:

    // bools for various options
    bool m_d_dropout = false;
    bool m_s_skipconn = false;

    // dropout data
    float m_d_rate = 0;
    uint8_t* m_d_dpmask = nullptr;
    std::bernoulli_distribution m_d_dropoutdist;
    size_t m_d_dpmask_bytes = 0;

    // convolutional data
    size_t m_c_filters = 0;
    size_t m_c_stride = 0;
    size_t m_c_size = 0;

    // skipconn data
    size_t m_s_idx = 0;
    size_t m_s_base = 0;
    size_t m_s_skip = 0;

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
    char* m_net = nullptr;
    float* m_w = nullptr;
    float* m_b = nullptr;
    size_t m_w_bytes = 0;
    size_t m_b_bytes = 0;

    // batch data
    char* m_batch = nullptr;
    float* m_z = nullptr;
    float* m_a = nullptr;
    float* m_nw = nullptr;
    float* m_dt = nullptr;
    float* m_dw = nullptr;
    float* m_db = nullptr;
    size_t m_z_bytes = 0;
    size_t m_a_bytes = 0;
    size_t m_dt_bytes = 0;
    size_t m_dw_bytes = 0;
    size_t m_db_bytes = 0;

    // test data
    char* m_test = nullptr;
    float* m_ta = nullptr;
    size_t m_ta_bytes = 0;

    // weight and bias size
    size_t wsize = 0;
    size_t bsize = 0;

    // general rng
    std::mt19937 gen;
};
