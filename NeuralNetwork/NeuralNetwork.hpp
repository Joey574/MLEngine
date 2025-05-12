#pragma once
class TestNetwork;

#define LOGLOSS 0
#define LOGSCORE 0
#define LOGDP 0


class NeuralNetwork {
    friend class TestNetwork;

public:

    // basic types for different user options
    enum class WeightInitialization {
        none, he, normalize, xavier
    };
    enum class LossMetric {
        none, mae, mse, accuracy, onehot
    };
    enum class ActivationFunctions {
        none, sigmoid, relu, leakyrelu, elu, softmax
    };

    NeuralNetwork() {}

    void Initialize(
        const std::string& path,
        const std::string& name,
        const std::vector<size_t>& dimensions,
        const std::vector<ActivationFunctions>& activations,
        LossMetric loss,
        LossMetric metric,
        WeightInitialization weightInit
    );

    nlohmann::json Fit(
        Dataset& dataset,
        size_t batch_size,
        size_t epochs,
        float learning_rate,
        int validation_freq,
        float validation_split,
        bool shuffle
    );


    // user utils
    std::string Summary() const;
    nlohmann::json Metadata();

    std::string CompactDimensions() const;
    std::string CompactActvations() const;

    int Load(int fd, WeightInitialization trueweight);
    int Save(int fd) const;


    // static utils
    static std::vector<size_t> ParseCompact(const std::string& dims);
    static std::vector<ActivationFunctions> ParseActvs(const std::string& actvs);
    static LossMetric ParseLossMetric(const std::string& lm);
    static WeightInitialization ParseWeight(const std::string& weight);

    static std::string ActivationString(const ActivationFunctions actv);
    static std::string WeightString(const WeightInitialization w);
    static std::string LossMetricString(const LossMetric lm);

    ~NeuralNetwork() {
        std::cout << "Deconstructing\n";
        if (m_network) { free(m_network); }
        if (m_batch_data) { free(m_batch_data); }
        if (m_test_data) { free(m_test_data); }
    }

private:

    struct Layer {
        size_t nodes;
        ActivationFunctions actvtype;
        void (*activation)(const float*, float*, size_t);
        void (*derivative)(const float*, float*, size_t);
    };

    struct Metric {
        LossMetric type;
        bool highestIsBest;
        float (*metric)(const float*, const float*, size_t, size_t);
    };

    struct Loss {
        LossMetric type;
        void (*loss)(const float*, const float*, float*, size_t, size_t);
    };

    // pointers to various memory blocks that contain all the data
    float* m_network;
    float* m_batch_data;
    float* m_test_data;

    float* m_biases;

    float* m_activation;

    float* m_d_total;
    float* m_d_weights;
    float* m_d_biases;

    float* m_test_activation;

    // size of various memory blocks
    size_t m_network_size;
    size_t m_weights_size;
    size_t m_biases_size;

    size_t m_batch_data_size;
    size_t m_batch_activation_size;

    size_t m_test_data_size;
    size_t m_test_activation_size;

    // internal network config
    std::vector<Layer> m_layers;
    Metric m_metric;
    Loss m_loss;
    WeightInitialization m_weight_init;

    unsigned int m_seed;
    std::string m_path;
    std::string m_name;

    nlohmann::json m_meta;

    void ForwardProp(
        const float* __restrict x_data,
        float* __restrict result_data,
        size_t activation_size,
        size_t num_elements
    );

    void BackProp(
        const float* __restrict x_data,
        const float* __restrict y_data,
        float learning_rate,
        size_t num_elements
    );

    std::string TestNetwork(
        const Dataset& dataset,
        nlohmann::json& history,
        size_t e
    );

    void DropoutFP(float* __restrict x, size_t n, float dropout) const;
    void DropoutBP(float* __restrict x, size_t n) const;

    void SaveBest(nlohmann::json& history, float score, size_t e);

    void AssignActvFunctions(const std::vector<ActivationFunctions>& actvs);
    void AssignLossFunctions(LossMetric loss, LossMetric metric);

    // activation functions
    static void Sigmoid(const float* __restrict x, float* __restrict y, size_t n);
    static void ReLU(const float* __restrict x, float* __restrict y, size_t n);
    static void LeakyReLU(const float* __restrict x, float* __restrict y, size_t n);
    static void ELU(const float* __restrict x, float* __restrict y, size_t n);
    static void Softmax(const float* __restrict x, float* __restrict y, size_t n);

    // derivatives functions
    static void SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void ReLUDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void ELUDerivative(const float* __restrict x, float* __restrict y, size_t n);

    // loss functions
    static void MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);

    // metric functions
    static float MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);

    // dot prods
    static void DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);

    // simd utils
    static float Sum256(__m256 x);
    static __m256 Exp256(__m256 x);

    // initialization utils
    void InitializeNetwork();
    void InitializeWeights();
    void InitializeBatchData(size_t num_elements);
    void InitializeTestData(size_t num_elements);
    void InitializeLoss(LossMetric loss);
    void InitializeMetric(LossMetric metric);

    // static private utils
    static void FitStart(nlohmann::json& history, size_t e, size_t bs, float lr);
    static void FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime);
    static void EpochStart(nlohmann::json& history);
    static void EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e);
};

/* Memory Layout

	 _____|m_network|_____ 
	|					  |
	|		weights		  |  <- m_weights_size
	|					  |
	|------|m_biases|-----|
	|					  |
	|		 biases		  |  <- m_bias_size
	|					  |
	 ---------------------

	m_network_size := m_weights_size + m_bias_size



	 ____|m_batch_data|____
	|					   |
	|		 total		   |  <- m_batch_activation_size
	|					   |
	|----|m_activation|----|
	|					   |
	|	   activation	   |  <- m_batch_activation_size
	|					   |
	|------|m_d_total|-----|
	|					   |
	|		d_total		   |  <- m_batch_activation_size
	|					   |
	|-----|m_d_weights|----|
	|					   |
	|	   d_weights	   |  <- m_weights_size
	|					   |
	|-----|m_d_biases|-----|
	|					   |
	|	    d_biases	   |  <- m_bias_size
	|					   |
	 ----------------------

	m_batch_data_size := (3 * m_batch_activation_size) + m_network_size



	 _____|m_test_data|_____
	|					    |
	|		  total		    |  <- m_test_activation_size
    |					    |
	|--|m_test_activation|--|
	|					    |
	|	   activation	    |  <- m_test_activation_size
	|					    |
	 -----------------------

    m_test_data_size := (2 * m_test_activation_size)

*/
