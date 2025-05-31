#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Layer/Layer.hpp"

class TestNetwork;

class NeuralNetwork {
    friend class TestNetwork;

    friend struct Layer;
    friend struct Activation;
    friend struct LossMetric;

public:

    // basic types for different user options
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    NeuralNetwork() {}

    void Initialize(
        const std::string& path,
        const std::string& name,
        const std::vector<size_t>& dims,
        const std::vector<Activation::Type>& actvs,
        LossMetric::Type loss,
        LossMetric::Type metric,
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


    nlohmann::json Metadata();

    int Load(int fd, WeightInitialization trueweight);
    int Save(int fd) const;

    static WeightInitialization ParseWeight(const std::string& w);
    static std::vector<size_t> ParseCompact(const std::vector<std::string>& dims);
    
    static std::string WeightName(WeightInitialization w);
    std::vector<std::string> CompactDimensions() const;
    std::vector<std::string> CompactActivations() const;

private:

    WeightInitialization m_weightinit;
    std::string m_path;
    std::string m_name;
    uint64_t m_seed;

    nlohmann::json m_meta;

    std::vector<Layer> m_layers;

    
    float* m_network;
    float* m_biases;
    size_t m_network_size;
    size_t m_weights_size;
    size_t m_biases_size;

    float* m_batch_data;
    float* m_batch_actv;
    float* m_d_total;
    float* m_d_weights;
    float* m_d_biases;
    size_t m_batch_data_size;
    size_t m_batch_actv_size;

    float* m_test_data;
    float* m_test_actv;
    size_t m_test_data_size;
    size_t m_test_actv_size;
    

    void ForwardProp(
        bool training,
        float* __restrict x,
        float* __restrict z,
        float* __restrict a,
        size_t n
    );

    void BackProp(
        const float* __restrict x,
        const float* __restrict y,
        float lr,
        size_t n
    );

    std::string TestNetwork(
        Dataset& dataset,
        nlohmann::json& history,
        size_t e
    ); 


    // initilization function
    void InitializeNetwork(const std::vector<size_t>& dims);
    void InitializeWeights(WeightInitialization type, const std::vector<std::size_t>& layers);
    void InitializeBatchData(size_t n);
    void InitializeTestData(size_t n);

    // logging utils
    static void FitStart(nlohmann::json& history, size_t e, size_t bs, float lr);
    static void FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime);
    static void EpochStart(nlohmann::json& history);
    static void EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e);
    void SaveBest(nlohmann::json& history, float score, size_t e);
       
    // dot prods
    static void DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);

    // math utils
    static float Sum256(__m256 _x);
    static __m256 Exp256(__m256 _x);
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
