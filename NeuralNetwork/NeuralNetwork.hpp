#pragma once

#include "../Layer/Layer.hpp"
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp";

class TestNetwork;

class NeuralNetwork {
    friend class TestNetwork;

public:

    // basic types for different user options
    enum class WeightInitialization {
        none, he, normalize, xavier
    };

    NeuralNetwork() {}

    void Initialize(
        const std::string& path,
        const std::string& name,
        const std::vector<size_t>& dimensions,
        const std::vector<Activation::Type>& activations,
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


    nlohmann::json Metadata();

    int Load(int fd, WeightInitialization trueweight);
    int Save(int fd) const;

private:

    void ForwardProp(
        bool training,
        const float* __restrict x,
        float* __restrict results,
        size_t actvsize,
        size_t n
    );

    void BackProp(
        const float* __restrict x,
        const float* __restrict y,
        float lr,
        size_t n
    );

    std::string TestNetwork(
        const Dataset& dataset,
        nlohmann::json& history,
        size_t e
    ); 
    
    std::vector<Layer> m_layers;
   
    // dot prods
    static void DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    static void DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
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
