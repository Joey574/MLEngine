#pragma once
class NeuralNetwork {
public:

    // basic types for different user options
    enum class WeightInitialization {
        none, he, normalize, xavier
    };
    enum class LossMetric {
        none, mae, accuracy, onehot
    };
    enum class ActivationFunctions {
        none, sigmoid, relu, leakyrelu, elu, softmax
    };

    // contains basic information about the training proccess
    struct History {
        std::chrono::duration<double, std::milli> train_time;
        std::chrono::duration<double, std::milli> epoch_time;
        std::vector<double> metric_history;
    };


    NeuralNetwork() {}

    void initialize(
        std::vector<size_t> dimensions,
        std::vector<ActivationFunctions> activations,
        LossMetric loss,
        LossMetric metric,
        WeightInitialization weightInit
    );

    History fit(
        float* x_train,
        float* y_train,
        float* x_valid,
        float* y_valid,
        size_t training_elements,
        size_t valid_elements,
        size_t batch_size,
        size_t epochs,
        float learning_rate,
        int validation_freq,
        float validation_split,
        bool shuffle
    );


    // user utils
    std::string summary() const;
    nlohmann::json metadata() const;

    std::string compactDimensions() const;
    std::string compactActvations() const;


    // static utils
    static std::vector<size_t> parseCompact(const std::string& dims);
    static std::vector<ActivationFunctions> parseActvs(const std::string& actvs);
    static LossMetric parseLossMetric(const std::string& lm);
    static WeightInitialization parseWeight(const std::string& weight);

    static std::string activationString(const ActivationFunctions actv);
    static std::string weightString(const WeightInitialization w);
    static std::string lossMetricString(const LossMetric lm);


    ~NeuralNetwork() {}

private:

    struct Layer {
        size_t nodes;
        ActivationFunctions actv;
        void (NeuralNetwork::* activation)(float*, float*, size_t);
        void (NeuralNetwork::* derivative)(float*, float*, size_t);
    };

    struct Metric {
        LossMetric type;
        float (NeuralNetwork::* metric)(float*, float*, size_t, size_t);
    };

    struct Loss {
        LossMetric type;
        void (NeuralNetwork::* loss)(float*, float*, float*, size_t, size_t);
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

    void forward_prop(
        float* x_data,
        float* result_data,
        size_t activation_size,
        size_t num_elements
    );

    void back_prop(
        float* x_data,
        float* y_data,
        float learning_rate,
        size_t num_elements
    );


    // dot prods
    void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);
    void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear);

    // initialization utils
    void initializeNetwork();
    void initializeWeights();
    void initialize_batch_data(size_t num_elements);
    void initialize_test_data(size_t num_elements);

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
	|		  total		   |  <- m_batch_activation_size
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
