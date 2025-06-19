#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Layer/Layer.hpp"

class NeuralNetwork {
    friend struct Layer;
    friend struct Activation;
    friend struct LossMetric;

public:

    NeuralNetwork() {}

    ~NeuralNetwork() {
        if (m_network) { free(m_network); }
        if (m_batch_data) { free(m_batch_data); }
        if (m_test_data) { free(m_test_data); }
    }

    void Initialize(
        const std::string& path,
        const std::string& name,
        YAML::Node& config,
        bool setweights
    );

    nlohmann::json Fit(Dataset& dataset, YAML::Node& config, nlohmann::json& history);

    int Load(int fd, Layer::WeightInitialization trueweight);
    int Save(int fd) const;

    static Layer::WeightInitialization ParseWeight(const std::string& w);
    static std::string WeightName(Layer::WeightInitialization w);

private:

    Layer::WeightInitialization m_weightinit;
    std::string m_path;
    std::string m_name;
    uint64_t m_seed;

    std::vector<Layer> m_layers;

    float* m_network;
    size_t m_network_size;

    float* m_batch_data;
    size_t m_batch_data_size;

    float* m_test_data;
    size_t m_test_data_size;

    size_t m_epoch_since_improvement;
    

    template <bool training> void ForwardProp(
        float* __restrict x,
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
        nlohmann::json& storedhistory,
        size_t e
    ); 


    // initilization function
    void InitializeNetwork(size_t size);
    void InitializeWeights(Layer::WeightInitialization type);
    void InitializeLayerData(size_t bn, size_t tn);
    void InitializeLayerPointers(size_t bn, size_t tn);

    // logging utils
    void FitStart(nlohmann::json& history, size_t e, size_t bs, float lr);
    void FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime);
    void EpochStart(nlohmann::json& history);
    void EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e);
    static std::string CleanTime(std::chrono::nanoseconds time);
    void SaveBest(nlohmann::json& history, nlohmann::json& storedhistory, float score, size_t e);
};

/* Memory Layout

    _____|m_network|_____ 
   |                     |
   |     layer0 data     |  <- layer0.size
   |                     |
   |---------------------|
   |                     |
   |     layerN data     |  <- layerN.size
   |                     |
    ---------------------

    m_network_size = sum(layers.size)


	____|m_batch_data|____
   |                      |
   |   layer0 batchdata   |  <- layer0.batchsize
   |                      |
   |--------------------- |
   |                      |
   |   layerN batchdata   |  <- layerN.batchsize
   |                      |
    ----------------------

    m_batch_data_size = sum(layers.batchsize)


    _____|m_test_data|_____
   |                       |
   |    layer0 testdata    |  <- layer0.testsize
   |                       |
   |-----------------------|
   |                       |
   |    layerN testdata    |  <- layerN.testsize
   |                       |
    -----------------------

    m_test_data_size = sum(layers.testsize)
*/
