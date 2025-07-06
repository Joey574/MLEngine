#pragma once
#include "../DataLoader/DataLoader.hpp"
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Layer/Layer.hpp"

struct State;

struct NeuralNetwork {
    friend struct Layer;
    friend struct Activation;
    friend struct LossMetric;
    friend struct State;

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

    nlohmann::json Fit(DataLoader& dataset, nlohmann::json& history);

    int Save(std::ofstream& file) const;
    int Load(std::ifstream& file);

    static Layer::WeightInitialization ParseWeight(const std::string& w);
    static std::string WeightName(Layer::WeightInitialization w);

    // visualization utils
    std::string Visualize();
    std::string InferenceCost();

    YAML::Node config;

private:

    Layer::WeightInitialization m_weightinit;
    std::string m_path;
    std::string m_name;

    std::vector<Layer> m_layers;

    float* m_network;
    size_t m_network_bytes;

    float* m_batch_data;
    size_t m_batch_data_bytes;

    float* m_test_data;
    size_t m_test_data_bytes;

    size_t m_epoch_since_improvement;
    

    template <bool training> void ForwardProp(
        float* __restrict x,
        size_t n
    );

    void BackProp(
        const float* __restrict x,
        const float* __restrict y,
        size_t n
    );

    std::string TestNetwork(
        DataLoader& dataset,
        nlohmann::json& history,
        nlohmann::json& storedhistory,
        size_t e
    );


    void SaveOptimizers() const;
    void LoadOptimizers();


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
    void SaveBest(nlohmann::json& history, nlohmann::json& storedhistory, float score, size_t e);

    static std::string CleanTime(std::chrono::nanoseconds time);
    static std::string CleanSize(size_t bytes);
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
