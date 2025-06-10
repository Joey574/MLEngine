#pragma once
#include "../Activation/Activation.hpp"
#include "../LossMetric/LossMetric.hpp"
#include "../Layer/Layer.hpp"

class TestNetwork;

class NeuralNetwork {
    friend struct Layer;
    friend struct Activation;
    friend struct LossMetric;

public:

    NeuralNetwork() {}

    void Initialize(
        const std::string& path,
        const std::string& name,
        const std::vector<size_t>& dims,
        const std::vector<Activation::Type>& actvs,
        LossMetric::Type loss,
        LossMetric::Type metric,
        Layer::WeightInitialization weightInit
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

    int Load(int fd, Layer::WeightInitialization trueweight);
    int Save(int fd) const;

    static Layer::WeightInitialization ParseWeight(const std::string& w);
    static std::vector<size_t> ParseCompact(const std::vector<std::string>& dims);
    
    static std::string WeightName(Layer::WeightInitialization w);
    std::vector<std::string> CompactDimensions() const;
    std::vector<std::string> CompactActivations() const;

private:

    Layer::WeightInitialization m_weightinit;
    std::string m_path;
    std::string m_name;
    uint64_t m_seed;

    nlohmann::json m_meta;

    std::vector<Layer> m_layers;

    float* m_network;
    size_t m_network_size;

    float* m_batch_data;
    size_t m_batch_data_size;

    float* m_test_data;
    size_t m_test_data_size;
    

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
        size_t e
    ); 


    // initilization function
    void InitializeNetwork(size_t size);
    void InitializeWeights(Layer::WeightInitialization type, const std::vector<std::size_t>& layers);
    void InitializeLayerData(size_t bn, size_t tn);
    void InitializeLayerPointers(size_t bn, size_t tn);

    // logging utils
    static void FitStart(nlohmann::json& history, size_t e, size_t bs, float lr);
    static void FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime);
    static void EpochStart(nlohmann::json& history);
    static void EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e);
    void SaveBest(nlohmann::json& history, float score, size_t e);

    // math utils
    static float Sum256(__m256 _x);
    static __m256 Exp256(__m256 _x);
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
