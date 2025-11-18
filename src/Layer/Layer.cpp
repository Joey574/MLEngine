#include "Layer.hpp"

void Layer::Define(const YAML::Node& layerConfig, const YAML::Node& optimizerConfig, const TrainingConfig& trainingConfig, size_t in, size_t out) {
    assert(!(defined || built));
    assert(!(optimizer.IsDefined() || optimizer.IsBuilt()));
    assert(layerConfig[Y_LAYERTYPE] && layerConfig[Y_NODES]);

    type = ParseType(layerConfig[Y_LAYERTYPE].as<std::string>());
    weightType = ParseWeightType(layerConfig[Y_WEIGHT].as<std::string>(Y_WEIGHT_DEFAULT));

    // set nodes, input nodes, and output nodes sizes
    nodes = layerConfig[Y_NODES].as<size_t>();
    iNodes = in;
    oNodes = out;

    // set activation function, default to linear
    activation.AssignPointers(layerConfig[Y_ACTIVATION].as<std::string>(Y_ACTV_DEFAULT));

    // set loss / metric functions, default to none
    lossMetric.AssignPointers(
        layerConfig[Y_LOSS].as<std::string>(Y_LOSS_DEFAULT),
        layerConfig[Y_METRIC].as<std::string>(Y_METRIC_DEFAULT)
    );

    // allocate tensors
    switch (type) {
        case Type::Input:
            weights = Tensor<float>(0);
            biases = Tensor<float>(0);
            weightDerivatives = Tensor<float>(0);
            biasDerivatives = Tensor<float>(0);
            break;
        case Type::Hidden: case Type::Output:
            weights = Tensor<float>(iNodes, nodes);
            biases = Tensor<float>(nodes);
            
            weightDerivatives = Tensor<float>(iNodes, nodes);
            biasDerivatives = Tensor<float>(nodes);
            totalDerivatives = Tensor<float>(trainingConfig.batchSize, nodes);
            break;
    }

    // allocate training and testing tensors
    testingTotals = Tensor<float>(trainingConfig.testSize, nodes);
    trainingTotals = Tensor<float>(trainingConfig.batchSize, nodes);
    testingActivations = Tensor<float>(trainingConfig.testSize, nodes);
    trainingActivations = Tensor<float>(trainingConfig.batchSize, nodes);

    optimizer.Define(optimizerConfig, weights.Size(), biases.Size());
    defined = true;
}
void Layer::Build() {
    assert(defined && !built);
    assert(optimizer.IsDefined() && !optimizer.IsBuilt());

    InitializeParameters();

    optimizer.Build(weights, biases, weightDerivatives, biasDerivatives);
    built = true;
}

void Layer::InitializeParameters() {
    std::random_device rd;
    std::mt19937 gen(rd());

    biases.Zero();
    const size_t n = weights.Size();

    if (weightType == WeightType::He) {
        float lower = 0.0f;
        float upper = std::sqrt(2.0f / nodes);
        std::normal_distribution<float> dist(lower, upper);

        for (size_t i = 0; i < n; i++) {
            weights.Data()[i] = dist(gen);
        }
    } else if (weightType == WeightType::Normalize) {
        float lower = -0.5f;
        float upper = 0.5f;
        std::uniform_real_distribution<float> dist(lower, upper);

        for (size_t i = 0; i < n; i++) {
            weights.Data()[i] = dist(gen) * std::sqrt(1.0f / nodes);
        }
    } else if (weightType == WeightType::Xavier) {
        float lower = (-1.0f / std::sqrt(nodes));
        float upper = 1.0f / std::sqrt(nodes);
        std::uniform_real_distribution<float> dist(lower, upper);

        for (size_t i = 0; i < n; i++) {
            weights.Data()[i] = dist(gen);
        }
    } else {
        weights.Zero();
    }
}

int Layer::Save(std::ofstream& f) const {
    assert(defined && built);

    if (!weights.IsEmpty()) f.write((char*)weights.Data(), weights.Size()*sizeof(float));
    if (!biases.IsEmpty()) f.write((char*)biases.Data(), biases.Size()*sizeof(float));
    return 0;
}
int Layer::Load(std::ifstream& f) {
    assert(defined && built);

    if (!weights.IsEmpty()) f.read((char*)weights.Data(), weights.Size()*sizeof(float));
    if (!biases.IsEmpty()) f.read((char*)biases.Data(), biases.Size()*sizeof(float));
    return 0;
}
