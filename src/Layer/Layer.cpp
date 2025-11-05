#include "Layer.hpp"

void Layer::Define(YAML::Node& layerConfig, YAML::Node& optimizerConfig, size_t in, size_t out) {
    assert(!(defined || built));
    assert(layerConfig[Y_LAYERTYPE] && layerConfig[Y_NODES]);

    type = ParseType(layerConfig[Y_LAYERTYPE].as<std::string>());

    // set nodes, input nodes, and output nodes sizes
    nodes = layerConfig[Y_NODES].as<size_t>();
    i_nodes = in;
    o_nodes = out;

    // set activation function, default to linear
    activation.AssignPointers(layerConfig[Y_ACTIVATION].as<std::string>(Y_ACTV_DEFAULT));

    // set loss / metric functions, default to none
    lossMetric.AssignPointers(
        layerConfig[Y_LOSS].as<std::string>(Y_LOSS_DEFAULT),
        layerConfig[Y_METRIC].as<std::string>(Y_METRIC_DEFAULT)
    );

    defined = true;
}

void Layer::Build() {
    assert(defined && !built);
    built = true;
}