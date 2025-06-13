#include "Layer.hpp"

nlohmann::json Layer::metadata() {
    if (!m_meta.contains(LAYERTYPE)) { m_meta[LAYERTYPE] = ParseName(type); }
    if (!m_meta.contains(NODES)) { m_meta[NODES] = nodes; }
    if (!m_meta.contains(ACTV) && activation.type != Activation::Type::none) { m_meta[ACTV] = Activation::ParseName(activation.type); }
    if (!m_meta.contains(LOSS) && lossmetric.ltype != LossMetric::Type::none) { m_meta[LOSS] = LossMetric::ParseName(lossmetric.ltype); }
    if (!m_meta.contains(METRIC) && lossmetric.mtype != LossMetric::Type::none) { m_meta[METRIC] = LossMetric::ParseName(lossmetric.mtype); }
    if (!m_meta.contains(DROPOUT) && m_d_dropout) { m_meta[DROPOUT] = m_d_rate; }
    if (!m_meta.contains(PARAMETERS)) { m_meta[PARAMETERS] = params; }

    return m_meta;
}

std::string Layer::ParseName(LayerType type) {
    switch (type) {
        case LayerType::input:
            return "input";
        case LayerType::hidden:
            return "hidden";
        case LayerType::output:
            return "output";
        case LayerType::convolutional:
            return "convolutional";
        default:
            return "none";
    }
}
Layer::LayerType Layer::ParseType(const std::string& type) {
    if (type == "input") {
        return LayerType::input;
    } else if (type == "hidden") {
        return LayerType::hidden;
    } else if (type == "output") {
        return LayerType::output;
    } else if (type == "convolutional") {
        return LayerType::convolutional;
    }
    return LayerType::none;
}


/// @brief only works with powers of 2
size_t Layer::RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
}