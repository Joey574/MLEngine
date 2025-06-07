#include "Layer.hpp"

nlohmann::json Layer::metadata() {
    if (!m_meta.contains(NODES)) { m_meta[NODES] = nodes; }
    if (!m_meta.contains(ACTV) && activation.type != Activation::Type::none) { m_meta[ACTV] = Activation::ParseName(activation.type); }
    if (!m_meta.contains(DROPOUT) && m_dropout > 0.0f) { m_meta[DROPOUT] = m_dropout; }
    if (!m_meta.contains(PARAMETERS)) { m_meta[PARAMETERS] = params; }

    return m_meta;
}