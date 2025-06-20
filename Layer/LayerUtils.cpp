#include "Layer.hpp"

std::string Layer::ParseName(LayerType type) {
    switch (type) {
        case LayerType::input:
            return "input";
        case LayerType::hidden:
            return "hidden";
        case LayerType::output:
            return "output";
        case LayerType::conv2D:
            return "conv2D";
        case LayerType::conv3D:
            return "conv3D";
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
    } else if (type == "conv2D") {
        return LayerType::conv2D;
    } else if (type == "conv3D") {
        return LayerType::conv3D;
    }
    return LayerType::none;
}

/// @brief only works with powers of 2
size_t Layer::RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
}

std::string Layer::CleanSize(size_t bytes) {
    long double dbytes = bytes;
    const double gb = 1e9;
    const double mb = 1e6;
    const double kb = 1e3;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (dbytes / gb > 1.00) {
        oss << dbytes / gb << " gb";
    } else if (dbytes / mb > 1.00) {
        oss << dbytes / mb << " mb";
    } else if (dbytes / kb > 1.00) {
        oss << dbytes / kb << " kb";
    } else {
        oss << dbytes << " b";
    }

    return oss.str();
}

std::string Layer::VisualizeNet() {
    std::string res = "";

    if (m_w) {
        res += "\n\t\tm_w: " + CleanSize(wsize*sizeof(float));
    }

    if (m_b) {
        res += "\n\t\tm_b: " + CleanSize(bsize*sizeof(float));
    }

    return res;
}
std::string Layer::VisualizeBatch() {
    std::string res = "";

    return res;
}