#include "Layer.hpp"

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