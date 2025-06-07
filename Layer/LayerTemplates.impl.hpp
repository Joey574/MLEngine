#pragma once
#include "Layer.hpp"

template <Layer::WeightInitialization init>
void Layer::SetWeights(float* data, uint64_t seed) {
    if (type == LayerType::input) { return; }

    float lowerRand;
    float upperRand;
    size_t idx = 0;
    
    std::default_random_engine gen(seed);

    // zero out biases
    memset(&data[wsize], 0, nodes*sizeof(float));

    if constexpr (init == WeightInitialization::he) {
        
        lowerRand = 0.0f;
        upperRand = std::sqrt(2.0f/nodes);

        std::normal_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen);
        }
    } else if constexpr (init == WeightInitialization::normalize) {
        
        lowerRand = -0.5f;
        upperRand = 0.5f;

        std::uniform_real_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen) * std::sqrt(1.0f/nodes);
        }
    } else if constexpr (init == WeightInitialization::xavier) {
        
        lowerRand = (-1.0f/std::sqrt(nodes));
        upperRand = 1.0f/std::sqrt(nodes);

        std::uniform_real_distribution<float> dist(lowerRand, upperRand);
        for (size_t i = 0; i < wsize; i++) {
            data[i] = dist(gen);
        }
    } else {
        // no weight initialization has been set, zero the weights
        memset(data, 0, wsize*sizeof(float));
    }
}
