#include "NeuralNetwork.hpp"

void NeuralNetwork::Forward(size_t elements) {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());

    for (size_t i = 0; i < layers.size(); i++) {
        float* __restrict input = i == 0 ? (*dataset).Data() : layers[i-1].Output<true>();
        layers[i].Forward(input, elements);
    }
}

void NeuralNetwork::Backward(size_t elements) {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());

    for (size_t i = 0; i < layers.size(); i++) {
        float* __restrict input = i == 0 ? (*dataset).Data() : layers[i-1].Output<true>();
        float* __restrict truth = i == layers.size()-1 ? nullptr : layers[i+1].Output<true>();
        float* __restrict nextWeights = i == layers.size()-1 ? (*dataset).Labels() : layers[i+1].Weights();

        layers[i].Backward(truth, input, nextWeights, elements);
        layers[i].Update(elements);
    }
}
Score NeuralNetwork::Validate() {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());
    
    return Score(0.0f, true);
}
