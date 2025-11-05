#include "NeuralNetwork.hpp"

void NeuralNetwork::Forward() {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());
}

void NeuralNetwork::Backward() {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());
}
Score NeuralNetwork::Validate() {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());
    
    return Score(0.0f, true);
}
