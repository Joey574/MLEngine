#include "NeuralNetwork.hpp"

void NeuralNetwork::Forward() {
    assert(defined && built);
}
void NeuralNetwork::Backward() {
    assert(defined && built);
}
Score NeuralNetwork::Validate() {
    assert(defined && built);
    return Score(0.0f, true);
}
