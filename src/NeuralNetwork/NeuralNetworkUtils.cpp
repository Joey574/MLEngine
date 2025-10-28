#include "NeuralNetwork.hpp"

int NeuralNetwork::Load(std::ifstream& file) {
    assert(defined && !built);

    built = true;
    return 0;
}
void NeuralNetwork::Save(std::ofstream& file) const {
    assert(defined && built);
}
void NeuralNetwork::LoadOptimizers(std::ifstream& file) {
    assert(defined && built);
}
void NeuralNetwork::SaveOptimizers(std::ofstream& file) const {
    assert(defined && built);
}
