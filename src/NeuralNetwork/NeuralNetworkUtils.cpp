#include "NeuralNetwork.hpp"

int NeuralNetwork::Load(std::ifstream& f) {
    assert(defined && built);

    int code = 0;
    for (Layer& l : layers) {
        code += l.Load(f);
    }
    return code;
}
int NeuralNetwork::Save(std::ofstream& f) const {
    assert(defined && built);

    int code = 0;
    for (const Layer& l : layers) {
        code += l.Save(f);
    }
    return code;
}
int NeuralNetwork::LoadOptimizers(std::ifstream& f) {
    assert(defined && built);

    int code = 0;
    for (Layer& l : layers) {
        code += l.Load(f);
    }
    return code;
}
int NeuralNetwork::SaveOptimizers(std::ofstream& f) const {
    assert(defined && built);

    int code = 0;
    for (const Layer& l : layers) {
        code += l.Save(f);
    }
    return code;
}
