#include "NeuralNetwork.hpp"

void NeuralNetwork::Forward(size_t startElement, size_t numElements) {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());

    for (size_t i = 0; i < layers.size(); i++) {

        // if this is the first layer, input is the dataset, else it is the last layer's output
        Tensor<float>* input = i == 0 ? 
            &(*dataset).Data(startElement) :
            &layers[i-1].Output<true>();

        layers[i].Forward<true>(*input, numElements);
    }
}

void NeuralNetwork::Backward(size_t startElement, size_t numElements) {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());

    for (size_t i = 0; i < layers.size(); i++) {

        // if this is the first layer, input would've been the dataset, else it was last layer's output
        Tensor<float>* input = i == 0 ?
            &(*dataset).Data(startElement) :
            &layers[i-1].Output<true>();

        // if this is the last layer, the truth is the dataset labels, else it was the next layer's output
        Tensor<float>* truth = i == layers.size()-1 ?
            &(*dataset).Labels(startElement) :
            &layers[i+1].Output<true>();

        // if this is the last layer, there are no next weight, else get the next layer's weights
        Tensor<float>* nextWeights = i == layers.size()-1 ?
            nullptr :
            &layers[i+1].Weights();

        layers[i].Backward(*truth, *input, *nextWeights, numElements);
        layers[i].Update(numElements);
    }
}
Score NeuralNetwork::Validate() {
    assert(defined && built);
    assert(dataset->IsBuilt() && dataset->IsDefined());
    
    return Score(0.0f, true);
}
