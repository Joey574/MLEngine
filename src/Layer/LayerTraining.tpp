#include "Layer.hpp"

template<bool training> void Layer::Forward(const Tensor<float>& input, size_t elements) {
    if (type == Type::Input) {
        InputForward<training>(input, elements);
    } else {
        HiddenForward<training>(input, elements);
    }
}

template<bool training> void Layer::InputForward(const Tensor<float>& input, size_t elements) {
    if constexpr (training) {
        MathUtils::PartialCopy<true>(input, trainingActivations);
    } else {
        MathUtils::PartialCopy<true>(input, testingActivations);
    }
}
template<bool training> void Layer::HiddenForward(const Tensor<float>& input, size_t elements) {
    Tensor<float>* z;
    Tensor<float>* a;

    if constexpr (training) {
        z = &trainingTotals;
        a = &trainingActivations;
    } else {
        z = &testingTotals;
        a = &testingActivations;
    }
    
    // copy biases into output, clearing old values
    MathUtils::CopyByRow(biases, *z);

    // compute dot prod between input and weights, and apply activation
    MathUtils::DotProd<true>(input, weights, *z);
    activation.activation(*z, *a);
}

void Layer::Backward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements) {
    if (type == Type::Input) { InputBackward(); }
    else if (type == Type::Hidden) { HiddenBackward(truth, input, nextWeights, elements); }
    else if (type == Type::Output) { OutputBackward(truth, input, elements); }
}

void Layer::InputBackward() {
    return;
}
void Layer::HiddenBackward(const Tensor<float>& truth, const Tensor<float>& input, const Tensor<float>& nextWeights, size_t elements) {
    MathUtils::DotProdTB<false>(truth, nextWeights, totalDerivatives);
    (activation.derivative)(trainingTotals, totalDerivatives);
    
    ComputeBackward(input, elements);
}
void Layer::OutputBackward(const Tensor<float>& truth, const Tensor<float>& input, size_t elements) {
    (*lossMetric.loss)(trainingActivations, truth, totalDerivatives);
    ComputeBackward(input, elements);
}

void Layer::ComputeBackward(const Tensor<float>& input, size_t elements) {
    MathUtils::DotProdTA<false>(input, totalDerivatives, weightDerivatives);
    MathUtils::SumColumns<false>(totalDerivatives, biasDerivatives);
}

void Layer::Update(size_t elements) {
    optimizer.Update(elements);
}
