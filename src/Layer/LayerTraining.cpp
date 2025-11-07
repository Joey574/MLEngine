#include "Layer.hpp"

void Layer::Forward(const float* __restrict input, size_t elements) {
    if (type == Type::Input) {
        InputForward(input, elements);
    } else {
        HiddenForward(input, elements);
    }
}

void Layer::InputForward(const float* __restrict input, size_t elements) {
    float* __restrict a;
    MathUtils::Copy(input, a, elements*nodes);
}
void Layer::HiddenForward(const float* __restrict input, size_t elements) {
    const float* __restrict w = weights;
    const float* __restrict b = biases;

    float* __restrict z;
    float* __restrict a;

    // copy biases into output, clearing old values
    for (size_t i = 0; i < elements; i++) {
        MathUtils::Copy(b, &z[i*biasSize], biasSize);
    }

    // compute dot prod between input and weights, and apply activation
    MathUtils::DotProd<true>(input, w, z, elements, iNodes, iNodes, nodes);
    activation.activation(z, a, elements, nodes);
}

void Layer::Backward(const float* __restrict truth, const float* __restrict input, const float* __restrict nextWeights, size_t elements) {
    if (type == Type::Input) { InputBackward(); }
    if (type == Type::Hidden) { HiddenBackward(truth, input, nextWeights, elements); }
    if (type == Type::Output) { OutputBackward(truth, input, elements); }
}

void Layer::InputBackward() {
    return;
}
void Layer::HiddenBackward(const float* __restrict truth, const float* __restrict input, const float* __restrict nextWeights, size_t elements) {
    const float* __restrict nw = nextWeights;
    const float* __restrict z = trainingTotals;
    float* __restrict dt = totalDerivatives;

    MathUtils::DotProdTB<false>(truth, nw, dt, elements, oNodes, nodes, oNodes);
    (activation.derivative)(z, dt, elements, nodes);
    
    ComputeBackward(input, elements);
}
void Layer::OutputBackward(const float* __restrict truth, const float* __restrict input, size_t elements) {
    const float* __restrict a;
    float* __restrict dt;

    (*lossMetric.loss)(a, truth, dt, elements, nodes);
    ComputeBackward(input, elements);
}

void Layer::ComputeBackward(const float* __restrict input, size_t elements) {
    float* __restrict dt = totalDerivatives;
    float* __restrict dw = weightDerivatives;
    float* __restrict db = biasDerivatives;

    // compute dw
    MathUtils::DotProdTA<false>(input, dt, dw, elements, iNodes, elements, nodes);

    // compute db, copy clears junk values and sets to first value
    MathUtils::Copy(dt, db, elements);
    MathUtils::SumColumns(&dt[nodes], db, elements-1, nodes);
}

void Layer::Update(size_t elements) {
    optimizer.Update(elements);
}
