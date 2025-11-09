#include "Layer.hpp"

void Layer::Forward(const Tensor<float> input, size_t elements) {
    if (type == Type::Input) {
        InputForward(input, elements);
    } else {
        HiddenForward(input, elements);
    }
}

void Layer::InputForward(const Tensor<float> input, size_t elements) {
    Tensor<float> a;
    MathUtils::Copy(input, a);
}
void Layer::HiddenForward(const Tensor<float> input, size_t elements) {
    const Tensor<float> w = weights;
    const Tensor<float> b = biases;

    Tensor<float> z;
    Tensor<float> a;

    // copy biases into output, clearing old values
    for (size_t i = 0; i < elements; i++) {
        MathUtils::Copy(b, &z.Data()[i*b.Size()]);
    }

    // compute dot prod between input and weights, and apply activation
    MathUtils::DotProd<true>(input, w, z, elements, iNodes, iNodes, nodes);
    activation.activation(z, a, elements, nodes);
}

void Layer::Backward(const Tensor<float> truth, const Tensor<float> input, const Tensor<float> nextWeights, size_t elements) {
    if (type == Type::Input) { InputBackward(); }
    if (type == Type::Hidden) { HiddenBackward(truth, input, nextWeights, elements); }
    if (type == Type::Output) { OutputBackward(truth, input, elements); }
}

void Layer::InputBackward() {
    return;
}
void Layer::HiddenBackward(const Tensor<float> truth, const Tensor<float> input, const Tensor<float> nextWeights, size_t elements) {
    const Tensor<float> nw = nextWeights;
    const Tensor<float> z = trainingTotals;
    Tensor<float> dt = totalDerivatives;

    MathUtils::DotProdTB<false>(truth, nw, dt, elements, oNodes, nodes, oNodes);
    (activation.derivative)(z, dt, elements, nodes);
    
    ComputeBackward(input, elements);
}
void Layer::OutputBackward(const Tensor<float> truth, const Tensor<float> input, size_t elements) {
    const Tensor<float> a;
    Tensor<float> dt;

    (*lossMetric.loss)(a, truth, dt, elements, nodes);
    ComputeBackward(input, elements);
}

void Layer::ComputeBackward(const Tensor<float> input, size_t elements) {
    Tensor<float> dt = totalDerivatives;
    Tensor<float> dw = weightDerivatives;
    Tensor<float> db = biasDerivatives;

    // compute dw
    MathUtils::DotProdTA<false>(input, dt, dw, elements, iNodes, elements, nodes);

    // compute db, copy clears junk values and sets to first value
    MathUtils::Copy(dt, db, elements);
    MathUtils::SumColumns(&dt[nodes], db, elements-1, nodes);
}

void Layer::Update(size_t elements) {
    optimizer.Update(elements);
}
