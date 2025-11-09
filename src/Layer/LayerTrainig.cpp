#include "LayerTraining.tpp"

template void Layer::Forward<true>(const Tensor<float>&, size_t);
template void Layer::Forward<false>(const Tensor<float>&, size_t);

template void Layer::InputForward<true>(const Tensor<float>&, size_t);
template void Layer::InputForward<false>(const Tensor<float>&, size_t);

template void Layer::HiddenForward<true>(const Tensor<float>&, size_t);
template void Layer::HiddenForward<false>(const Tensor<float>&, size_t);
