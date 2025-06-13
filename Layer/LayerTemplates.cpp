#include "Layer.hpp"
#include "LayerForwards.tpp"
#include "LayerBackwards.tpp"
#include "LayerUpdate.tpp"

template void Layer::InputForward<true>(float*, size_t);
template void Layer::InputForward<false>(float*, size_t);

template void Layer::BasicForward<true, true>(float*, size_t);
template void Layer::BasicForward<true, false>(float*, size_t);
template void Layer::BasicForward<false, true>(float*, size_t);
template void Layer::BasicForward<false, false>(float*, size_t);

template void Layer::ConvolutionalForward<true>(float*, size_t);
template void Layer::ConvolutionalForward<false>(float*, size_t);

template void Layer::BasicBackward<true>(const float*, const float*, size_t);
template void Layer::BasicBackward<false>(const float*, const float*, size_t);

template void Layer::BasicUpdate<true>(float, size_t);
template void Layer::BasicUpdate<false>(float, size_t);
template void Layer::MomentumUpdate<true>(float, size_t);
template void Layer::MomentumUpdate<false>(float, size_t);
