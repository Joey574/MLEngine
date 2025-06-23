#include "Layer.hpp"
#include "LayerForwards.tpp"
#include "LayerBackwards.tpp"
#include "LayerUpdate.tpp"

template void Layer::InputForward<true>(float*, size_t);
template void Layer::InputForward<false>(float*, size_t);

template void Layer::BasicForward<true , true , true>(float*, size_t);
template void Layer::BasicForward<false, true , true>(float*, size_t);
template void Layer::BasicForward<true , false, true>(float*, size_t);
template void Layer::BasicForward<false, false, true>(float*, size_t);
template void Layer::BasicForward<true , true , false>(float*, size_t);
template void Layer::BasicForward<false, true , false>(float*, size_t);
template void Layer::BasicForward<true , false, false>(float*, size_t);
template void Layer::BasicForward<false, false, false>(float*, size_t);

template void Layer::Convolutional2DForward<true >(float*, size_t);
template void Layer::Convolutional2DForward<false>(float*, size_t);

template void Layer::BasicBackward<true , true >(const float*, const float*, size_t);
template void Layer::BasicBackward<false, true >(const float*, const float*, size_t);
template void Layer::BasicBackward<true , false>(const float*, const float*, size_t);
template void Layer::BasicBackward<false, false>(const float*, const float*, size_t);

template void Layer::BasicUpdate<true , true >(float, size_t);
template void Layer::BasicUpdate<true , false>(float, size_t);
template void Layer::BasicUpdate<false, true >(float, size_t);
template void Layer::BasicUpdate<false, false>(float, size_t);

template void Layer::MomentumUpdate<true , true >(float, size_t);
template void Layer::MomentumUpdate<true , false>(float, size_t);
template void Layer::MomentumUpdate<false, true >(float, size_t);
template void Layer::MomentumUpdate<false, false>(float, size_t);

