#include "Layer.hpp"
#include "LayerForwards.tpp"
#include "LayerBackwards.tpp"

template void Layer::InputForward<true >(float*, size_t);
template void Layer::InputForward<false>(float*, size_t);

template void Layer::BasicForward<true , true , true >(float*, size_t);
template void Layer::BasicForward<false, true , true >(float*, size_t);
template void Layer::BasicForward<true , false, true >(float*, size_t);
template void Layer::BasicForward<false, false, true >(float*, size_t);
template void Layer::BasicForward<true , true , false>(float*, size_t);
template void Layer::BasicForward<false, true , false>(float*, size_t);
template void Layer::BasicForward<true , false, false>(float*, size_t);
template void Layer::BasicForward<false, false, false>(float*, size_t);

template void Layer::Convolutional2DForward<true >(float*, size_t);
template void Layer::Convolutional2DForward<false>(float*, size_t);

template void Layer::BasicBackward<Layer::LayerType::input , true , true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::input , false, true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::input , true , false>(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::input , false, false>(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::hidden, true , true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::hidden, false, true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::hidden, true , false>(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::hidden, false, false>(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::output, true , true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::output, false, true >(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::output, true , false>(const float*, const float*, size_t);
template void Layer::BasicBackward<Layer::LayerType::output, false, false>(const float*, const float*, size_t);
