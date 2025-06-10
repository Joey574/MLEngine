#include "Layer.hpp"

template <bool training>
void Layer::forward(float* __restrict x, size_t n) {
    // calls out to the right forward prop based on passed arguments
	if constexpr (training) {
    	(this->*executeForwardTrain)(x, n);
	} else {
		(this->*executeForwardInfer)(x, n);
	}
}

void Layer::backward(const float* __restrict truth, const float* __restrict input, size_t n) {
    // calls out to the right back prop based on passed arguments
    (this->*executeBackward)(truth, input, n);
}

void Layer::update(float lr, size_t n) {
	// calls our to the right forward prop
	(this->*updateLayer)(lr, n);
}


template void Layer::forward<true>(float*, size_t);
template void Layer::forward<false>(float*, size_t);
