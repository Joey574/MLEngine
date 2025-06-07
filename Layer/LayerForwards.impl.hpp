#pragma once
#include "Layer.hpp"
#include "../MathUtils/MathUtils.hpp"

template <bool training>
void Layer::forward(float* __restrict x, size_t n) {
    // calls out to the right forward prop based on passed arguments
	if constexpr (training) {
    	(this->*executeForwardTrain)(x, n);
	} else {
		(this->*executeForwardInfer)(x, n);
	}
}

template <bool training>
void Layer::BasicForward(float* __restrict input, size_t n) {

    if (type == LayerType::input) { 
        if constexpr (training) {
            if ((uintptr_t)input%32 == 0) {
                m_a = input;
            } else {
                std::memcpy(m_a, input, nodes*n*sizeof(float));
            }
        } else {
            if ((uintptr_t)input%32 == 0) {
                m_ta = input;
            } else {
                std::memcpy(m_ta, input, nodes*n*sizeof(float));
            }
        }
        return; 
    }

    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    float* __restrict z;
    float* __restrict a;
    
    // change output pointers based on if we're training or not
    if constexpr (training) {
        z = m_z;
        a = m_a;
    } else {
        z = m_tz;
        a = m_ta;
    }

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*bsize], b, bsize*sizeof(float));
    }

    MathUtils::DotProdActv<false>(activation.type, input, w, z, a, n, inodes, inodes, nodes);
}

template <bool training>
void Layer::DropoutForward(float* __restrict input, size_t n) {
    // start by doing normal forward prop
    BasicForward<training>(input, n);

    // input and output should skip dropout
    if (type == LayerType::input || type == LayerType::output) {
        return;
    }

    // apply dropout if training
    if constexpr (training) {
        float* __restrict a = m_a;
        uint8_t* __restrict mask = m_d_dpmask;

        const float scale = 1.0f/(1.0f-m_d_dropout);

        #pragma omp parallel for simd
        for (size_t i = 0; i < n*nodes; i++) {
            const bool k = m_dropoutdist(gen);

            const size_t byteidx = i >> 3;
            const uint8_t bitidx = i & 7;

            if (k) {
                a[i] *= scale;
                mask[byteidx] |= (1 << bitidx);
            } else {
                a[i] = 0.0f;
                mask[byteidx] &= (0 << bitidx);
            }
        }
    }
}

template <bool training>
void Layer::ConvolutionalForward(float* __restrict input, size_t n) {
    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    float* __restrict z;
    float* __restrict a;
    
    // change output pointers based on if we're training or not
    if constexpr (training) {
        z = m_z;
        a = m_a;
    } else {
        z = m_tz;
        a = m_ta;
    }

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*bsize], b, bsize*sizeof(float));
    }

    // at this point we need to do our convolutional dot prod, the sliding, all that fun stuff
    // TODO: the best way it probably going to be making a conv dot prod that can dot prod with a view of a larger matrix

    // TODO: actually start taking into account input dimensions, must be able to take in n dimensional data
    size_t width = -1;
    size_t height = -1;

    const size_t halfsize = (m_c_size+1)/2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t x = halfsize; x < width-halfsize; x += m_c_stride) {
        for (size_t y = halfsize; y < height-halfsize; y += m_c_stride) {
            // call special convolutional dotprod here, effectively returns 1 value
        }
    }
}