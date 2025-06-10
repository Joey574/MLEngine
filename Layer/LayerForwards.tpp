#include "Layer.hpp"
#include "../MathUtils/MathUtils.hpp"

template <bool training> 
void Layer::InputForward(float* __restrict input, size_t n) {
    if constexpr (training) {
        std::memcpy(m_a, input, nodes*n*sizeof(float));
    } else {
        std::memcpy(m_ta, input, nodes*n*sizeof(float));
    }

    assert((uintptr_t)m_a%32==0);
    assert((uintptr_t)m_ta%32==0);
}

template <bool training, bool dropout>
void Layer::BasicForward(float* __restrict input, size_t n) {
    if (type == LayerType::input) {
        InputForward<training>(input, n);
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

    if constexpr (dropout && training) {
        ApplyDropoutFP(n);
    }
}

template <bool training>
void Layer::ConvolutionalForward(float* __restrict input, size_t n) {
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

    for (size_t f = 0; f < m_c_filters; f++) {
        const float* __restrict w = &m_w[f*bsize*bsize];

        for (size_t in = 0; in < n; in++) {
            const float* __restrict x = &input[inodes*in];
        }

        for (size_t r = halfsize; r < height-halfsize; r += m_c_stride) {
            for (size_t c = halfsize; c < width-halfsize; c += m_c_stride) {
                //z[0] += MathUtils::DotProdConv()
            }
        }

    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t x = halfsize; x < width-halfsize; x += m_c_stride) {
        for (size_t y = halfsize; y < height-halfsize; y += m_c_stride) {
            // call special convolutional dotprod here, effectively returns 1 value
            //z[0] += MathUtils::DotProdConv(input, w, );
        }
    }
}