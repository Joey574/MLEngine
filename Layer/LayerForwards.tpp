#include "Layer.hpp"

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

template <bool training, bool dropout, bool skipconn>
void Layer::BasicForward(float* __restrict input, size_t n) {
    assert((uintptr_t)input%32==0);

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

    if constexpr (skipconn) {
        float* __restrict input_skip = (*m_layers)[m_s_idx].Output<training>();
        const float* __restrict weight_skip = &m_w[m_s_base*nodes];

        MathUtils::DotProdAcum(input, w, z, n, m_s_base, m_s_base, nodes);
        MathUtils::DotProdAcum(input_skip, weight_skip, z, n, m_s_skip, m_s_skip, nodes);
        activation.activation(z, a, n, nodes);

    } else {
        MathUtils::DotProdAcum(input, w, z, n, inodes, inodes, nodes);
        activation.activation(z, a, n, nodes);
    }


    if constexpr (dropout && training) {
        ApplyDropoutFP(n);
    }
}

template <bool training>
void Layer::Convolutional2DForward(float* __restrict input, size_t n) {
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
    // TODO: the best way it probably going to be making a conv dot prod that can dot prod within a view of a larger matrix

    // TODO: actually start taking into account input dimensions, must be able to take in n dimensional data (scratch it, make 2d and 3d versions for now)
    size_t width = -1;
    size_t height = -1;

    for (size_t f = 0; f < m_c_filters; f++) {
        const float* __restrict w = &m_w[f*m_c_size*m_c_size];

        for (size_t i = 0; i < n; i++) {
            float* __restrict sample = &input[inodes*i];

            //MathUtils::Convolution2D(w, sample, );
        }
    }
}