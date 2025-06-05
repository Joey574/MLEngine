#include "Layer.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

void Layer::BasicForward(bool training, float* __restrict input, size_t n) {

    if (type == LayerType::input) { 
        if (training) {
            m_z = input; m_a = input; 
        } else {
            m_tz = input; m_ta = input;
        }
        return; 
    }

    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    float* __restrict z;
    float* __restrict a;
    
    // change output pointers based on if we're training or not
    if (training) {
        z = m_z;
        a = m_a;
    } else {
        z = m_tz;
        a = m_ta;
    }

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*nodes], b, nodes*sizeof(float));
    }

    // perform dot prod with input
    NeuralNetwork::DotProd<false>(input, w, z, n, inodes, inodes, nodes);

    // apply activation
    activation.activation(z, a, n*nodes);
}

void Layer::DropoutForward(bool training, float* __restrict input, size_t n) {
    // start by doing normal forward prop
    BasicForward(training, input, n);

    // input and output should skip dropout
    if (type == LayerType::input || type == LayerType::output) {
        return;
    }

    if (training) {
        float* __restrict a = m_a;
        uint8_t* __restrict mask = m_dpmask;

        // apply dropout
        const float scale = 1.0f/(1.0f-m_dropout);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0f-m_dropout);

        #pragma omp parallel for simd
        for (size_t i = 0; i < n*nodes; i++) {
            bool k = dist(gen);

            if (k) {
                a[i] *= scale;
                mask[i] = 1;            
            } else {
                a[i] = 0.0f;
                mask[i] = 0;
            }
        }
    }
}