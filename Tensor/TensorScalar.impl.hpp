#ifndef TNSR
    #include "Tensor.hpp"
#endif

template <typename T>
inline void Tensor<T>::ScaleBy(T a) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++) {
        data[i] *= a;
    }
}
