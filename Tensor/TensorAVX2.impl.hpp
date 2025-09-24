#ifndef TNSR
    #include "Tensor.hpp"
#endif

template <typename T>
inline void Tensor<T>::ScaleBy(T a) {
    static_assert(std::is_same_v<T, float>, "T must be float")

    const __m512 _a = _mm512_set1_ps(a);

    #pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-16; i += 16) {
        const __m256 _x = _mm512_load_ps(&data[i]);
        const __m256 _res = _mm512_mul_ps(_x, _a);

        _mm512_store_ps(&data[i], _res);
    }

    for (size_t i = size-(size%16); i < size; i++) {
        data[i] *= a;
    }
}
