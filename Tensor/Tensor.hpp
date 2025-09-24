#pragma once
#define TNSR

template <typename T>
struct Tensor {
public:
    size_t Size() const { return size; }
    std::vector<size_t> Stride() const { return stride; }
    std::vector<size_t> Dimensions() const { return Dimensions; }

    void ScaleBy(T a);
    
private:
    T* data;
    size_t size;

    std::vector<size_t> dimensions;
    std::vector<size_t> stride;
};

#if defined(__AVX512F__)
    #include "TensorAVX512.impl.hpp"    
#elif defined(__AVX2__) && defined(__FMA__)
    #include "TensorAVX2.impl.hpp"
#else
    #include "TensorScalar.impl.hpp"
#endif

#undef TNSR
