#include "MathUtils.hpp"

void MathUtils::Copy(const Tensor<float>& src, Tensor<float>& dest) {
    assert(src.Data() != nullptr && dest.Data() != nullptr);
    assert(src.Size() == dest.Size());
    assert(!src.HasNan());

    cblas_scopy(dest.Size(), src.Data(), 1, dest.Data(), 1);
}
void MathUtils::CopyByRow(const Tensor<float>& src, Tensor<float>& dest) {
    assert(src.Data() != nullptr && dest.Data() != nullptr);
    assert(dest.Size() % src.Size() == 0);
    assert(!src.HasNan());

    const size_t srcSize = src.Size();
    const size_t n = dest.Size() / srcSize;

    for (size_t i = 0; i < n; i++) {
        cblas_scopy(srcSize, src.Data(), 1, &dest.Data()[i*srcSize], 1);
    }
}

template <bool acum> void MathUtils::SumColumns(const Tensor<float>& a, Tensor<float>& b) {
    assert(a.Dimensionality() == b.Dimensionality()+1);
    assert(a.Data() != nullptr && b.Data() != nullptr);
    assert(!a.HasNan() && !b.HasNan());
    
    constexpr const size_t BLOCK_SIZE = 64;

    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    if constexpr (!acum) {
        b.Zero();
    }

    // TODO : find a way to optimally parallelize
    for (size_t r = 0; r < ar; r += BLOCK_SIZE) {
        for (size_t c = 0; c < ac; c += BLOCK_SIZE) {
            const size_t rMax = std::min(r + BLOCK_SIZE, ar);
            const size_t cMax = std::min(c + BLOCK_SIZE, ac);

            for (size_t i = r; i < rMax; i++) {

                #pragma omp simd
                for (size_t j = c; j < cMax; j++) {
                    b.Data()[j] += a.Data()[i*ac+j];
                }
            }
        }
    }
}
