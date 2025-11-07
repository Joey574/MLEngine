#include "MathUtils.hpp"

template <bool acum> void MathUtils::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(br == ac);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}
template <bool acum> void MathUtils::DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(br == ar);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}
template <bool acum> void MathUtils::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(bc == ac);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}

void MathUtils::ScaleBy(float* __restrict a, float scalar, size_t n) {
    cblas_sscal(n, scalar, a, 1);
}
void MathUtils::ScaleBy(const float* __restrict a, float* __restrict b, size_t n) {
    cblas_saxpy(n, 1.0f, a, 1, b, 1);
}

void MathUtils::Copy(const float* __restrict src, float* __restrict dest, size_t n) {
    cblas_scopy(n, src, 1, dest, 1);
};
float MathUtils::Sum(const float* __restrict a, size_t n) {
    return cblas_ssum(n, a, 1);
}

void MathUtils::SumColumns(const float* __restrict a, float* __restrict b, size_t ar, size_t ac) {
    // TODO : Parallelize
    for (size_t i = 0; i < ar; i++) {

        #pragma omp simd
        for (size_t j = 0; j < ac; j++) {
            b[j] += a[i*ac+j];
        }
    }
}
