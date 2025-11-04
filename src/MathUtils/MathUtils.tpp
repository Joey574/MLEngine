#include "MathUtils.hpp"

template <bool acum> void MathUtils::DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(br == ac);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}
template <bool acum> void MathUtils::DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(br == ar);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}
template <bool acum> void MathUtils::DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(bc == ac);
    constexpr int beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}

void MathUtils::ScaleBy(float* a, float scalar, size_t n) {
    cblas_sscal(n, scalar, a, 1);
}
void MathUtils::ScaleBy(const float* a, float* b, size_t n) {
    cblas_saxpy(n, 1.0f, a, 1, b, 1);
}

void MathUtils::Copy(const float* src, float* dest, size_t n) {
    cblas_scopy(n, src, 1, dest, 1);
};
float MathUtils::Sum(const float* a, size_t n) {
    return cblas_ssum(n, a, 1);
}
