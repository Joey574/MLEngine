#include "MathUtils.hpp"

void MathUtils::DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(ar == bc);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        0.0f, c, bc
    );
}

void MathUtils::DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(ac == bc);

    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        0.0f, c, bc
    );
}

void MathUtils::DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
    assert(ar == br);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        0.0f, c, bc
    );
}
