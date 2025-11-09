#include "MathUtils.hpp"

template <bool acum> void MathUtils::DotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    assert(a.Dimensionality == 2 && b.Dimensionality == 2 && c.Dimensionality == 2);

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];
    assert(br == ac);

    constexpr float beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a.Data(), ac, b.Data(), bc,
        beta, c.Data(), bc
    );
}
template <bool acum> void MathUtils::DotProdTA(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    assert(a.Dimensionality == 2 && b.Dimensionality == 2 && c.Dimensionality == 2);

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];
    assert(br == ar);

    constexpr float beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, a, ac, b, bc,
        beta, c, bc
    );
}
template <bool acum> void MathUtils::DotProdTB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    assert(a.Dimensionality == 2 && b.Dimensionality == 2 && c.Dimensionality == 2);

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];
    assert(bc == ac);

    constexpr float beta = acum ? 1.0f : 0.0f;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ar, bc, ac,
        1.0f, a.Data(), ac, b.Data(), bc,
        beta, c.Data(), bc
    );
}

void MathUtils::ScaleBy(Tensor<float>& a, float scalar) {
    cblas_sscal(a.Size(), scalar, a.Data(), 1);
}
void MathUtils::ScaleBy(const Tensor<float>& a, Tensor<float>& b) {
    assert(a.Size() == b.Size());
    cblas_saxpy(a.Size(), 1.0f, a.Data(), 1, b.Data(), 1);
}

void MathUtils::Copy(const Tensor<float>& src, Tensor<float>& dest) {
    assert(src.Size() == dest.Size());
    cblas_scopy(dest.Size(), src.Data(), 1, dest.Data(), 1);
};
float MathUtils::Sum(const Tensor<float>& a) {
    return cblas_ssum(a.Size(), a.Data(), 1);
}

template <bool acum> void MathUtils::SumColumns(const Tensor<float>& a, Tensor<float>& b) {
    assert(a.Dimensionality() == 2 && b.Dimensionality() == 2);

    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    for (size_t i = 0; i < ac; i++) {
        if constexpr (acum) {
            b.Data()[i] += cblas_ssum(a.Size(), a.Data(), ar);
        } else {
            b.Data()[i] = cblas_ssum(a.Size(), a.Data(), ar);
        }
    }
}
