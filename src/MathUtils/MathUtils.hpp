#pragma once

struct MathUtils {
public:

    template <bool acum> static inline void DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
        assert(br == ac);
        constexpr int beta = acum ? 1.0f : 0.0f;

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            ar, bc, ac,
            1.0f, a, ac, b, bc,
            beta, c, bc
        );
    }
    template <bool acum> static inline void DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
        assert(br == ar);
        constexpr int beta = acum ? 1.0f : 0.0f;

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            ar, bc, ac,
            1.0f, a, ac, b, bc,
            beta, c, bc
        );
    }
    template <bool acum> static inline void DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
        assert(bc == ac);
        constexpr int beta = acum ? 1.0f : 0.0f;

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            ar, bc, ac,
            1.0f, a, ac, b, bc,
            beta, c, bc
        );
    }

    static inline void ScaleBy(float* a, float scalar, size_t n) {
        cblas_sscal(n, scalar, a, 1);
    }
    static inline void Copy(const float* src, float* dest, size_t n) {
        cblas_scopy(n, src, 1, dest, 1);
    };

    static inline float Sum(const float* a, size_t n) {
        return cblas_ssum(n, a, 1);
    }

private:
};
