#pragma once
#include <cblas.h>

/* @brief
An entirely static struct that just contains various math implementations, useful to abstract away the complicated math bits, highly optimized
to allow any struct to perform effectively
*/
struct MathUtils {
public:

    template <bool ACUM> static inline void DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) { Sgemm<ACUM, false, false>(a, b, c, ar, ac, br, bc); }
    template <bool ACUM> static inline void DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) { Sgemm<ACUM, true, false>(a, b, c, ar, ac, br, bc); }
    template <bool ACUM> static inline void DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) { Sgemm<ACUM, false, true>(a, b, c, ar, ac, br, bc); }

    template <bool ACUM, bool TRANS_A, bool TRANS_B> static inline void Sgemm(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc) {
        const size_t M = TRANS_A ? ac : ar;
        const size_t N = TRANS_B ? br : bc;
        const size_t K = TRANS_A ? ar : ac;

        constexpr const float beta                 = ACUM ? 1.0f : 0.0f;
        constexpr const CBLAS_TRANSPOSE aT = TRANS_A ? CblasTrans : CblasNoTrans;
        constexpr const CBLAS_TRANSPOSE bT = TRANS_B ? CblasTrans : CblasNoTrans;

        cblas_sgemm(CblasRowMajor, aT, bT, M, N, K, 1.0f, a, ac, b, bc, beta, c, N);
    }

    template <bool ACUM> static inline void SumColumns(const float* a, float* b, size_t ar, size_t ac) {
        if constexpr (ACUM) {
            for (size_t r = 0; r < ar; r++) {
                cblas_saxpy(ac, 1.0f, &a[r * ac], 1, b, 1);
            }
        } else {
            // clear out old values
            cblas_saxpby(ac, 1.0f, &a[0 * ac], 1, 0.0f, b, 1);

            for (size_t r = 1; r < ar; r++) {
                cblas_saxpy(ac, 1.0f, &a[r * ac], 1, b, 1);
            }
        }
    }

    // image augmenting utils
    static float BilinearSample(const float* image, size_t w, size_t h, float fx, float fy);

    static void RotateImage(const float* image, float* out, size_t width, size_t height, float deg);
    static void ScaleImage(const float* image, float* out, size_t width, size_t height, float scale);
    static void ShearImage(const float* image, float* out, size_t width, size_t height, float shear);
    static void ElasticDeformImage(const float* image, float* out, const std::vector<float>& k, std::vector<float>& tmp, std::vector<float>& uxs, std::vector<float>& uys, std::mt19937& rd, size_t width, size_t height, float alpha, float sigma);

    static std::vector<float> MakeGaussianKernel2D(int rad, float sigma);
    static std::vector<float> MakeGaussianKernel1D(int rad, float sigma);

    static std::vector<float> Convolve2D(const std::vector<float>& f, const std::vector<float>& k, size_t w, size_t h, int rad);
    static void ConvolveHorizontal(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad);
    static void ConvolveVertical(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad);

    // math utils
    static float Sum256(__m256 _x);
    static float Sum512(__m512 _x);
    static float Max256(__m256 _x);
    static float Max512(__m512 _x);
    static __m256 Exp256(__m256 _x);
    static __m512 Exp512(__m512 _x);

    static void Scale(float* a, float scale, size_t n);
    static void Normalize(float* a, float lower, float upper, size_t n);

    static std::pair<float,float> ColMinMax(float* a, size_t rows, size_t cols, size_t c);
    static void NormalizeCol(float* a, float lower, float upper, size_t rows, size_t cols, size_t c);
    static void NormalizeCol(float* a, float lower, float upper, float min, float max, size_t rows, size_t cols, size_t c);

    // rng utils
    static uint32_t xorshift32(uint32_t state);
    static float fastRandFloat(uint32_t state);

    /// @brief only works with powers of 2
    static inline size_t RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
    }
};
