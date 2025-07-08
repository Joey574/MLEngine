#pragma once

/* @brief

*/
struct MathUtils {
public:
    template <bool clear> static void DotProd(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <bool clear> static void MatrixColumnSum(const float* a, float* b, size_t a_r, size_t a_c);

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

    static void Normalize(float* a, float sum, size_t n);
    static void Scale(float* a, float scale, size_t n);

    // rng utils
    static uint32_t xorshift32(uint32_t state);
    static float fastRandFloat(uint32_t state);

    /// @brief only works with powers of 2
    static inline size_t RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
    }
};