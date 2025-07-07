#include "MathUtils.hpp"

float MathUtils::BilinearSample(const float* __restrict image, size_t w, size_t h, float fx, float fy) {
    int x0 = fx;
    int y0 = fy;
    int x1 = x0+1;
    int y1 = y0+1;

    x0 = std::clamp(x0, 0, (int)w-1);
    x1 = std::clamp(x1, 0, (int)w-1);
    y0 = std::clamp(y0, 0, (int)h-1);
    y1 = std::clamp(y1, 0, (int)h-1);

    float dx = fx-x0;
    float dy = fy-y0;

    float v00 = image[y0*w + x0];
    float v10 = image[y0*w + x1];
    float v01 = image[y1*w + x0];
    float v11 = image[y1*w + x1];

    float v0 = v00+(v10-v00)*dx;
    float v1 = v01+(v11-v01)*dx;
    float v  = v0+(v1-v0)*dy;

    return v;
}

void MathUtils::RotateImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float deg) {
    const double rad = deg * M_PI / 180.0;
    const double cos_a = std::cos(rad);
    const double sin_a = std::sin(rad);

    const double cx = width / 2.0;
    const double cy = height / 2.0;

    #pragma omp parallel for
    for (size_t y = 0; y < height; y++) {

        #pragma omp simd
        for (size_t x = 0; x < width; x++) {
            double x0 = x - cx;
            double y0 = y - cy;

            double src_x =  cos_a * x0 + sin_a * y0 + cx;
            double src_y = -sin_a * x0 + cos_a * y0 + cy;

            int ix = static_cast<int>(std::round(src_x));
            int iy = static_cast<int>(std::round(src_y));

            // nearest-neighbor interpolation
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                out[y*width+x] = image[iy*width+ix];
            }
        }
    }
}
void MathUtils::ScaleImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float scale) {
    const float nw = width*scale;
    const float nh = height*scale;

    const float dx = (width-nw)/2.0f;
    const float dy = (height-nh)/2.0f;

    #pragma omp parallel for
    for (size_t y = 0; y < height; y++) {

        #pragma omp simd
        for (size_t x = 0; x < width; x++) {
            float srcx = std::min(std::max((x-dx)/scale, 0.0f), width - 1.001f);
            float srcy = std::min(std::max((y-dy)/scale, 0.0f), height - 1.001f);

            float value = BilinearSample(image, width, height, srcx, srcy);

            out[y*width+x] = value;
        }
    }
}
void MathUtils::ShearImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float shear) {
    float cx =  width  * 0.5f;
    float cy =  height * 0.5f;
    float det = 1.0f - shear * shear;

    #pragma omp parallel for
    for (size_t y = 0; y < height; y++) {

        #pragma omp simd
        for (size_t x = 0; x < width; x++) {
            float xr = x-cx;
            float yr = y-cy;

            float x0 = (xr-shear*yr)/det;
            float y0 = (-shear*xr+yr)/det;
            
            float fx = x0+cx;
            float fy = y0+cy;

            if (fx >= 0 && fx < width && fy >= 0 && fy < height) {
                out[y*width+x] = BilinearSample(image, width, height, fx, fy);
            }
        }
    }
}
void MathUtils::ElasticDeformImage(const float* __restrict image, float* __restrict out, const std::vector<float>& k, std::mt19937& rd, size_t width, size_t height, float alpha, float sigma) {
    std::vector<float> elasticImage(width*height, 0.0f);

    // generate displatement fields, ux, uy
    std::vector<float> ux(width*height);
    std::vector<float> uy(width*height);

    uint32_t rng_state = rd();

    #pragma omp parallel for
    for (size_t i = 0; i < width*height; i++) {
        uint32_t local_state = rng_state+i;
        ux[i] = fastRandFloat(local_state);
        uy[i] = fastRandFloat(local_state);
    }

    // apply gaussian smoothing
    int krad = std::ceil(3.0f*sigma);
    std::vector<float> tmp(width*height, 0.0f);
    std::vector<float> uxs(width*height, 0.0f);
    std::vector<float> uys(width*height, 0.0f);

    ConvolveHorizontal(ux, tmp, k, width, height, krad);
    ConvolveVertical(tmp, uxs, k, width, height, krad);

    ConvolveHorizontal(uy, tmp, k, width, height, krad);
    ConvolveVertical(tmp, uys, k, width, height, krad);

    // scale by alpha
    #pragma omp parallel for simd
    for (size_t i = 0; i < width*height; i++) {
        uxs[i] *= alpha;
        uys[i] *= alpha;
    }

    // map into fixed output
    float cx = width*0.5f;
    float cy = height*0.5f;

    #pragma omp parallel for
    for (size_t y = 0; y < height; y++) {

        #pragma omp simd
        for (size_t x = 0; x < width; x++) {
            const size_t idx = y*width+x;

            float xr = x-cx;
            float yr = y-cy;

            float fx = xr+uxs[idx]+cx;
            float fy = yr+uys[idx]+cy;

            if (fx >= 0 && fx < width && fy >= 0 && fy < height) {
                out[idx] = BilinearSample(image, width, height, fx, fy);
            }
        }
    }
}

std::vector<float> MathUtils::MakeGaussianKernel2D(int rad, float sigma) {
    int size = 2*rad+1;
    std::vector<float> k(size*size);    

    float sum = 0.0f;
    float inv2s2 = 1.0f/(2.0f*sigma*sigma);

    // generate kernel
    for (ssize_t dy = -rad; dy <= rad; dy++) {

        #pragma omp simd
        for (ssize_t dx = -rad; dx <= rad; dx++) {
            float v = std::exp(-(dx*dx+dy*dy)*inv2s2);
            k[(dy+rad)*size+(dx+rad)] = v;
            sum += v;
        }
    }

    // normalize
    Normalize(&k[0], sum, k.size());

    return k;
}
std::vector<float> MathUtils::MakeGaussianKernel1D(int rad, float sigma) {
    int size = 2*rad+1;
    std::vector<float> k(size);

    float sum = 0.0f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    #pragma omp simd
    for (ssize_t i = -rad; i <= rad; i++) {
        float v = std::exp(-i * i * inv2s2);
        k[i+rad] = v;
        sum += v;
    }

    Normalize(&k[0], sum, size);
    return k;
}

std::vector<float> MathUtils::Convolve2D(const std::vector<float>& f, const std::vector<float>& k, size_t w, size_t h, int rad) {
    int size = 2*rad+1;
    std::vector<float> convolved(w*h, 0.0f);

    #pragma omp parallel for
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float sum = 0.0f;

            for (ssize_t dy = -rad; dy <= rad; dy++) {
                size_t yy = std::clamp((int)y+(int)dy, 0, (int)h-1);

                for (ssize_t dx = -rad; dx <= rad; dx++) {
                    size_t xx = std::clamp<int>(x+dx, 0, w-1);
                    sum += f[yy*w+xx] * k[(dy+rad)*size+(dx+rad)];
                }
            }

            convolved[y*w+x] = sum;
        }
    }

    return convolved;
}
void MathUtils::ConvolveHorizontal(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad) {
    #pragma omp parallel for
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float sum = 0.0f;

            for (ssize_t dx = -rad; dx <= rad; dx++) {
                size_t xx = std::clamp<int>(x+dx, 0, w-1);
                sum += f[y*w+xx] * k[dx+rad];
            }

            out[y*w+x] = sum;
        }
    }
}
void MathUtils::ConvolveVertical(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad) {
    #pragma omp parallel for
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float sum = 0.0f;

            for (ssize_t dy = -rad; dy <= rad; dy++) {
                size_t yy = std::clamp<int>(y+dy, 0, h-1);
                sum += f[yy*w+x] * k[dy+rad];
            }

            out[y*w+x] = sum;
        }
    }
}
