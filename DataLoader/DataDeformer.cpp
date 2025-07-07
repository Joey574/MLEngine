#include "DataLoader.hpp"

void DataLoader::Deform(size_t e) {

    // apply data augments
    if (augment != nullptr) {
        (this->*augment)(e);
    }

    // shuffle new traindata
    Shuffle(e, trainData, trainLabels);    
}
void DataLoader::Shuffle(size_t e, Matrix& data, Matrix& labels) {
    std::mt19937 rng(SEED+22+e);

    // swap blocks in place
    for (size_t i = data.rows-1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        if (i != j) {
            auto block_id = data.data.begin() + (i*data.cols);
            auto block_jd = data.data.begin() + (j*data.cols);

            auto block_il = labels.data.begin() + (i*labels.cols);
            auto block_jl = labels.data.begin() + (j*labels.cols);

            std::swap_ranges(block_id, block_id+data.cols, block_jd);
            std::swap_ranges(block_il, block_il+labels.cols, block_jl);
        }
    }
}

size_t DataLoader::ApplyRotation(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float rot, float mrot, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(-rot, rot);

    // generate randomly rotated images of test dataset
    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {
            float deg = gen(rd);
            deg += deg < 0.0f ? -mrot : mrot;

            MathUtils::RotateImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, deg);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyScale(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float scale, float mscale, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(1.0f-scale, 1.0f+scale);

    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {
            float rscale = gen(rd);
            rscale += rscale < 1.0f ? -mscale : mscale;

            MathUtils::ScaleImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, rscale);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyShear(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float shear, float mshear, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(-shear, shear);

    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {

            float rshear = gen(rd);
            rshear += rshear < 0.0 ? -mshear : mshear;

            MathUtils::ShearImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, rshear);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyElasticDeform(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float alpha, float sigma, size_t samples, size_t a_idx) {
    // pre make gaussian kernel
    std::vector<float> k = MathUtils::MakeGaussianKernel1D(std::ceil(3.0f*sigma), sigma);
    
    // pre allocate scratch space
    std::vector<float> tmp(w*h);
    std::vector<float> uxs(w*h);
    std::vector<float> uys(w*h);

    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {

            MathUtils::ElasticDeformImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], k, tmp, uxs, uys, rd, w, h, alpha, sigma);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }
    
    return a_idx;
}
