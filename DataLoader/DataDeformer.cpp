#include "DataLoader.hpp"

void DataLoader::Deform(size_t e, const std::string& path) {

    // apply data augments
    if (augment != nullptr && e % refresh_every == 0) {
        (this->*augment)(e);
    }

    // special case for mandlebrot
    if (type == Type::mandlebrot && e % refresh_every == 0 && e != 0) {
        LoadMandlebrotTrainData(e, path);
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

// Writes a normalized [-1, 1] sampling of fourier series to x
void DataLoader::ComputeFourier(float* x, float xmin, float xmax, float ymin, float ymax, size_t series) {
    const double xv = -M_PI + ((double)x[0] - xmin) / ((double)xmax - xmin) * (2.0*M_PI);
    const double yv = -M_PI + ((double)x[1] - ymin) / ((double)ymax - ymin) * (2.0*M_PI);

    const double sx = std::sin(xv), cx = std::cos(xv);
    const double sy = std::sin(yv), cy = std::cos(yv);
    double skx = sx, ckx = cx, sky = sy, cky = cy;

    float* o = x + 2;

    for (size_t i = 0; i < series; i++, o += 4) {
        o[0] = (float)skx;
        o[1] = (float)ckx;
        o[2] = (float)sky;
        o[3] = (float)cky;

        const double nsx = skx * cx + ckx * sx;
        ckx = ckx * cx - skx * sx;
        skx = nsx;

        const double nsy = sky * cy + cky * sy;
        cky = cky * cy - sky * sy;
        sky = nsy;
    }

    x[0] = (float)(xv / M_PI);
    x[1] = (float)(yv / M_PI);
}
