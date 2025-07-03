#include "DataLoader.hpp"

/// @brief Returns the passed dataset as constrained by args
void DataLoader::LoadDataset(YAML::Node& config) {
    std::string dataset = config[Y_DATASET].as<std::string>();
    args = config[Y_DATASETARGS];

    if (dataset == "mnist") {
        LoadMNIST();
    } else if (dataset == "fmnist") {
        LoadFMNIST();
    } else if (dataset == "mandlebrot") {
        LoadMandlebrot();
    } else {
        std::cerr << "Failed to initialize dataset\n";
    }
}

void DataLoader::LoadMNIST() {
    name = "mnist";
    type = Type::mnist;
    hasTestData = true;

    // training dataset path
    std::string trainingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TrainingData/train-images.idx3-ubyte");
    std::string trainingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    std::string testingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TestingData/t10k-images.idx3-ubyte");
    std::string testingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TestingData/t10k-labels.idx1-ubyte");

    // open files
    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        std::cerr << "Failed to open dataset file(s)\n";
    }

    LoadMNISTStyleDataset(traind, trainl, testd, testl);

    // close files
    traind.close();
    trainl.close();
    testd.close();
    testl.close();
}
void DataLoader::LoadFMNIST() {
    name = "fmnist";
    type = Type::fmnist;
    hasTestData = true;

    // training dataset path
    std::string trainingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/FMNIST/TrainingData/train-images-idx3-ubyte");
    std::string trainingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/FMNIST/TrainingData/train-labels-idx1-ubyte");

    // testing dataset path
    std::string testingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/FMNIST/TestingData/t10k-images-idx3-ubyte");
    std::string testingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/FMNIST/TestingData/t10k-labels-idx1-ubyte");

    // open files
    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        std::cerr << "Failed to open dataset file(s)\n";
    }

    LoadMNISTStyleDataset(traind, trainl, testd, testl);

    // close files
    traind.close();
    trainl.close();
    testd.close();
    testl.close();
}
void DataLoader::LoadMandlebrot() {
    name = "mandlebrot";
    type = Type::mandlebrot;
    hasTestData = true;

    size_t n = args[Y_SAMPLES].as<size_t>(Y_SAMPLE_DEFAULT);
    size_t depth = args[Y_MANDLEDEPTH].as<size_t>(Y_MANDLEDEPTH_DEFAULT);
    size_t fourier = args[Y_FOURIERSERIES].as<size_t>(Y_FOURIER_DEFAULT);

    const size_t test_elements = 10000 > (n*0.1) ? 10000 : n*0.1;

    const double xMin = -2.5;
    const double xMax = 1.0;
    const double yMin = -1.1;
    const double yMax = 1.1;

    std::mt19937 gen(SEED-82);

    std::uniform_real_distribution<double> xrand(xMin, xMax);
    std::uniform_real_distribution<double> yrand(yMin, yMax);

    trainDataRows = n;
    trainDataCols = 2 + (fourier*4);
    testDataRows = test_elements;
    testDataCols = 2 + (fourier*4);

    trainLabelRows = n;
    trainLabelCols = 1;
    testLabelRows = test_elements;
    testLabelCols = 1;

    trainData = std::vector<float>(trainDataRows*trainDataCols);  
    testData = std::vector<float>(testDataRows*testDataCols);

    trainLabels = std::vector<float>(n);
    testLabels = std::vector<float>(test_elements);

    // build training dataset
    for (size_t i = 0; i < n; i++) {
        double x = xrand(gen);
        double y = yrand(gen);

        float m = InMandlebrot(x, y, depth);

        trainData[i*trainDataCols] = x;
        trainData[i*trainDataCols+1] = y;
        trainLabels[i] = m;

        ComputeFourier(&trainData[i*trainDataCols], fourier);
    }

    // build testing dataset
    for (size_t i = 0; i < test_elements; i++) {
        double x = xrand(gen);
        double y = yrand(gen);

        float m = InMandlebrot(x, y, depth);

        testData[i*testDataCols] = x;
        testData[i*testDataCols+1] = y;
        testLabels[i] = m;

        ComputeFourier(&testData[i*testDataCols], fourier);
    }

    #pragma omp parallel for
    for (size_t c = 0; c < trainDataCols; c++) {
        // find col min/max
        float min = trainData[c];
        float max = trainData[c];
        for (size_t i = 1; i < trainDataRows; i++) {
            if (trainData[i*trainDataCols+c] > max) { max = trainData[i*trainDataCols+c]; }
            if (trainData[i*trainDataCols+c] < min) { min = trainData[i*trainDataCols+c]; }
        }

        if (max <= min) {
            max = min + 1.0f;
        }

        const float range = max-min;

        // normalize training col
        for (size_t i = 0; i < trainDataRows; i++) {
            const size_t idx = i*trainDataCols+c;
            trainData[idx] = (trainData[idx] - min) / range;
        }

        // normalize testing col
        for (size_t i = 0; i < testDataRows; i++) {
            const size_t idx = i*testDataCols+c;
            testData[idx] = (testData[idx] - min) / range;
        }
    }
}

void DataLoader::LoadMNISTStyleDataset(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl) {
    ReadBigInt(&trainl);
    ReadBigInt(&trainl);

    ReadBigInt(&traind);
    int imagenum = ReadBigInt(&traind);
    int width = ReadBigInt(&traind);
    int height = ReadBigInt(&traind);

    // set up vector sizes
    originalData.data = std::vector<float>();
    originalLabels.data = std::vector<float>(imagenum);
    originalData.data.reserve(imagenum*width*height);

    originalData.rows = imagenum;
    originalData.cols = width*height;

    originalLabels.rows = imagenum;
    originalLabels.cols = 1;

    // parse out training data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        traind.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        originalData.data.insert(originalData.data.end(), floatdata.begin(), floatdata.end());

        // get label for data
        char byte;
        trainl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        originalLabels.data[i] = label;
    }

    ReadBigInt(&testl);
    ReadBigInt(&testl);

    ReadBigInt(&testd);
    imagenum = ReadBigInt(&testd);
    width = ReadBigInt(&testd);
    height = ReadBigInt(&testd);

    // set up vector sizes
    testData.data = std::vector<float>();
    testLabels.data = std::vector<float>(imagenum);
    testData.data.reserve(imagenum*width*height);

    testData.rows = imagenum;
    testData.cols = width*height;
  
    testLabels.rows = imagenum;
    testLabels.cols = 1;

    // parse out test data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        testd.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        testData.data.insert(testData.data.end(), floatdata.begin(), floatdata.end());

        // get label for data
        char byte;
        testl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        testLabels.data[i] = label;
    }

    trainData.rows = originalData.rows;
    trainData.cols = originalData.cols;

    // size in augmentations
    trainData.rows += originalData.rows*args[Y_ROT_VARIANTS].as<size_t>(Y_ROT_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_SCALE_VARIANTS].as<size_t>(Y_SCALE_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_SHEAR_VARIANTS].as<size_t>(Y_SHEAR_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_ELASTIC_VARIANTS].as<size_t>(Y_ELASTIC_VAR_DEFAULT);
    trainLabels.rows = originalData.rows;

    // reserve size for augmentations
    trainData.data = std::vector<float>(trainData.rows*trainData.cols, 0.0f);
    trainLabels.data = std::vector<float>(trainLabels.rows*trainLabels.cols, 0.0f);

    // set dataset dimensions
    dims = std::vector<size_t>(2, 28);

    size_t a_idx = base_samples;
    std::mt19937 rd(SEED+50);


    // add rotation variants
    if (args[Y_ROTATION]) {
        float rot = args[Y_ROTATION].as<float>();
        float mrot = args[Y_MIN_ROTATION].as<float>(Y_MIN_ROTATION_DEFAULT);

        std::uniform_real_distribution<float> gen(-rot, rot);
        size_t samples = args[Y_ROT_VARIANTS].as<size_t>(Y_ROT_VAR_DEFAULT);

        // generate randomly rotated images of test dataset
        for (size_t i = 0; i < base_samples; i++) {
            for (size_t j = 0; j < samples; j++) {
                float deg = gen(rd);
                deg += deg < 0.0f ? -mrot : mrot;

                RotateImage(&originalData[i*originalDataCols], &originalData[a_idx*originalDataCols], width, height, deg);
                originalLabels[a_idx] = originalLabels[i];

                a_idx++;
            }
        }
    }

    // add scale variants
    if (args[Y_SCALE]) {
        float scale = args[Y_SCALE].as<float>();
        float mscale = args[Y_MIN_SCALE].as<float>(Y_MIN_SCALE_DEFAULT);

        std::uniform_real_distribution<float> gen(1.0f-scale, 1.0f+scale);
        size_t samples = args[Y_SCALE_VARIANTS].as<size_t>(Y_SCALE_VAR_DEFAULT);

        for (size_t i = 0; i < base_samples; i++) {
            for (size_t j = 0; j < samples; j++) {
                float rscale = gen(rd);
                rscale += rscale < 1.0f ? -mscale : mscale;

                ScaleImage(&originalData[i*originalDataCols], &originalData[a_idx*originalDataCols], width, height, rscale);
                originalLabels[a_idx] = originalLabels[i];

                a_idx++;
            }
        }
    }

    // add shear variants
    if (args[Y_SHEAR]) {
        float shear = args[Y_SHEAR].as<float>();
        float mshear = args[Y_MIN_SHEAR].as<float>(Y_MIN_SHEAR_DEFAULT);

        std::uniform_real_distribution<float> gen(-shear, shear);
        size_t samples = args[Y_SHEAR_VARIANTS].as<size_t>(Y_SHEAR_VAR_DEFAULT);


        for (size_t i = 0; i < base_samples; i++) {
            for (size_t j = 0; j < samples; j++) {

                float rshear = gen(rd);
                rshear += rshear < 0.0 ? -mshear : mshear;

                ShearImage(&originalData[i*originalDataCols], &originalData[a_idx*originalDataCols], width, height, rshear);
                originalLabels[a_idx] = originalLabels[i];

                a_idx++;
            }
        }    
    }
}

int DataLoader::ReadBigInt(std::ifstream* f) {
    int lint;
    f->read(reinterpret_cast<char*>(&lint), sizeof(int));

    unsigned char* bytes = reinterpret_cast<unsigned char*>(&lint);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);

    return lint;
}

float DataLoader::BilinearSample(const float* image, size_t w, size_t h, float fx, float fy) {
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
void DataLoader::RotateImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float deg) {
    const double rad = deg * M_PI / 180.0;
    const double cos_a = std::cos(rad);
    const double sin_a = std::sin(rad);

    const double cx = width / 2.0;
    const double cy = height / 2.0;

    #pragma omp parallel for collapse(2)
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
void DataLoader::ScaleImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float scale) {
    const float nw = width*scale;
    const float nh = height*scale;

    const float dx = (width-nw)/2.0f;
    const float dy = (height-nh)/2.0f;

    #pragma omp parallel for collapse(2)
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
void DataLoader::ShearImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float shear) {
    float cx =  width  * 0.5f;
    float cy =  height * 0.5f;
    float det = 1.0f - shear * shear;

    #pragma omp parallel for collapse(2)
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
void DataLoader::ElasticDeformImage(const float* __restrict image, float* __restrict out, size_t width, size_t height, float alpha, float sigma) {
    std::vector<float> elasticImage(width*height, 0.0f);

    std::mt19937 rng(SEED+(((uintptr_t)image)%width));
    std::uniform_real_distribution<float> udist(-1.0f, 1.0f);

    // generate displatement fields, ux, uy
    std::vector<float> ux(width*height);
    std::vector<float> uy(width*height);

    for (size_t i = 0; i < width*height; i++) {
        ux[i] = udist(rng);
        uy[i] = udist(rng);
    }

    // build gaussian smoothing
    int krad = std::ceil(3.0f*sigma);
    std::vector<float> k = MakeGaussianKernel(krad, sigma);

    // apply gaussian smoothing
    std::vector<float> uxs = Convolve(ux, width, height, k, krad);
    std::vector<float> uys = Convolve(uy, width, height, k, krad);

    // scale by alpha
    for (size_t i = 0; i < width*height; i++) {
        uxs[i] *= alpha;
        uys[i] *= alpha;
    }

    // map into fixed output
    float cx = width*0.5f;
    float cy = height*0.5f;

    #pragma omp parallel for collapse(2)
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

std::vector<float> DataLoader::MakeGaussianKernel(int rad, float sigma) {
    int size = 2*rad+1;
    std::vector<float> k(size*size);    

    float sum = 0.0f;
    float inv2s2 = 1.0f/(2.0f*sigma*sigma);

    // generate kernel
    for (ssize_t dy = -rad; dy <= rad; dy++) {
        for (ssize_t dx = -rad; dx <= rad; dx++) {
            float v = std::exp(-(dx*dx+dy*dy)*inv2s2);
            k[(dy+rad)*size+(dx+rad)] = v;
            sum += v;
        }
    }

    // normalize
    for (size_t i = 0; i < k.size(); i++) {
        k[i] /= sum;
    }

    return k;
}
std::vector<float> DataLoader::Convolve(const std::vector<float>& f, size_t width, size_t height, const std::vector<float>& k, int rad) {
    int size = 2*rad+1;
    std::vector<float> convolved(width*height, 0.0f);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            float sum = 0.0f;

            for (ssize_t dy = -rad; dy <= rad; dy++) {
                size_t yy = std::clamp((int)y+(int)dy, 0, (int)height-1);

                for (ssize_t dx = -rad; dx <= rad; dx++) {
                    size_t xx = std::clamp((int)x+(int)dx, 0, (int)width-1);
                    sum += f[yy*width+xx] * k[(dy+rad)*size+(dx+rad)];
                }
            }

            convolved[y*width+x] = sum;
        }
    }

    return convolved;
}


float DataLoader::InMandlebrot(double x, double y, size_t it) {
    std::complex<double> c(x, y);
        std::complex<double> z = 0;

        for (size_t i = 0; i < it; i++) {
            z = z * z + c;
            if (std::abs(z) > 2.0) {
                return (1.0 - (1.0 / (((double)i / 50.0) + 1.0)));
            }
        }
        return 1.0f;
}
void DataLoader::ComputeFourier(float* x, size_t series) {
    float xv = x[0];
    float yv = x[1];

    #pragma omp parallel for
    for (size_t i = 0; i < series; i++) {
        x[2+(i*4)] = std::sin(std::pow(xv, i+2));
        x[2+(i*4)+1] = std::cos(std::pow(xv, i+2));

        x[2+(i*4)+2] = std::sin(std::pow(yv, i+2));
        x[2+(i*4)+3] = std::cos(std::pow(yv, i+2));
    }
}

std::string DataLoader::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}
