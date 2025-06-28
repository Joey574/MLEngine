#include "DataLoader.hpp"

/// @brief 
///  Returns the passed dataset as constrained by args
/// @param dataset 
/// @param args 
/// @return
Dataset DataLoader::LoadDataset(YAML::Node& config) {
    std::string dataset = config[Y_DATASET].as<std::string>();
    YAML::Node dsargs = config[Y_DATASETARGS];

    if (dataset == "mnist") {
        return LoadMNIST(dsargs);
    } else if (dataset == "fmnist") {
        return LoadFMNIST(dsargs);
    } else if (dataset == "mandlebrot") {
        return LoadMandlebrot(dsargs);
    }

    std::cerr << "Failed to load dataset\n";
    return Dataset{};
}

Dataset DataLoader::LoadMNIST(YAML::Node& args) {
    Dataset mnist(Datasets::MNIST, "mnist");
    mnist.hasTestData = true;

    // training dataset path
    std::string trainingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TrainingData/train-images.idx3-ubyte");
    std::string trainingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    std::string testingImages = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TestingData/t10k-images.idx3-ubyte");
    std::string testingLabels = ExpandPath("~/.local/share/ReconSuite/MLEngine/Datasets/MNIST/TestingData/t10k-labels.idx1-ubyte");

    // open training files
    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);

    ReadBigInt(&trainl);
    ReadBigInt(&trainl);

    ReadBigInt(&traind);
    int imagenum = ReadBigInt(&traind);
    int width = ReadBigInt(&traind);
    int height = ReadBigInt(&traind);

    // set up vector sizes
    mnist.trainData = std::vector<float>();
    mnist.trainLabels= std::vector<float>(imagenum);
    mnist.trainData.reserve(imagenum*width*height);

    mnist.trainDataRows = imagenum;
    mnist.trainDataCols = width*height;

    mnist.trainLabelRows = imagenum;
    mnist.trainLabelCols = 1;

    // parse out training data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        traind.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        mnist.trainData.insert(mnist.trainData.end(), floatdata.begin(), floatdata.end());

        // get label for data
        char byte;
        trainl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        mnist.trainLabels[i] = label;
    }

    traind.close();
    trainl.close();


    // open testing files
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    ReadBigInt(&testl);
    ReadBigInt(&testl);

    ReadBigInt(&testd);
    imagenum = ReadBigInt(&testd);
    width = ReadBigInt(&testd);
    height = ReadBigInt(&testd);

    // set up vector sizes
    mnist.testData = std::vector<float>();
    mnist.testLabels= std::vector<float>(imagenum);
    mnist.testData.reserve(imagenum*width*height);

    mnist.testDataRows = imagenum;
    mnist.testDataCols = width*height;
  
    mnist.testLabelRows = imagenum;
    mnist.testLabelCols = 1;

    // parse out test data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        testd.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        mnist.testData.insert(mnist.testData.end(), floatdata.begin(), floatdata.end());

        // get label for data
        char byte;
        testl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        mnist.testLabels[i] = label;
    }

    testd.close();
    testl.close();
    

    size_t base_samples = mnist.trainDataRows;
    
    if (args[Y_ROTATION]) {
        std::mt19937 rd(SEED+50);
        float rot = args[Y_ROTATION].as<float>();

        std::uniform_real_distribution<float> gen(-rot, rot);
        size_t samples = args[Y_ROT_VARIANTS].as<size_t>(Y_ROT_VAR_DEFAULT);

        // generate randomly rotated images of test dataset
        for (size_t i = 0; i < base_samples; i++) {
            for (size_t j = 0; j < samples; j++) {
                float deg = gen(rd);

                std::vector<float> image = RotateImage(&mnist.trainData[i*mnist.trainDataCols], width, height, deg);

                mnist.trainData.insert(mnist.trainData.end(), image.begin(), image.end());
                mnist.trainLabels.push_back(mnist.trainLabels[i]);
                mnist.trainDataRows++;
                mnist.trainLabelRows++;
            }
        }
    }

    if (args[Y_SCALE]) {
        std::mt19937 rd(SEED+71);
        float scale = args[Y_SCALE].as<float>();

        std::uniform_real_distribution<float> gen(1.0f-scale, 1.0f+scale);
        size_t samples = args[Y_SCALE_VARIANTS].as<size_t>(Y_SCALE_VAR_DEFAULT);

        for (size_t i = 0; i < base_samples; i++) {
            for (size_t j = 0; j < samples; j++) {
                float scale = gen(rd);

                std::vector<float> image = ScaleImage(&mnist.trainData[i*mnist.trainDataCols], width, height, scale);

                mnist.trainData.insert(mnist.trainData.end(), image.begin(), image.end());
                mnist.trainLabels.push_back(mnist.trainLabels[i]);
                mnist.trainDataRows++;
                mnist.trainLabelRows++;
            }
        }
    }

    return mnist;
}
Dataset DataLoader::LoadFMNIST(YAML::Node& args) {
    Dataset fmnist(Datasets::FMNIST, "fmnist");

    return fmnist;
}
Dataset DataLoader::LoadMandlebrot(YAML::Node& args) {
    Dataset mandlebrot(Datasets::MANDLEBROT, "mandlebrot");
    mandlebrot.hasTestData = true;
    mandlebrot.args = args;

    size_t n = args[Y_SAMPLES].as<size_t>(Y_SAMPLE_DEFAULT);
    size_t depth = args[Y_MANDLEDEPTH].as<size_t>(Y_MANDLEDEPTH_DEFAULT);
    size_t fourier = args[Y_FOURIERSERIES].as<size_t>(Y_FOURIER_DEFAULT);

    const size_t test_elements = 10000 > (n*0.1) ? 10000 : n*0.1;

    const double xMin = -2.5;
    const double xMax = 1.0;
    const double yMin = -1.1;
    const double yMax = 1.1;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> xrand(xMin, xMax);
    std::uniform_real_distribution<double> yrand(yMin, yMax);

    mandlebrot.trainDataRows = n;
    mandlebrot.trainDataCols = 2 + (fourier*4);
    mandlebrot.testDataRows = test_elements;
    mandlebrot.testDataCols = 2 + (fourier*4);

    mandlebrot.trainLabelRows = n;
    mandlebrot.trainLabelCols = 1;
    mandlebrot.testLabelRows = test_elements;
    mandlebrot.testLabelCols = 1;

    mandlebrot.trainData = std::vector<float>(mandlebrot.trainDataRows*mandlebrot.trainDataCols);  
    mandlebrot.testData = std::vector<float>(mandlebrot.testDataRows*mandlebrot.testDataCols);

    mandlebrot.trainLabels = std::vector<float>(n);
    mandlebrot.testLabels = std::vector<float>(test_elements);

    // build training dataset
    for (size_t i = 0; i < n; i++) {
        double x = xrand(gen);
        double y = yrand(gen);

        float m = InMandlebrot(x, y, depth);

        mandlebrot.trainData[i*mandlebrot.trainDataCols] = x;
        mandlebrot.trainData[i*mandlebrot.trainDataCols+1] = y;
        mandlebrot.trainLabels[i] = m;

        ComputeFourier(&mandlebrot.trainData[i*mandlebrot.trainDataCols], fourier);
    }

    // build testing dataset
    for (size_t i = 0; i < test_elements; i++) {
        double x = xrand(gen);
        double y = yrand(gen);

        float m = InMandlebrot(x, y, depth);

        mandlebrot.testData[i*mandlebrot.testDataCols] = x;
        mandlebrot.testData[i*mandlebrot.testDataCols+1] = y;
        mandlebrot.testLabels[i] = m;

        ComputeFourier(&mandlebrot.testData[i*mandlebrot.testDataCols], fourier);
    }

    #pragma omp parallel for
    for (size_t c = 0; c < mandlebrot.trainDataCols; c++) {
        // find col min/max
        float min = mandlebrot.trainData[c];
        float max = mandlebrot.trainData[c];
        for (size_t i = 1; i < mandlebrot.trainDataRows; i++) {
            if (mandlebrot.trainData[i*mandlebrot.trainDataCols+c] > max) { max = mandlebrot.trainData[i*mandlebrot.trainDataCols+c]; }
            if (mandlebrot.trainData[i*mandlebrot.trainDataCols+c] < min) { min = mandlebrot.trainData[i*mandlebrot.trainDataCols+c]; }
        }

        if (max <= min) {
            max = min + 1.0f;
        }

        const float range = max-min;

        // normalize training col
        for (size_t i = 0; i < mandlebrot.trainDataRows; i++) {
            const size_t idx = i*mandlebrot.trainDataCols+c;
            mandlebrot.trainData[idx] = (mandlebrot.trainData[idx] - min) / range;
        }

        // normalize testing col
        for (size_t i = 0; i < mandlebrot.testDataRows; i++) {
            const size_t idx = i*mandlebrot.testDataCols+c;
            mandlebrot.testData[idx] = (mandlebrot.testData[idx] - min) / range;
        }
    }
    
    return mandlebrot;
}

int DataLoader::ReadBigInt(std::ifstream* f) {
    int lint;
    f->read(reinterpret_cast<char*>(&lint), sizeof(int));

    unsigned char* bytes = reinterpret_cast<unsigned char*>(&lint);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);

    return lint;
}
std::vector<float> DataLoader::RotateImage(const float* image, size_t width, size_t height, float deg) {
    const double rad = deg * M_PI / 180.0;
    const double cos_a = std::cos(rad);
    const double sin_a = std::sin(rad);

    const double cx = width / 2.0;
    const double cy = height / 2.0;

    std::vector<float>rimage(width*height, 0.0f);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            double x0 = x - cx;
            double y0 = y - cy;

            double src_x =  cos_a * x0 + sin_a * y0 + cx;
            double src_y = -sin_a * x0 + cos_a * y0 + cy;

            int ix = static_cast<int>(std::floor(src_x));
            int iy = static_cast<int>(std::floor(src_y));

            // Nearest-neighbor interpolation
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                rimage[y*width+x] = image[iy*width+x];
            }
        }
    }

    return rimage;
}
std::vector<float> DataLoader::ScaleImage(const float* image, size_t width, size_t height, float scale) {
    std::vector<float> scaled(width * height, 0.0f);

    const float nw = width*scale;
    const float nh = height*scale;

    const float dx = (width-nw)/2.0f;
    const float dy = (height-nh)/2.0f;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            float srcx = std::min(std::max((x-dx)/scale, 0.0f), width - 1.001f);
            float srcy = std::min(std::max((y-dy)/scale, 0.0f), height - 1.001f);

            // bilinear interpolation
            int x0 = std::floor(srcx);
            int y0 = std::floor(srcy);
            int x1 = x0+1;
            int y1 = y0+1;

            float wx = srcx-x0;
            float wy = srcy-y0;

            float v00 = (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) ? image[y0*width+x0] : 0.0f;
            float v01 = (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) ? image[y0*width+x1] : 0.0f;
            float v10 = (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) ? image[y1*width+x0] : 0.0f;
            float v11 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) ? image[y1*width+x1] : 0.0f;

            float value = (1-wy)*((1-wx)*v00+wx*v01) + wy*((1-wx)*v10+wx*v11);

            scaled[y*width+x] = value;
        }
    }

    return scaled;
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