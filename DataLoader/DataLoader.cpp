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
}

int DataLoader::ReadBigInt(std::ifstream* f) {
    int lint;
    f->read(reinterpret_cast<char*>(&lint), sizeof(int));

    unsigned char* bytes = reinterpret_cast<unsigned char*>(&lint);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);

    return lint;
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
