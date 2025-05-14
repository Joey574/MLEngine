#include "DataLoader.hpp"

/// @brief 
///  Returns the passed dataset as constrained by args
/// @param dataset 
/// @param args 
/// @return
Dataset DataLoader::LoadDataset(const std::string& dataset, const std::vector<std::string>& dsargs) {

    if (dataset == "mnist") {
        return LoadMNIST();
    } else if (dataset == "fmnist") {
        return LoadFMNIST();
    } else if (dataset == "mandlebrot") {
        return LoadMandlebrot(dsargs);
    }

    std::cerr << "Failed to load dataset\n";
    return Dataset{};
}

Dataset DataLoader::LoadMNIST() {
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

    return mnist;
}

Dataset DataLoader::LoadFMNIST() {
    Dataset fmnist(Datasets::FMNIST, "fmnist");

    return fmnist;
}

Dataset DataLoader::LoadMandlebrot(const std::vector<std::string>& args) {
    Dataset mandlebrot(Datasets::MANDLEBROT, "mandlebrot");
    mandlebrot.hasTestData = true;
    mandlebrot.args = args;

    size_t n = atoi(args[0].c_str());
    size_t depth = atoi(args[1].c_str());
    size_t fourier = atoi(args[2].c_str());

    const size_t test_elements = 10000;

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