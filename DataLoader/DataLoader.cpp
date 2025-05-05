#include "DataLoader.hpp"

/// @brief 
///  Returns the passed dataset as constrained by args
/// @param dataset 
/// @param args 
/// @return
Dataset DataLoader::LoadDataset(const std::string& dataset, void* args[]) {

    if (dataset == "mnist") {
        return LoadMNIST();
    } else if (dataset == "fmnist") {
        return LoadFMNIST();
    } else if (dataset == "mandlebrot") {
        return LoadMandlebrot(args);
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

Dataset DataLoader::LoadMandlebrot(void* args[]) {
    if (args == nullptr) {
        return Dataset{};
    }
    Dataset mandlebrot(Datasets::MANDLEBROT, "mandlebrot");

    size_t num_elements = *(size_t*)args[0];
    size_t max_depth = *(size_t*)args[1];

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
std::string DataLoader::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}