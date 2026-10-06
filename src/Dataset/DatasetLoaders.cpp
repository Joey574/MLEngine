#include "Dataset.hpp"

int Dataset::LoadMNIST() {
    // training dataset path
    const std::string trainingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TrainingData/train-images.idx3-ubyte");
    const std::string trainingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    const std::string testingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TestingData/t10k-images.idx3-ubyte");
    const std::string testingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TestingData/t10k-labels.idx1-ubyte");

    std::ifstream traind(trainingImagesFile);
    std::ifstream trainl(trainingLabelsFile);

    std::ifstream testd(testingImagesFile);
    std::ifstream testl(testingLabelsFile);

    int code = LoadMNISTStyle(traind, trainl, testd, testl);

    traind.close();
    trainl.close();
    testd.close();
    testl.close();

    return code;
}
int Dataset::LoadFMNIST() {
    // training dataset path
    const std::string trainingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TrainingData/train-images-idx3-ubyte");
    const std::string trainingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TrainingData/train-labels-idx1-ubyte");

    // testing dataset path
    const std::string testingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TestingData/t10k-images-idx3-ubyte");
    const std::string testingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TestingData/t10k-labels-idx1-ubyte");

    std::ifstream traind(trainingImagesFile, std::ios::binary);
    std::ifstream trainl(trainingLabelsFile, std::ios::binary);

    std::ifstream testd(testingImagesFile, std::ios::binary);
    std::ifstream testl(testingLabelsFile, std::ios::binary);

    int code = LoadMNISTStyle(traind, trainl, testd, testl);

    traind.close();
    trainl.close();
    testd.close();
    testl.close();

    return code;
}

int Dataset::LoadMNISTStyle(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl) {
    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        return 1;
    }

    // discard magic number and other irelevant data
    ReadBigInt(&trainl);
    ReadBigInt(&trainl);
    ReadBigInt(&traind);
    size_t elements = ReadBigInt(&traind);
    size_t width    = ReadBigInt(&traind);
    size_t height   = ReadBigInt(&traind);

    trainingData   = Tensor<float>(width, height, elements);
    trainingLabels = Tensor<float>(elements);

    // read training data
    std::vector<uint8_t> bytes(width * height, 0);
    std::vector<float> floatData(bytes.size(), 0);

    for (size_t i = 0; i < elements; i++) {
        // read image and convert to float
        traind.read((char*)bytes.data(), bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatData.begin(), [](uint8_t v) { return (float)v / 255.0f; });

        // copy into tensor
        cblas_scopy(floatData.size(), floatData.data(), 1, &trainingData.Data()[i * width * height], 1);

        // read and insert label
        char byte;
        trainl.read(&byte, 1);
        trainingLabels.Data()[i] = (float)byte;
    }

    // discard magic number and other irelevant data
    ReadBigInt(&testl);
    ReadBigInt(&testl);
    ReadBigInt(&testd);
    elements = ReadBigInt(&testd);
    width    = ReadBigInt(&testd);
    height   = ReadBigInt(&testd);

    testingData   = Tensor<float>(width, height, elements);
    testingLabels = Tensor<float>(elements);

    for (size_t i = 0; i < elements; i++) {
        // read image and convert to float
        testd.read((char*)bytes.data(), bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatData.begin(), [](uint8_t v) { return (float)v / 255.0f; });

        // insert into tensor
        cblas_scopy(floatData.size(), floatData.data(), 1, &testingData.Data()[i * width * height], 1);

        // read and insert label
        char byte;
        testl.read(&byte, 1);
        testingLabels.Data()[i] = (int)byte;
    }

    return 0;
}

int Dataset::LoadMandlebrot() {}
