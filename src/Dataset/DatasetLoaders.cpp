#include "Dataset.hpp"

int Dataset::LoadMNISTStyle(const std::string& name) {
    // training dataset path
    const std::string trainingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-images.idx3-ubyte");
    const std::string trainingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    const std::string testingImagesFile = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-images.idx3-ubyte");
    const std::string testingLabelsFile = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-labels.idx1-ubyte");

    std::ifstream traind(trainingImagesFile, std::ios::binary);
    std::ifstream trainl(trainingLabelsFile, std::ios::binary);
    std::ifstream testd(testingImagesFile, std::ios::binary);
    std::ifstream testl(testingLabelsFile, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        return 1;
    }

    // discard magic number and other irelevant data
    ReadBigInt(&trainl);
    ReadBigInt(&trainl);
    ReadBigInt(&traind);
    size_t elements = ReadBigInt(&traind);
    size_t width = ReadBigInt(&traind);
    size_t height = ReadBigInt(&traind);

    trainingData = Tensor<float>(width, height, elements);
    trainingLabels = Tensor<float>(elements);

    // discard magic number and other irelevant data
    ReadBigInt(&testl);
    ReadBigInt(&testl);
    ReadBigInt(&testd);
    elements = ReadBigInt(&testd);
    width = ReadBigInt(&testd);
    height = ReadBigInt(&testd);

    testingData = Tensor<float>(width, height, elements);
    testingLabels = Tensor<float>(elements);

    traind.close();
    trainl.close();
    testd.close();
    testl.close();
    return 0;
}

int Dataset::LoadMandlebrot() {

}
