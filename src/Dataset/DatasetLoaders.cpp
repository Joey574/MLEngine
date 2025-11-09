#include "Dataset.hpp"

int Dataset::LoadMNISTStyle(const std::string& name) {
    // training dataset path
    const std::string trainingImages = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-images.idx3-ubyte");
    const std::string trainingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    const std::string testingImages = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-images.idx3-ubyte");
    const std::string testingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-labels.idx1-ubyte");

    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        return 1;
    }

    // discard magic number and other irelevant data
    ReadBigInt(&trainl);
    ReadBigInt(&trainl);
    ReadBigInt(&traind);
    elements = ReadBigInt(&traind);
    size_t width = ReadBigInt(&traind);
    size_t height = ReadBigInt(&traind);

    data = Tensor<float>(elements, width*height);
    labels = Tensor<float>(elements);

    traind.close();
    trainl.close();
    testd.close();
    testl.close();
    return 0;
}

int Dataset::LoadMandlebrot() {

}
