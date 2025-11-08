#include "Dataset.hpp"

int Dataset::LoadMNISTStyle(const std::string& name) {
    // training dataset path
    const std::string trainingImages = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-images.idx3-ubyte");
    const std::string trainingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TrainingData/train-labels.idx1-ubyte");

    // testing dataset pat
    const std::string testingImages = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-images.idx3-ubyte");
    const std::string testingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/"+name+"/TestingData/t10k-labels.idx1-ubyte");

    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        return 1;
    }


    traind.close();
    trainl.close();
    testd.close();
    testl.close();
    return 0;
}

int Dataset::LoadMandlebrot() {

}