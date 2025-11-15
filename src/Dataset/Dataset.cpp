#include "Dataset.hpp"

int Dataset::Define(YAML::Node& config) {
    assert(!(defined || built));
    assert(config[Y_DATASET]);
    this->config = &config;
    
    type = ParseType(config[Y_DATASET].as<std::string>());

    std::cout << "[i] Building dataset\n";

    int code = 0;
    switch (type) {
        case Type::MNIST: 
            code = LoadMNIST();
            break;
        case Type::FMNIST:
            code = LoadFMNIST();
            break;
        case Type::Mandlebrot:
            code = LoadMandlebrot();
            break;
    }

    defined = true;
    return code;
}

int Dataset::Build() {
    assert(defined && !built);
    assert(type != Type::None);

    assert(!trainingData.HasNan() && !trainingLabels.HasNan());
    assert(!testingData.HasNan() && !testingLabels.HasNan());   

    built = true;
    return 0;
}
