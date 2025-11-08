#include "Dataset.hpp"

int Dataset::Define(YAML::Node& config) {
    assert(!(defined || built));
    assert(config[Y_DATASET]);
    this->config = &config;
    
    type = ParseType(config[Y_DATASET].as<std::string>());

    defined = true;
    return 0;
}

int Dataset::Build() {
    assert(defined && !built);
    assert(type != Type::None);
    std::cout << "[i] Building dataset\n";

    switch (type) {
        case Type::MNIST: 
            LoadMNISTStyle("MNIST");
            break;
        case Type::FMNIST:
            LoadMNISTStyle("FMNIST");
            break;
            break;
        case Type::Mandlebrot:
            LoadMandlebrot();
            break;
    }

    built = true;
    return 0;
}
