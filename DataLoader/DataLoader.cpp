#include "DataLoader.hpp"


/// @brief 
///  Returns the passed dataset as constrained by args
/// @param dataset 
/// @param args 
/// @return
DataLoader::Dataset DataLoader::LoadDataset(Datasets dataset, void* args[]) {
    switch (dataset) {
        case Datasets::MNIST:
            return LoadMNIST();
        case Datasets::FMNIST:
            return LoadFMNIST();
        case Datasets::MANDLEBROT:
            return LoadMandlebrot(args);
        default:
            return Dataset{};
    }
}

DataLoader::Dataset DataLoader::LoadMNIST() {
    Dataset mnist;

    return mnist;
}

DataLoader::Dataset DataLoader::LoadMandlebrot(void* args[]) {
    if (args == nullptr) {
        return Dataset{};
    }

    Dataset mandlebrot;
    size_t num_elements = *(size_t*)args[0];
    size_t max_depth = *(size_t*)args[1];


    std::cout << num_elements << "\n" << max_depth << "\n";
    return mandlebrot;
}