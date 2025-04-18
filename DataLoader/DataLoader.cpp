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

    return Dataset{Datasets::NONE};
}

Dataset DataLoader::LoadMNIST() {
    Dataset mnist(Datasets::MNIST);

    return mnist;
}

Dataset DataLoader::LoadFMNIST() {
    Dataset fmnist(Datasets::FMNIST);

    return fmnist;
}

Dataset DataLoader::LoadMandlebrot(void* args[]) {
    if (args == nullptr) {
        return Dataset{Datasets::NONE};
    }
    Dataset mandlebrot(Datasets::MANDLEBROT);

    size_t num_elements = *(size_t*)args[0];
    size_t max_depth = *(size_t*)args[1];


    std::cout << num_elements << "\n" << max_depth << "\n";
    return mandlebrot;
}