#include <filesystem>
#include <unistd.h>
#include <iostream>


std::string expand_path(const std::string& path);
bool create_dir(const std::string& path);
bool dir_exists(const std::string& path);
bool file_exists(const std::string& path);


struct Dataset {
    size_t num_elements;
    float* data;
};

struct State {
    std::string modelname;
    std::string workspace;
    std::string datasets;
    std::string models;

    NeuralNetwork* model;
    Dataset* dataset;

    void Init() {
        workspace = expand_path("~/.local/share/ReconSuite/MLEngine");

        // create / validate workspace for datasets
        datasets = workspace+"/Datasets";
        create_dir(datasets+"/MNIST");
        create_dir(datasets+"/FMNIST");

        // create / validate workspace for models
        models = workspace+"/Models";
        create_dir(models);

        // make sure datasets are on the computer
        int ndatasets = 0;
        ndatasets += file_exists(datasets);

        std::cout << "Datasets found: " << ndatasets << "\n";
    }
};

std::string expand_path(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}

bool create_dir(const std::string& path) {
    std::string expanded_path = expand_path(path);

    if (!std::filesystem::exists(expanded_path)) {
        return std::filesystem::create_directories(expanded_path);
    }

    return std::filesystem::is_directory(expanded_path);
}

bool dir_exists(const std::string& path) {
    return std::filesystem::exists(expand_path(path)) && std::filesystem::is_directory(expand_path(path));
}
bool file_exists(const std::string& path) {
    return std::filesystem::exists(expand_path(path));
}

