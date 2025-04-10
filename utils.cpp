#include <filesystem>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <fcntl.h>

#include "NeuralNetwork/NeuralNetwork.hpp"

std::string expand_path(const std::string& path);
bool create_dir(const std::string& path);
bool dir_exists(const std::string& path);
bool file_exists(const std::string& path);

enum Datasets {
    MNIST=0, FMNIST=1, MANDLEBROT=2
};

struct Dataset {
    size_t rows;
    size_t cols;
    float* data;

    Datasets meta_idx;
};

struct DatasetMeta {
    std::string name;
    bool exists;
};

struct State {
    std::string modelname;
    std::string p_workspace;
    std::string p_datasets;
    std::string p_models;

    NeuralNetwork* model;
    Dataset* dataset;

    std::vector<DatasetMeta> datasetmeta;

    std::atomic<size_t> save_id = 0;

    void Init() {
        p_workspace = expand_path("~/.local/share/ReconSuite/MLEngine");

        // create / validate workspace for datasets
        p_datasets = p_workspace+"/Datasets";
        create_dir(p_datasets+"/MNIST/TrainingData");
        create_dir(p_datasets+"/MNIST/TestingData");

        create_dir(p_datasets+"/FMNIST/TrainingData");
        create_dir(p_datasets+"/FMNIST/TestingData");

        // create and validate workspace for models
        p_models = p_workspace+"/Models";
        create_dir(p_models);

        // collect dataset metadata
        datasetmeta = std::vector<DatasetMeta>(3);

        // mnist dataset
        datasetmeta[Datasets::MNIST].name = "mnist";
        datasetmeta[0].exists = (
            file_exists(p_datasets+"/MNIST/TrainingData/train-images.idx3-ubyte") &&
            file_exists(p_datasets+"/MNIST/TrainingData/train-labels.idx1-ubyte") && 
            file_exists(p_datasets+"/MNIST/TestingData/t10k-images.idx3-ubyte") && 
            file_exists(p_datasets+"/MNIST/TestingData/t10k-labels.idx1-ubyte")
        );

        // fmnist dataset
        datasetmeta[1].name = "fmnist";
        datasetmeta[1].exists = (
            file_exists(p_datasets+"/FMNIST/TrainingData/train-images-idx3-ubyte") &&
            file_exists(p_datasets+"/FMNIST/TrainingData/train-labels-idx1-ubyte") && 
            file_exists(p_datasets+"/FMNIST/TestingData/t10k-images-idx3-ubyte") && 
            file_exists(p_datasets+"/FMNIST/TestingData/t10k-labels-idx1-ubyte")
        );

        // mandlebrot dataset is generated not loaded from disk
        datasetmeta[2].name = "mandlebrot";
        datasetmeta[2].exists = true;
    }

    void SaveInit() {
        // create state.meta file
        int fd = open((p_models+"/"+modelname+"/state.meta").c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);

        /*

            Meta data, at this point all values should be set,
            we should store data regarding which dataset was used,
            network parameters that are set, like dimensions, loss, metric, etc.
            
            should also store historical data that isn't needed later, like weight init, etc.

            [int32]: Dataset
            [int32]: loss
            [int32]: metric
            [int32]: weight_init
            [uint64]: dims_size
            [variable]: dims_string
                        
        
        */

        std::cout << sizeof(Datasets::MNIST) << "\n";
        std::cout << sizeof(datasetmeta.size()) << "\n";

    }

    void Save() {
        if(!dir_exists(p_models+"/"+modelname)) {
            create_dir(p_models+"/"+modelname);
        }

        std::thread t([&]() {           
            
            // open file for save
            std::string filepath = p_models+"/"+modelname+"/"+std::to_string(save_id.fetch_add(1, std::memory_order_relaxed))+".model";
            int fd = open(filepath.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);

            // write model data to save

            
        });

        t.detach();
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
