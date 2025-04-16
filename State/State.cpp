#include "State.hpp"

void State::Init() {
    NeuralNetwork t = NeuralNetwork();
    model = &t;
    
    p_workspace = ExpandPath("~/.local/share/ReconSuite/MLEngine");

    // create / validate workspace for datasets
    p_datasets = p_workspace+"/Datasets";
    CreateDir(p_datasets+"/MNIST/TrainingData");
    CreateDir(p_datasets+"/MNIST/TestingData");

    CreateDir(p_datasets+"/FMNIST/TrainingData");
    CreateDir(p_datasets+"/FMNIST/TestingData");

    // create and validate workspace for models
    p_models = p_workspace+"/Models";
    CreateDir(p_models);

    // collect dataset metadata
    datasetmeta = std::vector<DatasetMeta>(3);

    // mnist dataset
    datasetmeta[Datasets::MNIST].name = "mnist";
    datasetmeta[0].exists = (
        FileExists(p_datasets+"/MNIST/TrainingData/train-images.idx3-ubyte") &&
        FileExists(p_datasets+"/MNIST/TrainingData/train-labels.idx1-ubyte") && 
        FileExists(p_datasets+"/MNIST/TestingData/t10k-images.idx3-ubyte") && 
        FileExists(p_datasets+"/MNIST/TestingData/t10k-labels.idx1-ubyte")
    );

    // fmnist dataset
    datasetmeta[1].name = "fmnist";
    datasetmeta[1].exists = (
        FileExists(p_datasets+"/FMNIST/TrainingData/train-images-idx3-ubyte") &&
        FileExists(p_datasets+"/FMNIST/TrainingData/train-labels-idx1-ubyte") && 
        FileExists(p_datasets+"/FMNIST/TestingData/t10k-images-idx3-ubyte") && 
        FileExists(p_datasets+"/FMNIST/TestingData/t10k-labels-idx1-ubyte")
    );

    // mandlebrot dataset is generated not loaded from disk
    datasetmeta[2].name = "mandlebrot";
    datasetmeta[2].exists = true;
}
void State::SaveInit() {
    if(!DirExists(p_models+"/"+modelname) || !FileExists(p_models+"/"+modelname+"/state.meta")) {
        CreateDir(p_models+"/"+modelname);
    } else {
        // directory and state.meta exist, return
        return;
    }

    // create state.meta file
    std::ofstream file(p_models+"/"+modelname+"/state.meta", std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        error = "big time mess up what is going on bud";
        return;
    }

    nlohmann::json metadata = model->metadata();
    metadata["dataset"] = dataset->type;

    std::string dump = metadata.dump(4);
    file.write(dump.c_str(), dump.size());
    file.close();
}

void State::Save(size_t id) {
    std::thread t([&]() {
        
        // open file for save
        std::string filepath = p_models+"/"+modelname+"/"+std::to_string(id)+".model";
        int fd = open(filepath.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);

        // write model data to save

        
    });

    t.detach();
}
void State::Load(Dataset& dataset) {
    
}

void State::Build(const std::string& pdims, const std::string& pactvs, const std::string& pmetric, const std::string& ploss, const std::string& pweight) {
    std::vector<size_t> dimensions = NeuralNetwork::parse_compact(pdims);
    std::vector<NeuralNetwork::ActivationFunctions> activations = NeuralNetwork::parse_actvs(pactvs);
    NeuralNetwork::LossMetric loss = NeuralNetwork::parse_lm(ploss);
    NeuralNetwork::LossMetric metric = NeuralNetwork::parse_lm(pmetric);
    NeuralNetwork::WeightInitialization weight = NeuralNetwork::parse_weight(pweight);

    model->initialize(dimensions, activations, loss, metric, weight);
}
void State::Start() {

}

bool State::ModelExists() {
    if (DirExists(p_models+modelname) && modelname != "") {
        return true;
    }

    return false;
}