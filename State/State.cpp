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
}
void State::SaveInit() {
    if(!DirExists(p_models+"/"+modelname)) {
        CreateDir(p_models+"/"+modelname);
    }

    // create state.meta file
    std::ofstream file(p_models+"/"+modelname+"/state.meta", std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    nlohmann::json metadata = model->metadata();
    metadata["dataset"] = (int)dataset.type;

    std::string dump = metadata.dump(4).append("\n");
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
void State::Load() {
    
}

void State::Build(const std::string& pdims, const std::string& pactvs, const std::string& pmetric, const std::string& ploss, const std::string& pweight, const std::string& data) {
    dataset = DataLoader::LoadDataset(data, nullptr);
    
    std::vector<NeuralNetwork::ActivationFunctions> activations = NeuralNetwork::parseActvs(pactvs);
    std::vector<size_t> dimensions = NeuralNetwork::parseCompact(pdims);

    NeuralNetwork::LossMetric metric = NeuralNetwork::parseLossMetric(pmetric);
    NeuralNetwork::LossMetric loss = NeuralNetwork::parseLossMetric(ploss);

    NeuralNetwork::WeightInitialization weight = NeuralNetwork::parseWeight(pweight);

    // initialize model with provided options
    model->initialize(dimensions, activations, loss, metric, weight);
}
void State::Start() {

}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
