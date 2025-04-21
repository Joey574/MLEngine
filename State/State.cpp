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
    std::ofstream file(p_models+"/"+modelname+"/state.meta", std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    nlohmann::json metadata = model->Metadata();
    metadata["dataset"] = dataset.name;

    std::string dump = metadata.dump(4).append("\n");
    file.write(dump.c_str(), dump.size());
    file.close();
}

void State::Save(size_t id) {
    std::thread t([&id, this]() {
        
        // open file for save
        std::string filepath = p_models+"/"+modelname+"/"+std::to_string(id)+".model";
        int fd = open(filepath.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);

        // write model data to save
        model->Save(fd);
        close(fd);        
    });

    t.detach();
}
void State::Load() {
    // load state.meta file
    std::ifstream f(p_models+"/"+modelname+"/state.meta");
    nlohmann::json metadata = nlohmann::json::parse(f);
    std::string weight = metadata["weights"];

    // find the most recent model save
    DIR* dir;
    dirent* ent;
    std::string file;
    std::string directory = p_models+"/"+modelname;
    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string f(ent->d_name);

            if (!f.ends_with(".model")) {
                continue;
            }
        }
        closedir(dir);
    }

    Build(metadata["dimensions"], metadata["activations"], metadata["metric"], metadata["loss"], weight, metadata["dataset"]);

    if (weight == "none") {
        int fd = open(file.c_str(), O_RDONLY, 0644);
        model->Load(fd);
        close(fd);
    }  
}

void State::Build(const std::string& pdims, const std::string& pactvs, const std::string& pmetric, const std::string& ploss, const std::string& pweight, const std::string& data) {
    dataset = DataLoader::LoadDataset(data, nullptr);
    
    std::vector<NeuralNetwork::ActivationFunctions> activations = NeuralNetwork::ParseActvs(pactvs);
    std::vector<size_t> dimensions = NeuralNetwork::ParseCompact(pdims);

    NeuralNetwork::LossMetric metric = NeuralNetwork::ParseLossMetric(pmetric);
    NeuralNetwork::LossMetric loss = NeuralNetwork::ParseLossMetric(ploss);

    NeuralNetwork::WeightInitialization weight = NeuralNetwork::ParseWeight(pweight);

    // initialize model with provided options
    model->Initialize(dimensions, activations, loss, metric, weight);
}
void State::Start(size_t batchsize, size_t epochs, float learningrate, int validfreq, float validsplit) {
    nlohmann::json history = model->Fit(dataset, batchsize, epochs, learningrate, validfreq, validsplit, true);

    // model has finished training, parse existing history data if any, and append new training history
    std::cout << history.dump(4) << "\n";

    std::fstream file(p_models+"/"+modelname+"/history.meta");
    
    file.close();   
}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
