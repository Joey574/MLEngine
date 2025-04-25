#include "State.hpp"

void State::Init() {
    model = new NeuralNetwork();
    
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

void State::Save() {
    int id = MostRecentSave() + 1;

    // open file for save
    std::string filepath = p_models+"/"+modelname+"/"+std::to_string(id)+".model";
    int fd = open(filepath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644);

    // write model data to save
    int err = model->Save(fd);
    close(fd);

    if (err) {
        std::cerr << "Failed to save model\n";
    }
}
void State::Load() {
    // load state.meta file
    std::ifstream f(p_models+"/"+modelname+"/state.meta");
    nlohmann::json metadata = nlohmann::json::parse(f);
    std::string weight = metadata["weights"];

    // get the most recent save and format as a file
    int mrs = MostRecentSave();
    std::string file = p_models+"/"+modelname+"/"+std::to_string(mrs)+".model";

    if (mrs != -1) {
        weight = "none";
    }

    Build(metadata["dimensions"], metadata["activations"], metadata["metric"], metadata["loss"], weight, metadata["dataset"]);

    if (weight == "none") {
        std::cout << "Loading parameters from file (" << file.substr(file.find_last_of('/')+1) << ")\n";
        int fd = open(file.c_str(), O_RDONLY, 0644);
        int err = model->Load(fd, NeuralNetwork::ParseWeight(metadata["weights"]));
        close(fd);

        if (err) {
            // build the model again
            std::cerr << "Failed to load parameters, rebuilding model\n";
            Build(metadata["dimensions"], metadata["activations"], metadata["metric"], metadata["loss"], metadata["weights"], metadata["dataset"]);
        }
    } else {
        std::cout << "No save found, rebuilding model\n";
    }
}

void State::Build(const std::string& pdims, const std::string& pactvs, const std::string& pmetric, const std::string& ploss, const std::string& pweight, const std::string& data) {
    if (dataset.type == Datasets::NONE) {
        dataset = DataLoader::LoadDataset(data, nullptr);
    }
    
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

    // model has finished training, parse existing history data, if any, and append new training history
    std::ifstream ifile(p_models+"/"+modelname+"/history.meta");
    nlohmann::json storedhistory;

    // try to parse out existing history data
    try {
        storedhistory = nlohmann::json::parse(ifile);
    } catch (nlohmann::json::parse_error& e) {
        storedhistory = nlohmann::json::array();
    }
    ifile.close();

    // append new history and dump to string
    storedhistory.push_back(history);
    std::string dump = storedhistory.dump(4) + "\n";

    // store new history data in file
    std::ofstream ofile(p_models+"/"+modelname+"/history.meta", std::ios::trunc);
    ofile.write(dump.c_str(), dump.size());
    ofile.close();
}
