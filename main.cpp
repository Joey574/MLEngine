#include "NeuralNetwork/NeuralNetwork.hpp"
#include "DataLoader/DataLoader.hpp"

// type protos
typedef struct State;

// parsing util protos
void list_datasets(const State& state);

// file util protos
std::string expand_path(const std::string& path);
bool create_dir(const std::string& path);
bool dir_exists(const std::string& path);
bool file_exists(const std::string& path);

struct State {
    public:
        std::string modelname;
        std::string p_workspace;
        std::string p_datasets;
        std::string p_models;
        std::string error = "";
    
        NeuralNetwork* model;
        Dataset* dataset;
    
        std::vector<DatasetMeta> datasetmeta;
    
        std::atomic<size_t> save_id = 0;
    
        void Init() {
            NeuralNetwork t = NeuralNetwork();
            model = &t;
            
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
            if(!dir_exists(p_models+"/"+modelname)) {
                create_dir(p_models+"/"+modelname);
            } else {
                // directory already exists, check state.meta file for config
            }
    
            // create state.meta file
            std::ofstream file(p_models+"/"+modelname+"/state.meta", std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                error = "big time mess up what is going on bud";
                return;
            }
    
            /*
    
                Meta data, at this point all values should be set,
                we should store data regarding which dataset was used,
                network parameters that were used, like dimensions, loss, metric, etc.
                
                should also store historical data that isn't needed later, like weight init, etc.
    
                probably want to move this to json later
                            
            */
    
            nlohmann::json metadata = model->metadata();
            //metadata["dataset"] = dataset->type;
    
            std::string dump = metadata.dump(4);
            file.write(dump.c_str(), dump.size());
            file.close();
        }
    
        void Save(size_t id) {
            std::thread t([&]() {
                
                // open file for save
                std::string filepath = p_models+"/"+modelname+"/"+std::to_string(id)+".model";
                int fd = open(filepath.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    
                // write model data to save
    
                
            });
    
            t.detach();
        }
        void Load() {
    
        }
    
        void Build(const std::string& pdims, const std::string& pactvs, const std::string& pmetric, const std::string& ploss, const std::string& pweight, Dataset& dataset) {
            std::vector<size_t> dimensions = NeuralNetwork::parse_compact(pdims);
            std::vector<NeuralNetwork::ActivationFunctions> activations = NeuralNetwork::parse_actvs(pactvs);
            NeuralNetwork::LossMetric loss = NeuralNetwork::parse_lm(ploss);
            NeuralNetwork::LossMetric metric = NeuralNetwork::parse_lm(pmetric);
            NeuralNetwork::WeightInitialization weight = NeuralNetwork::parse_weight(pweight);
    
            model->initialize(dimensions, activations, loss, metric, weight);
        }
    
    private:
    
};


int main(int argc, char* argv[]) {
    State state;
    state.Init();

    // dataset args
    std::string dataset = "";
    std::vector<std::string> datasetargs;
    std::string dims = "";
    std::string actvs = "";
    std::string loss = "";
    std::string metric = "";
    std::string weight = "";

    // training args
    size_t epochs = 1;
    std::string train_for = "";
    float learning_rate = 0.1;
    size_t batch_size = 500;
    int validation_freq = -1;
    float validation_split = 0.0f;

    // flags
    bool listdatasets;

    CLI::App app{"App description ig?"};
    app.add_option("-m,--model", state.modelname, "loads model from disk (if model doesn't exist new model will be made)");
    app.add_option("-d,--dataset", dataset, "trains on the given dataset");
    app.add_option("-g,--datasetargs", datasetargs, "args for generating the dataset if applicable")->delimiter(',');
    app.add_option("-i,--dimensions", dims, "sets model dimensions");
    app.add_option("-l,--loss", loss, "trains model with given loss algorithm");
    app.add_option("-r,--metric", metric, "evaluates model with given metric");

    app.add_option("-e,--epochs", epochs, "number of epochs to train for")->default_val(1);
    app.add_option("-t,--trainfor", train_for, "length of time to train for, if both epochs and trainfor are set, epochs are used");
    app.add_option("-a,--learningrate", learning_rate, "learning rate to use for training")->default_val(0.1);
    app.add_option("-b,--batchsize", batch_size, "batch size to use for training")->default_val(500);
    app.add_option("-f,--validationfreq", validation_freq, "how often to validate the model in epochs")->default_val(-1);
    app.add_option("-s,--validationsplit", validation_split, "percentage (0-1) of dataset to use as validation set if one isn't provided")->default_val(0.0);

    app.add_flag("--ld", listdatasets, "lists available datasets and some metadata");
    CLI11_PARSE(app, argc, argv);


    if (listdatasets) {
        list_datasets(state);
    }

    Dataset data;

    if (dir_exists(state.p_models+state.modelname) && state.modelname != "") {
        state.Load();
    } else {
        // build new model based on passed args
        if (dataset == "" || dims == "" || actvs == "" || loss == "" || metric == "" || state.modelname == "") {
            std::cout << app.help();
            exit(1);
        }

        Dataset data = DataLoader::LoadDataset(dataset, nullptr);
        state.Build(dims, actvs, metric, loss, weight, data);
    }

    // initialize save location and prep meta data
    state.SaveInit();
    if (state.error != "") {
        std::cerr << state.error;
    }

    // model built, start training
}

// parsing utils
void list_datasets(const State& state) {
    for (size_t i = 0; i < state.datasetmeta.size(); i++) {
        std::cout << state.datasetmeta[i].name << ": " << state.datasetmeta[i].exists << "\n";
    }

    exit(0);
}


// file utils
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
