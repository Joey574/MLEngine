#include "State.hpp"

void State::Init() {
    model = new NeuralNetwork();
    
    p_workspace = ExpandPath("~/.local/share/ReconSuite/MLEngine");

    // create / validate workspace for models
    p_models = p_workspace+"/Models";
    if (!DirExists(p_models)) {
        CreateDir(p_models);
    }
}
void State::SaveInit() {
    if(!DirExists(p_models+"/"+modelname)) {
        CreateDir(p_models+"/"+modelname);
    }

    // save yaml config if one doesn't exist
    if (!FileExists(p_models+"/"+modelname+"/config.yml")) {
        std::ofstream file(p_models+"/"+modelname+"/config.yml", std::ios::trunc);

        // deep copy config
        std::stringstream ss;
        ss << config;
        YAML::Node basic = YAML::Load(ss.str());

        // remove data that isn't relevent to the config
        basic.remove(Y_EPOCHS);
        basic.remove(Y_BATCHSIZE);
        basic.remove(Y_VALIDFREQ);

        file << basic << "\n";
        file.close();
    }
}

void State::Load() {
    // build with no weights, just setting dimensions, activations etc
    Build(false);

    // attempt to load save from file
    std::string file = p_models+"/"+modelname+"/"+modelname+".model";
    if (FileExists(file)) {
        std::cout << "Loading parameters from file (" << file.substr(file.find_last_of('/')+1) << ")\n";
        std::ifstream ifile(file);
        int err = model->Load(ifile);
        ifile.close();

        if (err) {
            // failed to laod, build model again
            std::cerr << "Failed to load parameters, rebuilding model\n";
            Build(true);
        }        
    } else {
        std::cout << "No save found, rebuilding model\n";
        Build(true);
    }
}
void State::Build(bool setweights) {
    if (dataset.type == Datasets::NONE) {
        dataset = DataLoader::LoadDataset(config);
        config[Y_LAYERS][0][Y_NODES] = dataset.trainDataCols;
    }

    // initialize model with provided options
    model->Initialize(p_models+"/"+modelname+"/", modelname, config, setweights);
}
void State::Start() {

    // parse existing history data, if any
    std::ifstream ifile(p_models+"/"+modelname+"/history.meta");
    nlohmann::json storedhistory;

    // try to parse out existing history data
    try {
        storedhistory = nlohmann::json::parse(ifile);
    } catch (nlohmann::json::parse_error& e) {}
    ifile.close();

    // train model and get new history
    nlohmann::json history = model->Fit(dataset, storedhistory);

    // store new history data in file
    std::ofstream ofileh(p_models+"/"+modelname+"/history.meta", std::ios::trunc);
    ofileh << history.dump(4) << "\n";
    ofileh.close();
}
