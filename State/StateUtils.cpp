#include "State.hpp"

std::string State::ModelMetadata() const {
    if (!FileExists(p_models+"/"+modelname+"/config.yml")) {
        return "[]";
    }

    YAML::Emitter out;
    out << config;

    return std::string(out.c_str());
}
std::string State::ModelHistory() const {
    if (!FileExists(p_models+"/"+modelname+"/history.meta")) {
        return "[]";
    }

    std::ifstream f(p_models+"/"+modelname+"/history.meta");

    nlohmann::json history = nlohmann::json::parse(f);
    return history.dump(4);
}
std::string State::AvailableModels() const {
    // walk model directory and collect models
    DIR* dir;
    dirent* ent;
    std::string models = "";

    if ((dir = opendir(p_models.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string f(ent->d_name);

            if (f == "." || f == "..") {
                continue;
            }
            
            models += f + ":\n";

            // collect basic metadata of the model
            YAML::Node metadata = YAML::LoadFile(p_models+"/"+f+"/config.yml");
            models = models.append("\tDataset: ").append(metadata[Y_DATASET].as<std::string>()).append("\n");
        }
    }

    return models;
}
std::string State::DeleteModel() const {
    const std::filesystem::path dir = p_models+"/"+modelname;

    std::filesystem::remove_all(dir);
    return "\"" + modelname + "\" has been deleted";
}
std::string State::ResetModel() const {
    std::filesystem::remove((p_models+"/"+modelname+"/history.meta"));
    std::filesystem::remove((p_models+"/"+modelname+"/"+modelname+".model"));

    return "\"" + modelname + "\" has been reset";
}
std::string State::VisualizeModel() {
    Build(false);
    
    // initialize model data and pointers
    model->InitializeLayerData(config[Y_BATCHSIZE].as<size_t>(), dataset.testDataRows);
    model->InitializeLayerPointers(config[Y_BATCHSIZE].as<size_t>(), dataset.testDataRows);

    return model->Visualize();
}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
bool State::IsValid() {
    if (config[Y_LAYERS] && config[Y_WEIGHT] && config[Y_MODELNAME] && config[Y_DATASET] && modelname != "") {
        return true;
    }

    return false;
}