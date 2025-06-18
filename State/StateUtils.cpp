#include "State.hpp"

std::string State::ModelMetadata(const std::string& m) const {
    if (!FileExists(p_models+"/"+m+"/config.yml")) {
        return "[]";
    }

    YAML::Emitter out;
    YAML::Node con = YAML::LoadFile(p_models+"/"+m+"/config.yml");
    out << con;

    return std::string(out.c_str());
}
std::string State::ModelHistory(const std::string& m) const {
    if (!FileExists(p_models+"/"+m+"/history.meta")) {
        return "[]";
    }

    std::ifstream f(p_models+"/"+m+"/history.meta");

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
std::string State::DeleteModel(const std::string& m) const {
    const std::filesystem::path dir = p_models+"/"+m;

    std::filesystem::remove_all(dir);
    return "\"" + m + "\" has been deleted";
}
std::string State::ResetModel(const std::string& m) const {
    std::filesystem::remove((p_models+"/"+m+"/history.meta"));
    std::filesystem::remove((p_models+"/"+m+"/"+m+".model"));

    return "\"" + m + "\" has been reset";
}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
bool State::IsValid() {
    if (config[Y_LAYERS] && config[Y_WEIGHT] && config[Y_MODELNAME] && config[Y_DATASET]) {
        return true;
    }

    return false;
}