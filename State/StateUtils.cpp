#include "State.hpp"

std::string State::ModelMetadata(const std::string& m) const {
    if (!FileExists(p_models+"/"+m+"/state.meta")) {
        return "{}";
    }

    std::ifstream f(p_models+"/"+m+"/state.meta");

    nlohmann::json meta = nlohmann::json::parse(f);
    return meta.dump(4);
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
            nlohmann::json metadata = nlohmann::json::parse(ModelMetadata(f));
            models = models.append("\tDataset: ").append(metadata["Dataset"]).append("\n");
            models = models.append("\tParameters: ").append(std::to_string((int)metadata["Parameters"])).append("\n");
        }
    }

    return models;
}
std::string State::DeleteModel(const std::string& m) const {
    const std::filesystem::path dir = p_models+"/"+m;

    std::filesystem::remove_all(dir);
    return "Model: \"" + m + "\" has been deleted";
}
std::string State::ResetModel(const std::string& m) const {
    std::filesystem::remove(p_models+m+"/history.meta");
    std::filesystem::remove(p_models+m+"/"+m+".model");

    std::fstream f(p_models+"/state.meta");
    nlohmann::json meta = nlohmann::json::parse(f);

    meta.erase("Best Score Ever");

    f.close();

    return "Model: \"" + m + "\" has been reset";
}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
