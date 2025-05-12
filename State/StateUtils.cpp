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
            models = models.append("\tDataset: ").append(metadata[DATASET]).append("\n");
            models = models.append("\tParameters: ").append(std::to_string((int)metadata[PARAMETERS])).append("\n");
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

    std::ifstream fi(p_models+"/"+m+"/state.meta");
    nlohmann::json meta = nlohmann::json::parse(fi);
    fi.close();

    meta.erase(BESTEVSCORE);

    std::ofstream fo(p_models+"/"+m+"/state.meta", std::ios::trunc);
    std::string dump = meta.dump(4)+"\n";
    fo.write(dump.c_str(), dump.length());
    fo.close();

    return "\"" + m + "\" has been reset";
}

bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
