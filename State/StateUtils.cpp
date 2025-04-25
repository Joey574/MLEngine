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
            models = models.append("\tDataset: ").append(metadata["dataset"]).append("\n");
            models = models.append("\tParameters: ").append(std::to_string((int)metadata["parameters"])).append("\n");
        }
    }

    return models;
}


bool State::ModelExists() {
    if (DirExists(p_models+"/"+modelname) && modelname != "") {
        return true;
    }

    return false;
}
int State::MostRecentSave() {
    // find the most recent model save
    DIR* dir;
    dirent* ent;
    int highest = -1;

    std::string file;
    std::string directory = p_models+"/"+modelname+"/";

    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string f(ent->d_name);
 
            if (!f.ends_with(".model")) {
                continue;
            }
 
            std::string sstr = f.substr(0, f.find_last_of('.'));
 
            if (!sstr.empty() && std::all_of(sstr.begin(), sstr.end(), ::isdigit)) {
                int t = std::atoi(sstr.c_str());
                if (t > highest) {
                    highest = t;
                }
            }
        }
        closedir(dir);
    }

    return highest;
}
