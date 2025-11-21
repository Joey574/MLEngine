#include "State.hpp"

bool State::ModelExists() const {
    return FileExists(path+"/"+name+".model");
}

bool State::IsValid() const {
    return true;
}

YAML::Node State::ParseArgs(int argc, char* argv[]) {
    if (argc < 2) [[unlikely]] {
        return YAML::Node{};
    }

    bool deleteModel = false;

    std::string file = "";
    for (int i = 1; i < argc; i++) {

        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0 && argc > i+1) {
            file = argv[i+1];
        } else if (strcmp(argv[i], "--delete") == 0) {
            deleteModel = true;
        }
    }

    if (file.empty()) {
        std::cerr << "[x] No config passed / found\n";
        return YAML::Node{};
    }

    auto config = YAML::LoadFile(file);

    if (deleteModel) DeleteModel(config[Y_MODELNAME].as<std::string>());

    return config;
}

void State::DeleteModel(const std::string& name) {
    const std::filesystem::path dir = modelPath+"/"+name;
    std::filesystem::remove_all(dir);
    exit(0);
}
