#include "State.hpp"

std::string State::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') [[unlikely]] {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}

bool State::CreateDirectory(const std::string& path) {
    std::string fullPath = ExpandPath(path);

    if (!std::filesystem::exists(fullPath)) { 
        return std::filesystem::create_directories(fullPath);
    }

    return std::filesystem::is_directory(fullPath);
}
bool State::DirectoryExists(const std::string& path) {
    std::string fullPath = ExpandPath(path);

    return std::filesystem::exists(fullPath) && std::filesystem::is_directory(fullPath);
}
bool State::FileExists(const std::string& path) {
    return std::filesystem::exists(ExpandPath(path));
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
}
