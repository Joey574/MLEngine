#include "State.hpp"

/// @brief Handles loading and initializng data to / from disk and starts training process
int State::Start(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "No arguments passed\n";
        return 1;
    }

    // parse arguments and return the passed config file path
    std::string configFile = ParseArgs(argc, argv);
    if (configFile.empty()) {
        std::cerr << "No config found\n";
        return 1;
    }

    config = YAML::LoadFile(configFile);
    name = config[Y_MODELNAME].as<std::string>();
    SEED = config[Y_SEED].as<uint64_t>(std::random_device{}());

    path = modelPath+"/"+name;
    InitializeSaveLocation();

    if (ModelExists()) {
        std::cout << "Loading existing model\n";
        Load();
    } else {
        if (!IsValid()) {
            std::cerr << "Invalid model passed\n";
            return 1;
        }

        std::cout << "Creating new model\n";
        Build();
    }

    return Train();
}

/// @brief Initializes model directory on disk
void State::InitializeSaveLocation() const {
    assert(!path.empty());

    if (!DirectoryExists(path)) {
        CreateDirectory(path);
    }
}
