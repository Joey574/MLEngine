#include "State.hpp"

/// @brief Handles loading and initializng data to / from disk and starts training process
int State::Start(int argc, char* argv[]) {

    // parse arguments and return the passed config file path
    std::string configFile = ParseArgs(argc, argv);
    if (configFile.empty()) {
        std::cerr << "No config passed / found\n";
        return 1;
    }

    config = YAML::LoadFile(configFile);
    name = config[Y_MODELNAME].as<std::string>();
    SEED = config[Y_SEED].as<uint64_t>(std::random_device{}());
    path = modelPath+"/"+name;

    // create save directory for model
    InitializeSaveLocation();

    if (ModelExists()) {
        std::cout << "Loading existing model\n";
        Load();
    } else {
        if (!IsValid()) [[unlikely]] {
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
