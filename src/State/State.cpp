#include "State.hpp"

/// @brief Handles loading and initializng data to / from disk and starts training process
int State::Start(int argc, char* argv[]) {

    // parse arguments and return the passed config file path
    config = ParseArgs(argc, argv);
    name   = config[Y_MODELNAME].as<std::string>();
    SEED   = config[Y_SEED].as<uint64_t>(std::random_device{}());
    path   = modelPath + "/" + name;

    // create save directory for model
    InitializeSaveLocation();

    if (ModelExists()) {
        std::cout << "[i] Loading existing model\n";
        Load();
    } else {
        if (!IsValid()) [[unlikely]] {
            std::cerr << "[x] Invalid model passed\n";
            return 1;
        }

        std::cout << "[i] Creating new model\n";
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

int State::Load() {
    std::string file = path + "/" + name + ".model";
    if (!FileExists(file)) [[unlikely]] {
        std::cerr << "[x] Save file not found\n";
        return 1;
    }

    int code = 0;
    code += supervisor->Define(config, path, name);
    code += supervisor->Build();
    code += supervisor->Load();

    return code;
}
int State::Build() {
    int code = 0;
    code += supervisor->Define(config, path, name);
    code += supervisor->Build();

    return code;
}

int State::Train() {
    std::cout << "[i] Beginning training\n";
    history = supervisor->Train(history);

    // update history
    std::ofstream f(path + "/history.meta", std::ios::trunc);
    assert(f.is_open());

    f << history.dump(4) << "\n";
    f.close();

    return 0;
}
