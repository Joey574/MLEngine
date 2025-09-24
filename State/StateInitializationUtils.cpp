#include "State.hpp"

int State::Start(int argc, char* argv[]) {
    std::string configFile;

    CLI::App app{"MLEngine (0.0)\nTrain and save various neural networks"};

    app.add_option("-c,--config", configFile, "path to model's YAML config file");

    CLI11_PARSE(app, argc, argv);

    config = YAML::LoadFile(configFile);
    name = config[Y_MODELNAME].as<std::string>();

    #if defined(__AVX512F__)
        std::cout << "AVX512 Enabled\n";
    #elif defined(__AVX2__) && defined(__FMA__)
        std::cout << "AVX2 Enabled\n";
    #else
        std::cout << "Scalar Enabled\n";
    #endif


    if (config[Y_SEED]) {
        SEED = config[Y_SEED].as<uint64_t>();
    } else {
        std::random_device rd;
        SEED = rd();
        std::cout << "Global Seed: " << SEED << "\n";
    }

    if (ModelExists()) {
        std::cout << "Loading existing model\n";
        Load();
    } else {
        if (!IsValid()) {
            std::cout << app.help();
            return 1;
        }

        std::cout << "Creating new model\n";
        Build();
    }

    path = modelPath+"/"+name;
    InitializeSaveLocation();

    Train();

    return 0;
}

void State::InitializeSaveLocation() const {
    if (!DirectoryExists(path)) {
        CreateDirectory(path);
    }

    std::ofstream file(path+"/config.yml", std::ios::trunc);

    // deep copy the config
    std::stringstream ss;
    ss << config;
    YAML::Node basic = YAML::Load(ss.str());

    basic.remove(Y_EPOCHS);
    basic.remove(Y_BATCHSIZE);
    basic.remove(Y_VALIDFREQ);

    file << basic << "\n";
    file.close();
}
