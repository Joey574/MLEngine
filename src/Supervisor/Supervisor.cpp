#include "Supervisor.hpp"

/// @brief Defines internal data for members, does not build model
/// @return 0 for success
int Supervisor::Define(YAML::Node& config, std::string& path, std::string& name) {
    assert(!(defined || built));
    this->config = &config;
    this->path   = path;
    this->name   = name;

    // Store basic training configuration data
    trainingConfig.epochs         = (config)[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
    trainingConfig.batchSize      = (config)[Y_BATCHSIZE].as<size_t>(Y_BATCH_DEFAULT);
    trainingConfig.scoreFrequency = (config)[Y_VALIDFREQ].as<int>(Y_VALID_DEFAULT);

    int code = 0;
    code += dataset->Define(config);
    trainingConfig.testSize = dataset->TestingSamples();

    code += model->Define(config, *dataset, trainingConfig);

    defined = true;
    return code;
}

/// @brief Builds model based on internal data from members
/// @return 0 for success
int Supervisor::Build() {
    assert(defined && !built);

    int code = 0;
    code += dataset->Build();
    code += model->Build();

    built = true;
    return code;
}

/// @brief Trains the model based on previously defined config
/// @param history Existing model history
/// @return Updated model history
nlohmann::json Supervisor::Train(nlohmann::json& history) {
    assert(defined && built);

    const size_t iterations = ((*dataset).TrainingSamples() + trainingConfig.batchSize - 1) / trainingConfig.batchSize;

    for (size_t e = 0; e < trainingConfig.epochs && KEEPRUNNING; e++) {

        std::cout << "[i] " << e;
        for (size_t i = 0; i < iterations && KEEPRUNNING; i++) {
            size_t startElement = i * trainingConfig.batchSize;
            size_t numElements  = std::min((*dataset).TrainingSamples() - startElement, trainingConfig.batchSize);

            model->Forward(startElement, numElements);
            model->Backward(startElement, numElements);
        }

        if (trainingConfig.scoreFrequency > 0 && e % trainingConfig.scoreFrequency == 0 && KEEPRUNNING) {
            Score score = model->Validate();
            std::cout << ": " << score.GetScore();

            if (score.IsBetterThan(bestScore)) {
                bestScore = score;
                Save();
            }
        }
        std::cout << "\n";
    }

    return history;
}

int Supervisor::Save() const {
    assert(!path.empty() && !name.empty());
    assert(defined && built);

    const std::string file = path + "/" + name + ".model";
    std::ofstream f(file, std::ios::trunc);
    if (!f.is_open()) {
        return 1;
    }

    int code = model->Save(f);
    f.close();

    code += SaveOptimizers();
    return code;
}
int Supervisor::Load() {
    assert(!path.empty() && !name.empty());
    assert(defined && built);

    std::string file = path + "/" + name + ".model";
    std::ifstream f(file);
    if (!f.is_open()) {
        return 1;
    }

    int code = model->Load(f);
    f.close();

    code += LoadOptimizers();
    return code;
}

int Supervisor::SaveOptimizers() const {
    assert(!path.empty() && !name.empty());
    assert(defined && built);

    std::string file = path + "/" + name + ".optimizers";
    std::ofstream f(file);
    if (!f.is_open()) {
        return 1;
    }

    int err = model->SaveOptimizers(f);
    f.close();
    return err;
}
int Supervisor::LoadOptimizers() {
    assert(!path.empty() && !name.empty());
    assert(defined && built);

    std::string file = path + "/" + name + ".optimizers";
    std::ifstream f(file);
    if (!f.is_open()) {
        return 1;
    }

    int err = model->LoadOptimizers(f);
    f.close();
    return err;
}
