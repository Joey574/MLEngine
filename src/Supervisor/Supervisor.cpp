#include "Supervisor.hpp"

/// @brief Defines internal data for members, does not build model
/// @return 0 for success
int Supervisor::Define(YAML::Node& config, std::string& path, std::string& name) {
    assert(!(defined || built));
    this->config = &config;
    this->path = path;
    this->name = name;

    int code = 0;
    code += dataset->Define(config);
    code += model->Define(config, *dataset);

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

    size_t epochs = (*config)[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
    size_t batchSize = (*config)[Y_BATCHSIZE].as<size_t>(Y_BATCH_DEFAULT);
    int scoreFrequency = (*config)[Y_VALIDFREQ].as<int>(Y_VALID_DEFAULT);


    for (size_t e = 0; e < epochs; e++) {
        size_t startElement = e*batchSize;
        size_t numElements = std::min((*dataset).Samples()-startElement, batchSize);

        model->Forward(startElement, numElements);
        model->Backward(startElement, numElements);

        if (scoreFrequency > 0 && e % scoreFrequency == 0) {
            Score score = model->Validate();

            if (score.IsBetterThan(bestScore)) {
                bestScore = score;
                Save();
            }
        }
    }

    return history;
}

/// @brief Tries to load model
/// @return 0 for success
int Supervisor::Load() {
    assert(!path.empty() && !name.empty());
    assert(defined && !built);

    std::string file = path+"/"+name+".model";
    std::ifstream f(file);
    if (!f.is_open()) {
        return 1;
    }

    int err = model->Load(f);
    f.close();
    built = true;
    return err;
}

/// @brief Saves model
void Supervisor::Save() const {
    assert(!path.empty() && !name.empty());
    assert(defined && built);

    const std::string file = path+"/"+name+".model";
    std::ofstream f(file, std::ios::trunc);
    assert(f.is_open());

    model->Save(f);
    f.close();
}
