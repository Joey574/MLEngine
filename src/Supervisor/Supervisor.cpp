#include "Supervisor.hpp"

/// @brief Defines internal data for members, does not build model
int Supervisor::Define(YAML::Node& config) {
    assert(!(defined || built));
    this->config = &config;

    int code = 0;
    code += dataset->Define(config);
    code += model->Define(config, *dataset);

    defined = true;
    return code;
}

/// @brief Builds model based on internal data for members
int Supervisor::Build() {
    assert(defined && !built);

    int code = 0;
    code += dataset->Build();
    code += model->Build();

    built = true;
    return code;
}

nlohmann::json Supervisor::Train(nlohmann::json& history) {
    assert(defined && built);

    size_t epochs = (*config)[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
    int scoreFrequency = (*config)[Y_VALIDFREQ].as<int>(Y_VALID_DEFAULT);

    for (size_t e = 0; e < epochs; e++) {
        model->Forward();
        model->Backward();

        if (scoreFrequency != 0 && e % scoreFrequency == 0) {
            float score = model->Score();
        }
    }

    nlohmann::json h;
    return h;
}

void Supervisor::Load(const std::string& path, const std::string& name) {
    std::string file = path+"/"+name+".model";
}