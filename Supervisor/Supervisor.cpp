#include "Supervisor.hpp"

void Supervisor::Define(YAML::Node& config) {
    this->config = &config;
}

nlohmann::json Supervisor::Train(nlohmann::json& history) {
    size_t epochs = (*config)[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
    int scoreFrequency = (*config)[Y_VALIDFREQ].as<int>(Y_VALID_DEFAULT); 

    for (size_t e = 0; e < epochs; e++) {
        model->Forward();
        model->Backward();

        if (scoreFrequency != 0 && e % scoreFrequency == 0) {
            float score = model->Score();
        }
    }
}
