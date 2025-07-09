#include "Supervisor.hpp"

void Supervisor::Train() {
    auto trainstart = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n" << config << "\n\n";

    size_t epochs = config[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
    size_t valid_freq = config[Y_VALIDFREQ].as<size_t>(Y_VALID_DEFAULT);

    // first initialization pass, define important data etc
    InitializeNetworks();

    // load optimizers, model weights, etc
    StartNetworks();

    for (size_t e = 0; e < epochs && KEEPRUNNING; e++) {
        auto epochstart = std::chrono::high_resolution_clock::now();

        // apply dataset deformations and shuffle
        m_dataset.Deform(e);

        // advance each network 1 epoch and update weights
        AdvanceNetworks();

        if (e+1 % valid_freq == 0) {
            TestNetworks();
        }
    }

    // save optimizer state, model, etc
    EndNetworks();
    
}