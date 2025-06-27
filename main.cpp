#include <execinfo.h>
#include <cxxabi.h>
#include <yaml-cpp/yaml.h>

#include "NeuralNetwork/NeuralNetwork.hpp"
#include "State/State.hpp"
#include "MathUtils/MathUtils.hpp"
#include "Tensor/Tensor.hpp"

void displayMeta(State& state) {
    std::cout << state.ModelMetadata() << "\n";
    exit(0);
}
void displayHistory(State& state) {
    std::cout << state.ModelHistory() << "\n";
    exit(0);
}
void displayModels(State& state) {
    std::cout << state.AvailableModels() << "\n";
    exit(0);
}
void deleteModel(State& state) {
    std::cout << state.DeleteModel() << "\n";
    exit(0);
}
void resetModel(State& state) {
    std::cout << state.ResetModel() << "\n";
    exit(0);
}
void visualizeModel(State& state) {
    std::cout << state.VisualizeModel() << "\n";
    exit(0);
}

void handleInterupt(int signum) {
    std::cout << "\nProgram will exit after next epoch\n";
    KEEPRUNNING = false;
}
void segv(int signum) {
    void *array[30];
    int size = backtrace(array, 30);
    char **messages = backtrace_symbols(array, size);

    std::cerr << "Signal " << signum << " stack trace:\n";
    for (int i = 0; i < size; ++i) {
        std::cerr << "[" << i << "] ";
        char *mangled_name = nullptr, *offset_begin = nullptr, *offset_end = nullptr;

        // Find parens and +address offset inside the string
        for (char *p = messages[i]; *p; ++p) {
            if (*p == '(')
                mangled_name = p;
            else if (*p == '+')
                offset_begin = p;
            else if (*p == ')') {
                offset_end = p;
                break;
            }
        }

        if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
            *offset_begin = '\0';
            *offset_end = '\0';
            ++mangled_name;

            int status;
            char *real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
            if (status == 0)
                std::cerr << messages[i] << " : " << real_name << "+" << (offset_begin + 1) << "\n";
            else
                std::cerr << messages[i] << " : " << mangled_name << "+" << (offset_begin + 1) << "\n";

            free(real_name);
        } else {
            std::cerr << messages[i] << "\n";
        }
    }
    free(messages);
    exit(1);
}

int main(int argc, char* argv[]) {
    KEEPRUNNING = true;
    signal(SIGINT, handleInterupt);
    signal(SIGSEGV, segv);
    
    State state;
    state.Init();

    std::string config_file;

    // flags
    bool listhistory = false;
    bool listmeta = false;
    bool listmodels = false;
    bool deletemodel = false;
    bool resetmodel = false;
    bool visualizemodel = false;

    CLI::App app{"MLEngine (0.0)\nTrain and save various neural networks"};

    app.add_option("-c,--config", config_file, "path to model's YAML config file");

    app.add_flag("--meta", listmeta, "list model metadata");
    app.add_flag("--history", listhistory, "list model history");
    app.add_flag("--models", listmodels, "lists available models");
    app.add_flag("--delete", deletemodel, "deletes a given model");
    app.add_flag("--reset", resetmodel, "deletes model history and resets model weights");
    app.add_flag("--visualize", visualizemodel, "visualize memory layout of the network");

    CLI11_PARSE(app, argc, argv);

    if (listmodels) {
        displayModels(state);
    }

    // load configuration
    state.config = YAML::LoadFile(config_file);
    state.modelname = state.config[Y_MODELNAME].as<std::string>();

    // set global seed
    SEED = state.config[Y_SEED].as<uint64_t>(std::random_device{}());


    if (listmeta || listhistory || deletemodel || resetmodel || visualizemodel) {
        if (!state.ModelExists()) {
            std::cerr << "Model not found: " << state.config[Y_MODELNAME].as<std::string>() << "\n";
            exit(1);
        }

        
        if (listmeta) { displayMeta(state); }
        if (listhistory) { displayHistory(state); }      
        if (deletemodel) { deleteModel(state); }
        if (resetmodel) { resetModel(state); }  
        if (visualizemodel) { visualizeModel(state); }
    }

    #if defined(__AVX512F__)
        std::cout << "AVX512 Enabled\n";
    #elif defined(__AVX2__) && defined(__FMA__)
        std::cout << "AVX2 Enabled\n";
    #else
        std::cout << "Scalar Enabled\n";
    #endif

    if (state.ModelExists()) {
        std::cout << "Loading existing model\n";
        state.Load();
    } else {

        // build new model based on passed args
        if (!state.IsValid()) {
            std::cout << app.help();
            exit(1);
        }
        
        std::cout << "Creating new model\n";
        state.Build(true);
    }

    // initialize save location and data
    state.SaveInit();

    // model built, start training
    std::cout << "Training model...\n";
    state.Start();
}
