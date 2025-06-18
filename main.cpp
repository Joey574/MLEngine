#include <execinfo.h>
#include <cxxabi.h>
#include <yaml-cpp/yaml.h>

#include "NeuralNetwork/NeuralNetwork.hpp"
#include "State/State.hpp"

void displayMeta(State& state) {
    std::cout << state.ModelMetadata(state.modelname) << "\n";
    exit(0);
}
void displayHistory(State& state) {
    std::cout << state.ModelHistory(state.modelname) << "\n";
    exit(0);
}
void displayModels(State& state) {
    std::cout << state.AvailableModels() << "\n";
    exit(0);
}
void deleteModel(State& state) {
    std::cout << state.DeleteModel(state.modelname) << "\n";
    exit(0);
}
void resetModel(State& state) {
    std::cout << state.ResetModel(state.modelname) << "\n";
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

    // dataset args
    std::string dataset = "";
    std::vector<std::string> datasetargs;
    std::vector<std::string> dims;
    std::vector<std::string> actvs;
    std::string loss = "";
    std::string metric = "";
    std::string weight = "";

    // training args
    size_t epochs = 1;
    std::string train_for = "";
    float learning_rate = 0.1;
    size_t batch_size = 500;
    int validation_freq = -1;
    float validation_split = 0.0f;

    // flags
    bool listhistory = false;
    bool listmeta = false;
    bool listmodels = false;
    bool deletemodel = false;
    bool resetmodel = false;

    CLI::App app{"MLEngine (0.0)\nTrain and save various neural networks"};

    app.get_formatter()->right_column_width(200);
    auto model_options = app.add_option_group("Model Options", "How the model is built");
    auto training_options = app.add_option_group("Training Options", "How the model will be trained");
    auto flags = app.add_option_group("Flags", "displays information, does not train model");

    model_options->add_option("-c,--config", config_file, "config file path to load");

    flags->add_flag("--meta", listmeta, "list model metadata");
    flags->add_flag("--history", listhistory, "list model history");
    flags->add_flag("--models", listmodels, "lists available models");
    flags->add_flag("--delete", deletemodel, "deletes a given model");
    flags->add_flag("--reset", resetmodel, "deletes model history and resets model weights");

    CLI11_PARSE(app, argc, argv);

    if (listmodels) {
        displayModels(state);
    }

    // load configuration
    state.config = YAML::LoadFile(config_file);
    state.modelname = state.config[Y_MODELNAME].as<std::string>();

    if (listmeta || listhistory || deletemodel || resetmodel) {
        if (!state.ModelExists()) {
            std::cerr << "Model not found: " << state.config[Y_MODELNAME].as<std::string>() << "\n";
            exit(1);
        }

        if (listmeta) { displayMeta(state); }
        if (listhistory) { displayHistory(state); }      
        if (deletemodel) { deleteModel(state); }
        if (resetmodel) { resetModel(state); }  
    }


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

    // initialize save location and prep meta data
    state.SaveInit();

    // model built, start training
    std::cout << "Training model...\n";
    state.Start();
}
