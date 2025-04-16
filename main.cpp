#include "NeuralNetwork/NeuralNetwork.hpp"
#include "DataLoader/DataLoader.hpp"
#include "State/State.hpp"


// parsing util protos
void list_datasets(const State& state);

int main(int argc, char* argv[]) {
    State state;
    state.Init();

    // dataset args
    std::string dataset = "";
    std::vector<std::string> datasetargs;
    std::string dims = "";
    std::string actvs = "";
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
    bool listdatasets;

    CLI::App app{"MLEngine (0.0)"};
    app.add_option("-m,--model", state.modelname, "loads model from disk (if model doesn't exist new model will be made)");
    app.add_option("-d,--dataset", dataset, "trains on the given dataset");
    app.add_option("-g,--datasetargs", datasetargs, "args for generating the dataset if applicable")->delimiter(',');
    app.add_option("-i,--dimensions", dims, "sets model dimensions");
    app.add_option("-v,--activations", actvs, "activation functions to use");
    app.add_option("-l,--loss", loss, "trains model with given loss algorithm");
    app.add_option("-r,--metric", metric, "evaluates model with given metric");
    app.add_option("-w,--weight", weight, "what weight initialization tech to use");

    app.add_option("-e,--epochs", epochs, "number of epochs to train for")->default_val(1);
    app.add_option("-t,--trainfor", train_for, "length of time to train for, if both epochs and trainfor are set, epochs are used");
    app.add_option("-a,--learningrate", learning_rate, "learning rate to use for training")->default_val(0.1);
    app.add_option("-b,--batchsize", batch_size, "batch size to use for training")->default_val(500);
    app.add_option("-f,--validationfreq", validation_freq, "how often to validate the model in epochs")->default_val(-1);
    app.add_option("-s,--validationsplit", validation_split, "percentage (0-1) of dataset to use as validation set if one isn't provided")->default_val(0.0);

    app.add_flag("--ld", listdatasets, "lists available datasets and some metadata");
    CLI11_PARSE(app, argc, argv);


    if (listdatasets) {
        list_datasets(state);
    }

    Dataset data{};
    if (state.ModelExists()) {
        state.Load(data);
    } else {
        // build new model based on passed args
        if (dataset == "" || dims == "" || actvs == "" || loss == "" || metric == "" || state.modelname == "") {
            std::cout << app.help();
            exit(1);
        }

        data = DataLoader::LoadDataset(dataset, nullptr);
        state.Build(dims, actvs, metric, loss, weight);
    }

    // initialize save location and prep meta data
    state.SaveInit();
    if (state.error != "") {
        std::cerr << state.error;
    }

    // model built, start training
    state.Start();
}

// parsing utils
void list_datasets(const State& state) {
    for (size_t i = 0; i < state.datasetmeta.size(); i++) {
        std::cout << state.datasetmeta[i].name << ": " << state.datasetmeta[i].exists << "\n";
    }

    exit(0);
}
