#include "NeuralNetwork/NeuralNetwork.hpp"
#include "State/State.hpp"

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

    CLI::App app{"MLEngine (0.0)\nTrain and save various neural networks"};

    app.get_formatter()->right_column_width(200);
    auto model_options = app.add_option_group("Model Options", "How the model is built");
    auto training_options = app.add_option_group("Training Options", "How the model will be trained");

    model_options->add_option("-m,--model", state.modelname, "loads model from disk (if model doesn't exist new model will be made)");
    model_options->add_option("-d,--dataset", dataset, "trains on the given dataset");
    model_options->add_option("-g,--dsargs", datasetargs, "args for generating the dataset if applicable")->delimiter(',');
    model_options->add_option("-i,--dimensions", dims, "sets model dimensions");
    model_options->add_option("-v,--activations", actvs, "activation functions to use");
    model_options->add_option("-l,--loss", loss, "trains model with given loss algorithm");
    model_options->add_option("-r,--metric", metric, "evaluates model with given metric");
    model_options->add_option("-w,--weight", weight, "what weight initialization tech to use");

    training_options->add_option("-e,--epochs", epochs, "number of epochs to train for")->default_val(1);
    training_options->add_option("-t,--tf", train_for, "length of time to train for");
    training_options->add_option("-a,--lr", learning_rate, "learning rate to use for training")->default_val(0.1);
    training_options->add_option("-b,--bs", batch_size, "batch size to use for training")->default_val(500);
    training_options->add_option("-f,--vfreq", validation_freq, "how often to validate the model in epochs")->default_val(-1);
    training_options->add_option("-s,--vsplit", validation_split, "percentage (0-1) of dataset to use as validation set if one isn't provided")->default_val(0.0);

    CLI11_PARSE(app, argc, argv);

    if (state.ModelExists()) {
        std::cout << "Loading existing model\n";
        state.Load();
    } else {
        std::cout << "Creating new model\n";

        // build new model based on passed args
        if (dataset == "" || dims == "" || actvs == "" || loss == "" || metric == "" || state.modelname == "") {
            std::cout << app.help();
            exit(1);
        }

        state.Build(dims, actvs, metric, loss, weight, dataset);
    }

    // initialize save location and prep meta data
    state.SaveInit();

    // model built, start training
    std::cout << "Training model...\n";
    state.Start(batch_size, epochs, learning_rate, validation_freq, validation_split);
    state.Save();
}
