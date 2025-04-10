#include "NeuralNetwork/NeuralNetwork.hpp"
#include "DataLoader/DataLoader.hpp"
#include "utils.cpp"


void display_usage() {
    std::cout << 
    "MLEngine (0.0)\n"
    "Usage: MLEngine [Model Args] [Training Args]\n"
    "\tModel Options:\n\t\t"
    "-m model\t\tloads model from disk (if model doesn't exist, this is used as the models name)\n\t\t"
    "-d dataset\t\ttrains on the given dataset\n\t\t"
    "-di dimensions\t\tsets model dimensions\n\t\t"
    "-l loss\t\t\ttrains model with given loss algorithm\n\t\t"
    "-mr metric\t\toutputs evaluation with given metric\n"
    "\tTraining Options:\n\t\t"
    "-e epochs\t\tnumber of epochs to train for, default 1\n\t\t"
    "-tf train for\t\tlength of time to train for, if both -e and -tf are passed -e is used\n\t\t"
    "-lr learning rate\tlearning rate to use throughout training, default 0.1\n\t\t"
    "-bs batch size\t\tbatch size to use throughout training, default 500\n\t\t"
    "-vf validation freq.\thow often to validate the model in epochs, default never\n\t\t"
    "-vs validation split\tpercentage (0-1) of dataset to use for validation if one isn't provided, default 0\n"
    "\tUtilities:\n\t\t"
    "-ld\t\tlists datasets and basic metadata"
    "\n"
    ;

    exit(1);
}
void list_datasets(const State& state) {
    for (size_t i = 0; i < state.datasetmeta.size(); i++) {
        std::cout << state.datasetmeta[i].name << ": " << state.datasetmeta[i].exists << "\n";
    }

    exit(0);
}


int main(int argc, char* argv[]) {
    State state;
    NeuralNetwork model;
    state.model = &model;
    state.Init();

    /*

    -m model                - model to load                                 | Default | none


    ==== not needed if model is passed =====
    -d dataset              - dataset to use                                | Default | use model value
    -di dimensions          - dimensions of the model                       | Default | use model value
    -l loss                 - loss to use for training                      | Default | use model value
    -mr metric               - metric to grade on                           | Default | use model value


    ==== required regardless ====
    -e epochs               - epochs to train for                           | Default | 1
    -tf trainfor            - time to train for, exlusive w epochs          | Default | not used
    -lr learningrate        - learning rate for training                    | Default | 0.1
    -bs batchsize           - batch size of data                            | Default | 500
    -vf validationfreq      - how often we test the model,                  | Default | never
    -vs validationsplit     - how to split the dataset for valid. data      | Default | 0


    ==== utilities, won't train model ====
    -ld                     - lists datasets and relevent metadata

    
    Thinking about changing how we do model initialization, maybe worth just setting the values directly?
    Wouldn't have to store a bunch of local variables and could directly grab from NN and State to check if
    they were set properlly

    Need to start thinking about making this a standalone clt, args should be passed on the terminal
    Workspace and dataset storage should be stored in recon suite local storage? probably
    Training should encompass just iterating through a number of epochs on a dataset for a given model
    If that model doesn't exist, we create a new one with the given dimensions

    Support compact dimension styling like:
    784-128x6-10
    784x1-128-10x1
    
    */

    std::string modelname = "";

    std::string dataset = "";
    std::string dims = "";
    std::string loss = "";
    std::string metric = "";

    size_t epochs = 1;
    std::string train_for = "";
    float learning_rate = 0.1;
    size_t batch_size = 500;
    int validation_freq = -1;
    float validation_split = 0.0f;

    
    // fun time arg parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 ||
            strcmp(argv[i], "-h") == 0) {
                display_usage();
            }

        if (strcmp(argv[i], "-m") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            state.modelname = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-d") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            dataset = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-di") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            dims = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-l") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            loss = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-mr") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            metric = argv[i+1];
            i++;            
        } else if (strcmp(argv[i], "-e") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            epochs = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "-tf") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            train_for = argv[i+1];
            i++;
        } else if (strcmp(argv[i],"-lr") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            learning_rate = atof(argv[i+1]);
            i++;
        } else if (strcmp(argv[i],"-bs") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            batch_size = atoi(argv[i+1]);
            i++;            
        } else if (strcmp(argv[i],"-vf") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            validation_freq = atoi(argv[i+1]);
            i++;            
        } else if (strcmp(argv[i],"-vs") == 0) {
            if (!(argc > i + 1)) {
                display_usage();
            }

            validation_split = atof(argv[i+1]);
            i++;            
        } else if (strcmp(argv[i], "-ld") == 0) {
            list_datasets(state);
        } else {
            display_usage();
        }
    }

    state.Save();

    NeuralNetwork model;
    if (dir_exists(state.p_models+modelname) && modelname != "") {
        // attempt to load provided model

    } else {
        // build new model based on passed args
        if (dataset == "" || dims == "" || loss == "" || metric == "" || modelname == "") {
            display_usage();
        }
    }

    // model built, continue to train
}

