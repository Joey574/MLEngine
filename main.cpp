#include <iostream>

#include "NeuralNetwork/NeuralNetwork.hpp"
#include "DataLoader/DataLoader.hpp"


int main(int argc, char* argv[]) {
    // validate that datasets exist on the computer


    // validate local storage for models exists

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
        if (strcmp(argv[i], "-m") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            modelname = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-d") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            dataset = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-di") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            dims = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-l") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            loss = argv[i+1];
            i++;
        } else if (strcmp(argv[i], "-mr") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            metric = argv[i+1];
            i++;            
        } else if (strcmp(argv[i], "-e") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            epochs = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "-tf") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            train_for = argv[i+1];
            i++;
        } else if (strcmp(argv[i],"-lr") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            learning_rate = atof(argv[i+1]);
            i++;
        } else if (strcmp(argv[i],"-bs") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            batch_size = atoi(argv[i+1]);
            i++;            
        } else if (strcmp(argv[i],"-vf") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            validation_freq = atoi(argv[i+1]);
            i++;            
        } else if (strcmp(argv[i],"-vs") == 0) {
            if (!(argc > i + 1)) {
                return 1;
            }

            validation_split = atof(argv[i+1]);
            i++;            
        } else {
            return 1;
        }
    }

    NeuralNetwork model;
    if (modelname != "") {
        // attempt to load provided model
    } else {
        // build new model based on passed args
        if (dataset == "" || dims == "" || loss == "" || metric == "") {
            return 1;
        }
    }

    // model built, continue to train
}
