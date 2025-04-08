#pragma once

#include <vector>
#include <iostream>

class DataLoader {
    public:

    struct Dataset {
        size_t num_elements;
        float* data;
    };

    enum class Datasets {
        MNIST, FMNIST, MANDLEBROT
    };

    static Dataset LoadDataset(Datasets dataset, void* args[]);
    static Dataset LoadMNIST();
    static Dataset LoadFMNIST();
    static Dataset LoadMandlebrot(void* args[]);

};