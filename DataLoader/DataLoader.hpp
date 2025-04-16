#include "../Dependencies/pch.h"

#pragma once
class DataLoader {
public:

    static Dataset LoadDataset(const std::string& dataset, void* args[]);
    static Dataset LoadMNIST();
    static Dataset LoadFMNIST();
    static Dataset LoadMandlebrot(void* args[]);

};