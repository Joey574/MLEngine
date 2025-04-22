#pragma once

class DataLoader {
public:

    static Dataset LoadDataset(const std::string& dataset, void* args[]);
    static Dataset LoadMNIST();
    static Dataset LoadFMNIST();
    static Dataset LoadMandlebrot(void* args[]);

private:
    static int ReadBigInt(std::ifstream* f);
    static std::string ExpandPath(const std::string& path);
};