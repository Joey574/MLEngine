#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../DataLoader/DataLoader.hpp"

#pragma once
struct State {
public:

    std::string modelname;

    State() : dataset(Dataset{Datasets::NONE}) {}
    
    void Init();
    void SaveInit();

    void Save(size_t id);
    void Load();

    void Build(
        const std::string& pdims, 
        const std::string& pactvs, 
        const std::string& pmetric, 
        const std::string& ploss, 
        const std::string& pweight,
        const std::string& data
    );

    void Start();

    // static utils
    static std::string ExpandPath(const std::string& path);
    static bool CreateDir(const std::string& path);
    static bool DirExists(const std::string& path);
    static bool FileExists(const std::string& path); 

    bool ModelExists();
        
private:

    std::string p_workspace;
    std::string p_datasets;
    std::string p_models;

    NeuralNetwork* model;
    Dataset dataset;
};