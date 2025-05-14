#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../DataLoader/DataLoader.hpp"

#pragma once
struct State {
public:

    std::string modelname;

    State() {}
    
    void Init();
    void SaveInit();

    void Load();

    void Build(
        const std::string& pdims, 
        const std::string& pactvs, 
        const std::string& pmetric, 
        const std::string& ploss, 
        const std::string& pweight,
        const std::string& data,
        const std::vector<std::string>& dsargs
    );
    void Build(const nlohmann::json& meta);

    void Start(
        size_t batchsize, 
        size_t epochs, 
        float learningrate, 
        int validfreq, 
        float validsplit
    );

    std::string ModelMetadata(const std::string& m) const;
    std::string ModelHistory(const std::string& m) const;
    std::string DeleteModel(const std::string& m) const;
    std::string ResetModel(const std::string& m) const;
    
    std::string AvailableModels() const;

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