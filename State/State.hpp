#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../DataLoader/DataLoader.hpp"

/* @brief

*/
struct State {
public:

    std::string modelname;

    YAML::Node config;
    nlohmann::json history;

    State() {}
    
    void Init();
    void SaveInit();

    void Load();

    void Build(bool setweights);
    void Start();

    std::string ModelMetadata() const;
    std::string ModelHistory() const;
    std::string DeleteModel() const;
    std::string ResetModel() const;
    std::string VisualizeModel();
    
    std::string AvailableModels() const;

    // static utils
    static std::string ExpandPath(const std::string& path);
    static bool CreateDir(const std::string& path);
    static bool DirExists(const std::string& path);
    static bool FileExists(const std::string& path); 

    bool ModelExists();
    bool IsValid();
        
private:

    std::string p_workspace;
    std::string p_datasets;
    std::string p_models;

    NeuralNetwork* model;
    DataLoader dataset;
};