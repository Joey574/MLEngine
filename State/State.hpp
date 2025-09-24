#pragma once
#include "../Supervisor/Supervisor.hpp"

struct State {
public:

    State() {
        supervisor = new Supervisor();

        workspacePath = ExpandPath("~/.local/share/MLEngine");

        modelPath = workspacePath+"/Models";
        datasetPath = workspacePath+"/Datasets";
    }
    ~State() {
        delete supervisor;
    }

    int Start(int argc, char* argv[]);

private:
    std::string name;
    Supervisor* supervisor;

    YAML::Node config;
    nlohmann::json history;

    std::string workspacePath;
    std::string datasetPath;
    std::string modelPath;

    std::string path;

    void Load();
    void Build();

    void Train();

    void InitializeSaveLocation() const;

    bool ModelExists() const;
    bool IsValid() const;

    static std::string ExpandPath(const std::string& path);
    static bool CreateDirectory(const std::string& path);
    static bool DirectoryExists(const std::string& path);
    static bool FileExists(const std::string& path);
};
