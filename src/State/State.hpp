#pragma once
#include "../Supervisor/Supervisor.hpp"

struct State {
    public:

    State() {
        supervisor = new Supervisor();

        workspacePath = ExpandPath("~/.local/share/MLEngine");
        datasetPath = workspacePath+"/Datasets";
        modelPath = workspacePath+"/Models";
    }
    ~State() {
        if (supervisor) { delete supervisor; }
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

    int Load();
    int Build();

    int Train();

    void InitializeSaveLocation() const;

    bool ModelExists() const;
    bool IsValid() const;

    YAML::Node ParseArgs(int argc, char* argv[]);
    void DeleteModel(const std::string& name);

    static std::string ExpandPath(const std::string& path);
    static bool CreateDirectory(const std::string& path);
    static bool DirectoryExists(const std::string& path);
    static bool FileExists(const std::string& path);
};
