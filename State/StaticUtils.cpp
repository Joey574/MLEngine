#include "State.hpp"

std::string State::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}
bool State::CreateDir(const std::string& path) {
    std::string expanded_path = ExpandPath(path);

    if (!std::filesystem::exists(expanded_path)) {
        return std::filesystem::create_directories(expanded_path);
    }

    return std::filesystem::is_directory(expanded_path);
}
bool State::DirExists(const std::string& path) {
    return std::filesystem::exists(ExpandPath(path)) && std::filesystem::is_directory(ExpandPath(path));
}
bool State::FileExists(const std::string& path) {
    return std::filesystem::exists(ExpandPath(path));
}
