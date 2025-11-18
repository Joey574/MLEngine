#include "State.hpp"

std::string State::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') [[unlikely]] {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}

bool State::CreateDirectory(const std::string& path) {
    std::string fullPath = ExpandPath(path);

    if (!std::filesystem::exists(fullPath)) { 
        return std::filesystem::create_directories(fullPath);
    }

    return std::filesystem::is_directory(fullPath);
}
bool State::DirectoryExists(const std::string& path) {
    std::string fullPath = ExpandPath(path);

    return std::filesystem::exists(fullPath) && std::filesystem::is_directory(fullPath);
}
bool State::FileExists(const std::string& path) {
    return std::filesystem::exists(ExpandPath(path));
}

