#include "State.hpp"

int State::Load() {
    std::string file = path+"/"+name+"/.model";
    if (!FileExists(file)) {
        std::cerr << "Save file not found\n";
        return 1;
    }

    return supervisor->Load(path, name);
}

int State::Build() {
    int code = 0;
    code += supervisor->Define(config);
    code += supervisor->Build();

    return code;
}

int State::Train() {
    std::cout << "Beginning training\n";
    history = supervisor->Train(history);

    // update history
    std::ofstream file(path+"/history.meta", std::ios::trunc);
    
    file << history.dump(4) << "\n";
    file.close();

    return 0;
}
