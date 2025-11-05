#include "State.hpp"

int State::Load() {
    std::string file = path+"/"+name+"/.model";
    if (!FileExists(file)) [[unlikely]] {
        std::cerr << "[x] Save file not found\n";
        return 1;
    }

    int code = 0;
    code += supervisor->Define(config, path, name);
    code += supervisor->Load();

    return code;
}

int State::Build() {
    int code = 0;
    code += supervisor->Define(config, path, name);
    code += supervisor->Build();

    return code;
}

int State::Train() {
    std::cout << "[i] Beginning training\n";
    history = supervisor->Train(history);

    // update history
    std::ofstream f(path+"/history.meta", std::ios::trunc);
    assert(f.is_open());
    
    f << history.dump(4) << "\n";
    f.close();

    return 0;
}
