#include "Definitions/globals.hpp"
#include "State/State.hpp"
#include <csignal>
#include <cstdlib>
#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <thread>

void handleInterupt(int) {
    std::cout << "\nProgram will exit shortly\n";
    KEEPRUNNING = false;
}
void segv(int signum) {
    void* array[30];
    const int size  = backtrace(array, 30);
    char** messages = backtrace_symbols(array, size);

    std::cerr << "Signal " << signum << " stack trace:\n";
    for (int i = 0; i < size; ++i) {
        std::cerr << "[" << i << "] ";
        char* mangled_name = nullptr;
        char* offset_begin = nullptr;
        char* offset_end   = nullptr;

        // Find parens and +address offset inside the string
        for (char* p = messages[i]; *p != 0; ++p) {
            if (*p == '(') {
                mangled_name = p;
            } else if (*p == '+') {
                offset_begin = p;
            } else if (*p == ')') {
                offset_end = p;
                break;
            }
        }

        if ((mangled_name != nullptr) && (offset_begin != nullptr) && (offset_end != nullptr) && mangled_name < offset_begin) {
            *offset_begin = '\0';
            *offset_end   = '\0';
            ++mangled_name;

            int status      = 0;
            char* real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
            if (status == 0) {
                std::cerr << messages[i] << " : " << real_name << "+" << (offset_begin + 1) << "\n";
            } else {
                std::cerr << messages[i] << " : " << mangled_name << "+" << (offset_begin + 1) << "\n";
            }

            free(real_name);
        } else {
            std::cerr << messages[i] << "\n";
        }
    }
    free(messages);
    exit(1);
}

auto main(int argc, char* argv[]) -> int {
    KEEPRUNNING = true;
    signal(SIGINT, handleInterupt);
    signal(SIGSEGV, segv);

    const int ideal = std::thread::hardware_concurrency() / 2;
    omp_set_num_threads(ideal);

#ifdef DEBUG
    std::cout << "[i] Running in DEBUG mode\n";
#endif

    State state;
    return state.Start(argc, argv);
}
