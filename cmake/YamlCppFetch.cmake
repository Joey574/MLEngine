# YamlCppFetch.cmake
include(FetchContent)
find_package(yaml-cpp QUIET)

if(NOT yaml-cpp_FOUND AND NOT TARGET yaml-cpp::yaml-cpp)
    message(STATUS "Fetching yaml-cpp")
    
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()
