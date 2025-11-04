# OpenBLASFetch.cmake
include(FetchContent)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS QUIET)

if(NOT OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS)
    message(STATUS "Fetching OpenBLAS")

    # Disable OpenBLAS verbosity
    set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
    set(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "" FORCE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")


    # Compile options
    set(NO_SHARED TRUE)
    set(USE_OPENMP TRUE)
    set(ONLY_CBLAS TRUE)
    set(BUILD_COMPLEX OFF)
    set(BUILD_COMPLEX16 OFF)

    FetchContent_Declare(
        openblas
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG develop
        GIT_SHALLOW TRUE
        QUIET
    )

    FetchContent_MakeAvailable(openblas)
    set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
else()
    message(STATUS "Using detected OpenBLAS")
endif()
