cmake_minimum_required(VERSION 3.18)
project(MLEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

set(RELEASE_FLAGS_CXX -march=native -mtune=native -DNDEBUG -Ofast -fopenmp -lopenblas)
set(RELEASE_FLAGS_CUDA -march=native -mtune=native -O3 --extended-lambda)

set(DEBUG_FLAGS_CXX -march=native -O0 -g -DDEBUG -rdynamic -fopenmp -lopenblas)
set(DEBUG_FLAGS_CUDA -march=native -O0 --extended-lambda)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_program(CCACHE_PROGRAM ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)
add_executable(MLEngine ${SOURCE_FILES})

# set flags
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${DEBUG_FLAGS_CXX}>
    $<$<COMPILE_LANGUAGE:CUDA>:${DEBUG_FLAGS_CUDA}>
  )
else()
  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${RELEASE_FLAGS_CXX}>
    $<$<COMPILE_LANGUAGE:CUDA>:${RELEASE_FLAGS_CUDA}>
  )
endif()

# packages
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

target_link_libraries(MLEngine PRIVATE OpenMP::OpenMP_CXX yaml-cpp ${BLAS_LIBRARIES})
target_precompile_headers(MLEngine PRIVATE src/Dependencies/pch.h)


# Include FetchContent module for downloading dependencies
include(FetchContent)

# Fetch the official parrot library from NVlabs/parrot
FetchContent_Declare(
  parrot
  GIT_REPOSITORY https://github.com/NVlabs/parrot.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)

# Make parrot available
FetchContent_MakeAvailable(parrot)

# Add parrot include directory from FetchContent
FetchContent_GetProperties(parrot)
if(parrot_POPULATED)
    include_directories(${parrot_SOURCE_DIR})
    message(STATUS "Using parrot library from: ${parrot_SOURCE_DIR}")

    # Get CCCL from parrot's build and prioritize it over system headers
    # This ensures we use Thrust 3.2.0+ instead of system Thrust 3.0.1
    FetchContent_GetProperties(cccl)
    if(cccl_POPULATED)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR})
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/cub)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/thrust)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/libcudacxx/include)

        # Add compiler flags to prioritize CCCL headers
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/cub")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/thrust")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/libcudacxx/include")

        message(STATUS "Using CCCL (Thrust 3.2.0+) from: ${cccl_SOURCE_DIR}")
    endif()
endif()