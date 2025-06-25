cmake_minimum_required(VERSION 3.18)
project(MLEngine LANGUAGES CXX)

# cpp standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# default release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

set(RELEASE_FLAGS
  -march=native -mtune=native
  -flto=auto -fomit-frame-pointer -funroll-loops -fipa-pta
  -fdevirtualize-speculatively
  -O3 -Ofast -fopenmp
  -fvect-cost-model=unlimited -DNDEBUG
  -falign-functions=32 -falign-loops=32
  -ffast-math -fno-math-errno -fassociative-math -freciprocal-math
  -fno-signed-zeros -fno-trapping-math -fmodulo-sched
  -fmodulo-sched-allow-regmoves -fpredictive-commoning
  -fhoist-adjacent-loads -ftree-loop-distribution
  -ftree-loop-vectorize -ftree-slp-vectorize -ftree-vectorize
  -Wno-unused-result -frename-registers -fschedule-insns -fschedule-insns2
  -fweb -fno-semantic-interposition
)

set(DEBUG_FLAGS 
  -O0 -g -DDEBUG -rdynamic
  -fno-omit-frame-pointer
  -fno-lto -mavx2 -mfma
  -fopenmp
)

# use ccache
find_program(CCACHE_PROGRAM ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})


# collect all cpp files
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
  "Activation/*.cpp"
  "DataLoader/*.cpp"
  "Layer/*.cpp"
  "LossMetric/*.cpp"
  "MathUtils/*.cpp"
  "NeuralNetwork/*.cpp"
  "State/*.cpp"
  "Tensor/*.cpp"
  "Optimizer/*.cpp"
  "Scheduler/*.cpp"
  "main.cpp"
)

# executable
add_executable(MLEngine ${SOURCE_FILES})


# set flags
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_options(MLEngine PRIVATE ${DEBUG_FLAGS})
else()
  target_compile_options(MLEngine PRIVATE ${RELEASE_FLAGS})
endif()

# static linking
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -lpthread -lm -ldl")

# yaml package
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(MLEngine PRIVATE OpenMP::OpenMP_CXX yaml-cpp)

target_link_libraries(MLEngine PRIVATE yaml-cpp)

target_precompile_headers(MLEngine PRIVATE Dependencies/pch.h)