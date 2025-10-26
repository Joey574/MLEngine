cmake_minimum_required(VERSION 3.18)
project(MLEngine LANGUAGES CXX)

# cpp standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

set(RELEASE_FLAGS_CXX
  -march=native -mtune=native
  -flto=auto -fomit-frame-pointer -funroll-loops -fipa-pta
  -fdevirtualize-speculatively 
  -fno-unwind-tables -fno-asynchronous-unwind-tables
  -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
  -Ofast -fopenmp
  -fvect-cost-model=unlimited -DNDEBUG
  -falign-functions=32 -falign-loops=32
  -ffast-math -fno-math-errno -fassociative-math -freciprocal-math
  -fno-signed-zeros -fno-trapping-math -fmodulo-sched
  -fmodulo-sched-allow-regmoves -fpredictive-commoning
  -fhoist-adjacent-loads -ftree-loop-distribution
  -ftree-loop-vectorize -ftree-slp-vectorize -ftree-vectorize
  -Wno-unused-result -frename-registers -fschedule-insns -fschedule-insns2
  -fweb -fipa-cp-clone -floop-nest-optimize -ftree-partial-pre
  -funsafe-loop-optimizations -freorder-blocks-and-partition
  -fno-semantic-interposition -finline-functions -finline-functions-called-once
  -fwhole-program -floop-interchange -floop-block -floop-strip-mine
  -funswitch-loops -fira-loop-pressure -fsched-pressure
  -ftree-loop-distribute-patterns -ftree-loop-im -fsplit-loops -fsplit-paths
)

set(DEBUG_FLAGS_CXX
  -march=native
  -O0 -g -DDEBUG -rdynamic
  -fno-lto -fopenmp
)


# use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp)
add_executable(MLEngine ${SOURCE_FILES})

# set flags
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_options(MLEngine PRIVATE ${DEBUG_FLAGS})
else()
  target_compile_options(MLEngine PRIVATE ${RELEASE_FLAGS})
endif()

# packages
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(MLEngine PRIVATE OpenMP::OpenMP_CXX yaml-cpp)

target_precompile_headers(MLEngine PRIVATE src/Dependencies/pch.h)
