cmake_minimum_required(VERSION 3.28)
project(MLEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Optionally enable cuda
option(USE_CUDA "Enable CUDA support (if available)" ON)
include(DetectCUDA)
if(USE_CUDA_FOUND)
  add_compile_definitions(USE_PARROT=1)
  enable_language(CUDA)
endif()

# -----------------------
# Default to a release build
# -----------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

# -----------------------
# Optionally enable ccache
# -----------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# -----------------------
# Grab CXX and CUDA source files
# -----------------------
file(GLOB_RECURSE CXX_SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES CONFIGURE_DEPENDS src/*.cu)

set(SOURCE_FILES ${CXX_SOURCE_FILES})
if(USE_CUDA_FOUND)
  list(APPEND SOURCE_FILES ${CUDA_SOURCE_FILES})
endif()

# -----------------------
# Fetch and build dependencies
# -----------------------
include(FetchContent)
include(OpenBLASFetch)
include(YamlCppFetch)
include(ParrotFetch)

# -----------------------
# Add executable
# -----------------------
add_executable(MLEngine ${SOURCE_FILES})

# -----------------------
# Build flags
# -----------------------
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(DEBUG_FLAGS_CXX -march=native -O0 -g -DDEBUG -rdynamic)
  set(DEBUG_FLAGS_CUDA -march=native -O0 --extended-lambda)

  target_compile_options(MLEngine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${DEBUG_FLAGS_CXX}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_CUDA_FOUND}>>:${DEBUG_FLAGS_CUDA}>
  )
else()
  set(RELEASE_FLAGS_CXX -march=native -mtune=native -flto -DNDEBUG -Ofast -O3)
  set(RELEASE_FLAGS_CUDA -march=native -mtune=native -O3 --extended-lambda)

  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${RELEASE_FLAGS_CXX}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_CUDA_FOUND}>>:${RELEASE_FLAGS_CUDA}>
  )
endif()

# Precompiled header
target_precompile_headers(MLEngine PRIVATE src/Dependencies/pch.h)

# -----------------------
# Link libraries
# -----------------------
find_package(OpenMP REQUIRED)
target_link_libraries(MLEngine PRIVATE ${BLAS_LIBRARIES} yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX)
