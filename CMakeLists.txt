cmake_minimum_required(VERSION 3.18)
project(MLEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# optionally enable cuda
option(USE_CUDA "Enable CUDA support (if available)" ON)

# -----------------------
# Default to a release build
# -----------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

# -----------------------
# Detect and enable CUDA if found
# -----------------------
set(USE_CUDA_FOUND FALSE)
if(USE_CUDA)
  find_program(NVCC_FOUND nvcc)
  if(NVCC_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(USE_CUDA_FOUND TRUE)
    message(STATUS "CUDA compiler found: ${NVCC_FOUND} -- enabling CUDA")
  else()
    set(USE_CUDA_FOUND FALSE)
    message(WARNING "CUDA requested (USE_CUDA=ON) but nvcc not found. Building without CUDA.")
  endif()
else()
  message(STATUS "USE_CUDA=OFF -> Building without CUDA support.")
endif()


# -----------------------
# Optionally enable ccache
# -----------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
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


# create executable
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
  set(RELEASE_FLAGS_CXX -march=native -mtune=native -DNDEBUG -Ofast)
  set(RELEASE_FLAGS_CUDA -march=native -mtune=native -O3 --extended-lambda)

  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${RELEASE_FLAGS_CXX}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_CUDA_FOUND}>>:${RELEASE_FLAGS_CUDA}>
  )
endif()

# -----------------------
# Required CPU packages
# -----------------------
find_package(OpenMP REQUIRED)
target_link_libraries(MLEngine PRIVATE OpenMP::OpenMP_CXX)

# Precompiled header
target_precompile_headers(MLEngine PRIVATE src/Dependencies/pch.h)

# Include FetchContent module for downloading dependencies
include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(OPENBLAS_BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS ON)
set(USE_OPENMP ON)

include(OpenBLASFetch)
include(YamlCppFetch)
include(ParrotFetch)
target_link_libraries(MLEngine PRIVATE openblas yaml-cpp::yaml-cpp)
